import pysqlite3
import sys

sys.modules["sqlite3"] = pysqlite3
sys.path.append("../")

from chat_models.OpenAI_Chat import OpenAI_Chat
from chat_models.Client import Client

import json
import multiprocessing
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple
from tqdm import tqdm
from urllib.parse import urlparse


# ============================================================
# GPU / Endpoint Utilities
# ============================================================

def _detect_num_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _build_endpoints(openai_api_base, num_gpus):
    host = "127.0.0.1"
    scheme = "http"

    if openai_api_base:
        if "://" not in openai_api_base:
            openai_api_base = "http://" + openai_api_base
        parsed = urlparse(openai_api_base)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.scheme:
            scheme = parsed.scheme

    start_port = 11434
    return [f"{scheme}://{host}:{start_port + i}/v1" for i in range(num_gpus)]


# ============================================================
# RAG FAILURE CLASSIFICATION
# ============================================================

def _is_hard_rag_failure(err):
    if not err:
        return False
    e = err.lower()
    keywords = [
        "timeout",
        "connection",
        "refused",
        "unreachable",
        "502",
        "503",
        "504",
        "exception",
        "traceback",
    ]
    return any(k in e for k in keywords)


def _is_soft_rag_failure(answer, error):
    if error and not _is_hard_rag_failure(error):
        return True
    if answer is None:
        return True
    if len(answer.strip()) < 30:
        return True
    return False


# ============================================================
# GENERATION WORKER (parallel pool)
# ============================================================

def generation_worker(args):
    (
        item,
        enhanced_query,
        images,
        model_name,
        offline_model,
        openai_api_base,
        max_retries,
        retry_delay,
    ) = args

    item_id = item.get("id")

    print(f"[GEN] Item {item_id}: Starting generation...")

    last_exception = None
    for attempt in range(max_retries):
        try:
            if model_name.startswith("gpt"):
                client = OpenAI_Chat(model_name=model_name, messages=[])
            else:
                client = Client(model_name=offline_model, openai_api_base=openai_api_base, messages=[])

            response = client.chat(prompt=enhanced_query, images=images)
            item[model_name] = response
            item["history"] = client.get_history()

            print(f"[GEN] Item {item_id}: ✓ Generation successful")
            return item

        except Exception as e:
            last_exception = e
            print(f"[GEN] Item {item_id}: Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    print(f"[GEN] Item {item_id}: ✗ Generation failed after retries")
    item[model_name] = -1
    item["generation_error"] = str(last_exception)
    return item


# ============================================================
# RAG WORKER PROCESS
# ============================================================

def rag_worker_process(
    rag_request_q,
    rag_response_q,
    test_model,
    embed_model_name,
    device,
    api_base,
    do_reset_collection: bool,
    rag_status_q,
    crop_dictionary_path: Optional[str],
    enable_query_enrichment: bool,
):
    import asyncio
    from rag_agent.main import MainAgent
    from rag_agent.crop_query_enrichment import CropQueryEnricher

    print(f"[RAG Worker] Starting worker for endpoint: {api_base}")

    # ---------------------------
    # Initialization + READY/FAILED
    # ---------------------------
    try:
        rag_agent = MainAgent(
            test_model=test_model,
            embed_model_name=embed_model_name,
            device=device,
            api_base=api_base,
        )

        if do_reset_collection:
            print(f"[RAG Worker] Endpoint {api_base}: Resetting collection (rank0 only)...")
            try:
                rag_agent.reset_collection()
                print(f"[RAG Worker] Endpoint {api_base}: ✓ Collection reset")
            except Exception as e:
                print(f"[RAG Worker] Endpoint {api_base}: ✗ reset_collection failed: {e}")
                rag_status_q.put(("FAILED", api_base, str(e)))
                return
        else:
            print(f"[RAG Worker] Endpoint {api_base}: Skipping reset_collection (non-rank0)")

        # (Optional hardening) Re-bind collection handle explicitly in non-rank0 workers
        # to avoid any chance of stale collection handles if reset happened elsewhere.
        if not do_reset_collection:
            try:
                rag_agent.collection = rag_agent.client.get_or_create_collection(
                    name="meta-mirage_collection",
                    embedding_function=rag_agent.embedding_function,
                )
            except Exception as e:
                print(f"[RAG Worker] Endpoint {api_base}: ✗ Failed to refresh collection handle: {e}")
                rag_status_q.put(("FAILED", api_base, str(e)))
                return

        rag_runner = rag_agent.main()

        dict_data = None
        if enable_query_enrichment and crop_dictionary_path:
            try:
                with open(crop_dictionary_path, encoding="utf-8") as df:
                    dict_data = json.load(df)
                print(f"[RAG Worker] Loaded crop dictionary from {crop_dictionary_path}")
            except Exception as e:
                print(f"[RAG Worker] Failed to load crop dictionary ({crop_dictionary_path}): {e}")
        crop_enricher = CropQueryEnricher(
            api_base=api_base,
            model=test_model,
            dictionary=dict_data,
            enabled=bool(enable_query_enrichment and dict_data is not None),
        )

        rag_status_q.put(("READY", api_base))

    except Exception as e:
        print(f"[RAG Worker] Initialization failed: {e}")
        rag_status_q.put(("FAILED", api_base, str(e)))
        return

    # ---------------------------
    # Event loop
    # ---------------------------
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    request_count = 0
    RESTART_INTERVAL = 1000

    while True:
        request = rag_request_q.get()
        if request is None:
            print(f"[RAG Worker] Endpoint {api_base}: Shutting down.")
            break

        if len(request) == 4:
            item_id, query, location, attempt = request
        else:
            item_id, query, attempt = request
            location = None
        request_count += 1

        rag_agent.current_location = location
        print(f"[RAG Worker] Item {item_id}: Received (Attempt {attempt})")
        print(f"[RAG Worker] Item {item_id}: Query length = {len(query)} chars")

        if request_count % RESTART_INTERVAL == 0:
            print(f"[RAG Worker] Endpoint {api_base}: Restarting agent after {request_count} requests")
            try:
                rag_agent = MainAgent(
                    test_model=test_model,
                    embed_model_name=embed_model_name,
                    device=device,
                    api_base=api_base,
                )
                # Refresh collection handle after restart as well
                rag_agent.collection = rag_agent.client.get_or_create_collection(
                    name="meta-mirage_collection",
                    embedding_function=rag_agent.embedding_function,
                )
                rag_runner = rag_agent.main()
            except Exception as e:
                print(f"[RAG Worker] Endpoint {api_base}: ✗ Restart failed: {e}")
                rag_response_q.put((item_id, None, str(e), False, api_base, attempt, query))
                continue

        effective_query = crop_enricher.enrich(query)
        try:
            session_id = f"rag_session_{item_id}"
            events = loop.run_until_complete(rag_runner.run_debug(effective_query, session_id=session_id))

            rag_answer = None
            tool_calls = []
            agent_texts = []
            web_search_performed = False

            if isinstance(events, list):
                print(f"[RAG Worker] Item {item_id}: {len(events)} events returned")

                for idx, event in enumerate(events):
                    author = getattr(event, "author", "unknown")
                    print(f"[RAG Worker] Item {item_id}: Event {idx} by {author}")

                    try:
                        calls = event.get_function_calls()
                        if calls:
                            for tc in calls:
                                tool_name = getattr(tc, "name", "unknown")
                                tool_calls.append(tool_name)
                                print(f"[RAG Worker] Item {item_id}: → Tool called: {tool_name}")
                    except Exception:
                        pass

                    if author == "Rag_Agent":
                        if hasattr(event, "content") and hasattr(event.content, "parts"):
                            for part in event.content.parts:
                                if hasattr(part, "text") and part.text:
                                    agent_texts.append(part.text)

                web_search_performed = any("web_search" in t.lower() for t in tool_calls)

                if agent_texts:
                    rag_answer = agent_texts[-1].strip()

            if rag_answer:
                print(f"[RAG Worker] Item {item_id}: ✓ Extracted answer ({len(rag_answer)} chars)")
                rag_response_q.put(
                    (item_id, rag_answer, None, web_search_performed, api_base, attempt, effective_query)
                )
            else:
                print(f"[RAG Worker] Item {item_id}: ✗ No valid answer extracted")
                rag_response_q.put(
                    (item_id, None, "No RAG answer found in response", False, api_base, attempt, effective_query)
                )

        except Exception as e:
            print(f"[RAG Worker] Item {item_id}: ✗ Exception during RAG: {e}")
            rag_response_q.put((item_id, None, str(e), False, api_base, attempt, effective_query))


# ============================================================
# MAIN GENERATE CLASS
# ============================================================

def _resolve_crop_dictionary_path(
    crop_dictionary_path: Optional[str],
    inference_dir: Path,
) -> Tuple[Optional[str], bool]:
    """Return (absolute path or None, whether file exists for enrichment). Empty path disables."""
    if crop_dictionary_path is None or not str(crop_dictionary_path).strip():
        return None, False
    p = Path(crop_dictionary_path)
    if not p.is_absolute():
        p = inference_dir / p
    resolved = p.resolve()
    if resolved.is_file():
        return str(resolved), True
    print(f"[Generate] Crop dictionary not found at {resolved}; query enrichment disabled.")
    return None, False


class Generate:

    def __init__(self, raw_data_file, output_file,
                 model_name="gpt-4o",
                 openai_api_base="",
                 num_processes=None,
                 embed_model_name="BAAI/bge-base-en-v1.5",
                 test_model="Qwen2.5-VL-3B-Instruct",
                 device="None",
                 crop_dictionary_path: Optional[str] = "CropDatabase.json",
                 enable_query_enrichment: bool = True):

        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name.split("/")[-1]
        self.offline_model = model_name
        self.openai_api_base = openai_api_base
        self.num_processes = num_processes or os.cpu_count()

        self.embed_model_name = embed_model_name
        self.test_model = test_model
        self.device = device

        inference_dir = Path(__file__).resolve().parent
        path_resolved, found = _resolve_crop_dictionary_path(crop_dictionary_path, inference_dir)
        self.crop_dictionary_path = path_resolved
        self.enable_query_enrichment = bool(enable_query_enrichment) and found

        self.max_retries = 5
        self.retry_delay = 5
        self.max_rag_attempts = 2
        self.rag_inflight_per_gpu = 2

    def get_prompt(self, item):
        question = item["question"]
        state = (item.get("meta_data_state") or "").strip()
        county = (item.get("meta_data_county") or "").strip()
        location = f"{state}, {county}" if state and county else (state or "")
        user_message = f"[User location: {location}]\n\n{question}" if location else question

        images = item.get("images", [])
        new_images = []
        dir_path = os.path.dirname(os.path.abspath(self.raw_data_file))

        for img in images:
            new_path = os.path.join(dir_path, img)
            if not os.path.exists(new_path):
                print(f"Image path {new_path} does not exist. Skipping.")
                continue
            new_images.append(new_path)

        return {"user": user_message, "images": new_images, "location": location}

    def generate(self):

        with open(self.raw_data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if self.model_name in item and item[self.model_name] not in (-1, None):
                            processed_ids.add(item["id"])
                    except:
                        continue

        items = []
        for idx, item in enumerate(data):
            if "id" not in item:
                item["id"] = f"row_{idx}_{time.time_ns()}"
            if item["id"] not in processed_ids:
                items.append(item)

        print(f"Items to process: {len(items)}")

        ctx = multiprocessing.get_context("spawn")

        num_gpus = _detect_num_gpus() or 1
        endpoints = _build_endpoints(self.openai_api_base, num_gpus)

        rag_request_q = ctx.Queue(maxsize=num_gpus * self.rag_inflight_per_gpu)
        rag_response_q = ctx.Queue()

        rag_status_q = ctx.Queue()

        rag_workers = []

        # Start rank0 first
        rank0_ep = endpoints[0]
        p0 = ctx.Process(
            target=rag_worker_process,
            args=(
                rag_request_q,
                rag_response_q,
                self.test_model,
                self.embed_model_name,
                self.device,
                rank0_ep,
                True,
                rag_status_q,
                self.crop_dictionary_path,
                self.enable_query_enrichment,
            ),
        )
        p0.start()
        rag_workers.append(p0)

        # Wait for rank0 READY
        msg = rag_status_q.get(timeout=300)
        if msg[0] != "READY":
            raise RuntimeError(f"Rank0 RAG worker failed: {msg}")

        print(f"[MAIN] Rank0 READY: {msg[1]}")

        # Now start the rest (non-rank0)
        for ep in endpoints[1:]:
            p = ctx.Process(
                target=rag_worker_process,
                args=(
                    rag_request_q,
                    rag_response_q,
                    self.test_model,
                    self.embed_model_name,
                    self.device,
                    ep,
                    False,
                    rag_status_q,
                    self.crop_dictionary_path,
                    self.enable_query_enrichment,
                ),
            )
            p.start()
            rag_workers.append(p)

        # Wait for all to be READY (optional but recommended)
        ready = 1
        target = len(endpoints)
        while ready < target:
            msg = rag_status_q.get(timeout=300)
            if msg[0] == "READY":
                ready += 1
                print(f"[MAIN] Worker READY: {msg[1]} ({ready}/{target})")
            else:
                raise RuntimeError(f"Worker failed during startup: {msg}")

        time.sleep(1)
        for p, ep in zip(rag_workers, endpoints):
            print(f"[MAIN] RAG worker {ep} alive={p.is_alive()} pid={p.pid}")

        dead = [ep for p, ep in zip(rag_workers, endpoints) if not p.is_alive()]
        if dead:
            print(f"[MAIN] WARNING: Some RAG workers died at startup: {dead}")

        pool = ctx.Pool(processes=self.num_processes)

        pending = {}
        idx = 0
        total = len(items)
        pbar = tqdm(total=total)

        def write(item):
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        def generation_done(item):
            write(item)
            pbar.update(1)

        while idx < total and not rag_request_q.full():
            item = items[idx]
            idx += 1
            prompt = self.get_prompt(item)
            pending[item["id"]] = (item, prompt, 1)
            rag_request_q.put((item["id"], prompt["user"], prompt.get("location"), 1))

        completed = 0

        while completed < total:
            item_id, rag_answer, rag_error, web_flag, endpoint, attempt, effective_query = (
                rag_response_q.get()
            )

            if item_id not in pending:
                continue

            item, prompt, attempts = pending[item_id]
            item["RAG_endpoint"] = endpoint
            item["RAG_attempt"] = attempt
            item["RAG_web_search_performed"] = web_flag

            if rag_error is None and rag_answer and not _is_soft_rag_failure(rag_answer, rag_error):
                enhanced = effective_query + "\n\nadditional context: " + rag_answer
                item["RAG_status"] = "successful"
                item["RAG_used"] = True

            else:
                if _is_hard_rag_failure(rag_error) and attempts < self.max_rag_attempts:
                    print(f"[MAIN] Item {item_id}: Retrying RAG...")
                    pending[item_id] = (item, prompt, attempts + 1)
                    rag_request_q.put((item_id, prompt["user"], prompt.get("location"), attempts + 1))
                    continue

                if _is_soft_rag_failure(rag_answer, rag_error):
                    print(f"[MAIN] Item {item_id}: Soft fail → fallback to original query")
                    enhanced = effective_query
                    item["RAG_status"] = "soft_fail"
                    item["RAG_used"] = False
                else:
                    print(f"[MAIN] Item {item_id}: Hard fail → skipping generation")
                    item["RAG_status"] = "hard_fail"
                    write(item)
                    pbar.update(1)
                    completed += 1
                    del pending[item_id]
                    continue

            del pending[item_id]

            pool.apply_async(
                generation_worker,
                args=((item, enhanced, prompt["images"],
                       self.model_name,
                       self.offline_model,
                       self.openai_api_base,
                       self.max_retries,
                       self.retry_delay),),
                callback=generation_done,
            )

            completed += 1

            while idx < total and not rag_request_q.full():
                item = items[idx]
                idx += 1
                prompt = self.get_prompt(item)
                pending[item["id"]] = (item, prompt, 1)
                rag_request_q.put((item["id"], prompt["user"], prompt.get("location"), 1))

        pool.close()
        pool.join()

        for _ in rag_workers:
            rag_request_q.put(None)
        for p in rag_workers:
            p.join()

        pbar.close()
        print("Processing completed.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", default="gpt-4o")
    parser.add_argument("--openai_api_base", default="")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count())
    parser.add_argument("--embed_model_name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--test_model", default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="None")
    # Crop DB: default CropDatabase.json beside this file; use "" to disable enrichment.
    parser.add_argument(
        "--crop_dictionary_path",
        default="CropDatabase.json",
        help="Crop dictionary JSON path (relative paths resolve from Inference/). Empty string disables.",
    )
    parser.add_argument(
        "--disable_query_enrichment",
        action="store_true",
        help="Skip crop query enrichment even if the dictionary file exists.",
    )
    args = parser.parse_args()

    crop_path = (args.crop_dictionary_path or "").strip() or None

    generator = Generate(
        raw_data_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        openai_api_base=args.openai_api_base,
        num_processes=args.num_processes,
        embed_model_name=args.embed_model_name,
        test_model=args.test_model,
        device=args.device,
        crop_dictionary_path=crop_path,
        enable_query_enrichment=not args.disable_query_enrichment,
    )

    generator.generate()