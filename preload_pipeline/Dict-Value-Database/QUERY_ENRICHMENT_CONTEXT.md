# Context brief: crop-dictionary query enrichment (for implementation planning)

Use this as background for designing and implementing the feature.

---

## Goal

Before the existing RAG pipeline runs (`retrieve` → confidence → optional web → …), run a **preprocessing step**:

1. Take the **user query** (the same string currently sent to the RAG worker: the full `prompt["user"]` from `Generate.get_prompt`, not a stripped question).
2. Load a **crop dictionary** from a **JSON file** (structure aligned with the Dict-Value-Database / crop metadata used elsewhere in the project).
3. Pass **question body + dictionary** to an **LLM** with instructions roughly:
   - If the query references **category-level information** for a crop (e.g. a field that exists in the dictionary for that crop) **but the crop name does not appear** in the query, **inject the crop name** into the query.
   - If there is **no meaningful overlap** between query and dictionary-driven signals, **return the query unchanged**.
4. Treat the **recombined full user string** (prefix + enriched body, or original on failure) as the **only** user query string for the rest of the pipeline (retrieval, tools, final formatting).

Exact prompting and matching rules should be specified during implementation; the product intent is **disambiguation / completion** when the user implies a crop without naming it.

---

## Architectural decision (agreed)

- **Do not** rely on `rag_agent/main.py`’s `if __name__ == "__main__"` block for batch runs. Batch inference uses **`Inference/generate.py`**, which imports `MainAgent` inside **`rag_worker_process`** and calls `rag_runner.run_debug(query, session_id=…)`.
- **Enrichment runs inside each RAG worker process** — **not** a separate subprocess or extra worker. The **same** process: **optional enrichment first**, then **`run_debug`** on the effective query. No second process is started for enrichment; work is split across GPUs by **which worker** handles the job (each worker uses its own `api_base` / endpoint).
- **Preserve the full user prompt from `get_prompt`:** The queued string is authoritative. The **`[User location: …]\n\n` prefix** (when present) must be **preserved exactly** for downstream RAG and generation. Implementation: **deterministically split** `prompt["user"]` into **prefix** and **question body** (e.g. regex or first blank line after the location block); send **body + dictionary** to the crop LLM; **recombine** `effective_query = prefix + enriched_body`. The prefix is **not** something the model should rewrite; only the body is enriched (or the prompt must explicitly forbid changing the prefix if an alternative is ever used).
- Each worker loads the dictionary JSON **once per worker** (e.g. at worker startup or lazy first-use cache inside that process).
- The **same OpenAI-compatible endpoint / model** used for that worker’s RAG stack is used for the enrichment LLM call (same `api_base`, `test_model` pattern as today).
- **Testing entry points** (`Inference/test.py`, `rag_agent/test_standalone.py`) are **out of scope** for now; no requirement to change them initially.
- **Dictionary file placement (runtime):** For runs, the JSON can live under **`Inference/`** (e.g. next to `generate.py`) so paths can be resolved from `Path(__file__).resolve().parent` in `generate.py` and passed into workers (or reconstructed consistently in the worker). The **authoritative** dictionary build still lives under this pipeline; copy or symlink the built JSON into `Inference/` when running batch inference.

---

## Enrichment implementation (new class in `rag_agent/`)

Core logic lives in a **dedicated class** under the **`rag_agent/`** directory (new module), not in `Inference/` as the primary home. Responsibilities:

1. **Split** the full `query` into **prefix** (location block + delimiter) and **question body** using a small, deterministic rule.
2. **One chat completion** to the same **`api_base` / model** as that worker’s RAG, with **dictionary JSON** (passed in or loaded once per worker) + **body** and instructions matching Goal §3.
3. **Parse** the model output to a single string (plain text or small JSON schema).
4. **Recombine:** `effective_query = prefix + enriched_body` when the model returns only the body; or validate a full-string contract if chosen.
5. **On any failure** (timeout, parse error, missing file when enabled): **`effective_query = query`** (original full user string) and log.

**Optional enrichment (two senses):**

- **Configurable:** A flag or empty dictionary path can **disable** enrichment entirely; the worker passes **`query` unchanged** to `run_debug`.
- **Soft failure:** When enrichment is enabled, load/API failures still **fall back** to the original `query`.

---

## Current data flow (relevant parts)

1. **`Generate.get_prompt(item)`** builds `prompt["user"]`, optionally prefixed with `[User location: …]\n\n` plus the question, and `location` for metadata.
2. **`Generate.generate()`** queues `(item_id, prompt["user"], location, attempt)` to `rag_request_q`.
3. **`rag_worker_process`** dequeues, sets `rag_agent.current_location = location`, runs **enrichment** (if enabled) to produce **`effective_query`**, then **`run_debug(effective_query, session_id=…)`**.
4. The worker returns **`(enriched_user, rag_answer)`** (or extends the existing response tuple so the parent receives **`effective_query`** alongside `rag_answer`).
5. On success, the main process builds the final prompt for the answer model as  
   **`effective_query` + `"\n\nadditional context: "` + `rag_answer`**  
   (not the original `prompt["user"]` unless enrichment was skipped or failed and `effective_query` equals `prompt["user"]`). On soft RAG failure, fall back as today (e.g. user-only prompt without RAG block).

**RAG and downstream generation both use the same enriched user text** (`effective_query`) for consistency.

---

## Implementation surface (likely files)

| Area | Role |
|------|------|
| **`rag_agent/` (new module + class)** | **Primary:** crop-dictionary enrichment — split prefix/body, LLM call, parse, recombine, fallback. Instantiate or call from the worker with `api_base`, `test_model`, dictionary data or path. |
| **`Inference/generate.py`** | Wire **`rag_worker_process`**: load dict path once per worker (or pass from `Generate`), call the enrichment class, pass **`effective_query`** to `run_debug`, extend **`rag_response_q`** so the main process receives **`effective_query`** for building `enhanced`. |

---

## Worker / multiprocessing notes

- Workers use **`multiprocessing` with `spawn`** (see `generate.py`). Each worker is a **separate process** → **each loads its own copy** of the JSON (acceptable; file is small compared to models).
- Pass **absolute path** to the dictionary into the worker (constructor args to `rag_worker_process` from `Generate.__init__` / CLI) so cwd differences do not break loading.
- Enrichment does **not** add another process per worker; it is **sequential** inside the existing worker loop.

---

## LLM / API alignment

- RAG already uses **`api_base`** (vLLM/Ollama-style OpenAI-compatible) and **`test_model`** per worker.
- Enrichment should use the **same** base URL and model name unless there is a reason to use a different model (document if so).
- Implement as a **single completion/chat** call with a **strict output**: e.g. only the final question string for the body (or JSON `{"query": "..."}`) to avoid parsing ambiguity.

---

## Edge cases to handle in the plan

- **`[User location: X]` prefix:** **Agreed:** preserve the prefix exactly; only the **question body** is passed to / rewritten by the crop LLM, then recombined with the original prefix.
- **Failures:** If enrichment LLM fails (timeout, parse error), **fallback** to original full `query` and log.
- **Retries:** RAG retries re-use the same queued `query`; re-enriching on retry is redundant but harmless, or cache by `query` in-process if desired.

---

## Success criteria (for the plan)

1. Dictionary JSON is loaded **once per RAG worker process** (not per request unless required).
2. **Enriched** string (`effective_query`) is what **`run_debug`** receives when enrichment is enabled and succeeds.
3. **Generation** step uses **`effective_query`** (returned from the worker) for concatenation with `rag_answer`, not the stale unenriched `prompt["user"]` alone.
4. Path to JSON is robust (absolute path; runtime copy under `Inference/` when using batch inference).
5. Prefix from `get_prompt` is preserved; no separate enrichment subprocess.
6. Tests/scripts explicitly out of scope can remain unchanged for v1.

---

## Related assets in this repo

- **Dictionary build:** `preload_pipeline/Dict-Value-Database/scripts/build_crop_dictionary.py`
- **Sample / test output:** `preload_pipeline/Dict-Value-Database/output/` (e.g. sanity JSON)
