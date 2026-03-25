# Context brief: crop-dictionary query enrichment (for implementation planning)

Use this as background for designing and implementing the feature.

---

## Goal

Before the existing RAG pipeline runs (`retrieve` → confidence → optional web → …), run a **preprocessing step**:

1. Take the **user query** (the same string currently sent to the RAG worker).
2. Load a **crop dictionary** from a **JSON file** (structure aligned with the Dict-Value-Database / crop metadata used elsewhere in the project).
3. Pass **query + dictionary** to an **LLM** with instructions roughly:
   - If the query references **category-level information** for a crop (e.g. a field that exists in the dictionary for that crop) **but the crop name does not appear** in the query, **inject the crop name** into the query.
   - If there is **no meaningful overlap** between query and dictionary-driven signals, **return the query unchanged**.
4. Treat the **LLM output** as the **only** user query string for the rest of the pipeline (retrieval, tools, final formatting).

Exact prompting and matching rules should be specified during implementation; the product intent is **disambiguation / completion** when the user implies a crop without naming it.

---

## Architectural decision (agreed)

- **Do not** rely on `rag_agent/main.py`’s `if __name__ == "__main__"` block for batch runs. Batch inference uses **`Inference/generate.py`**, which imports `MainAgent` inside **`rag_worker_process`** and calls `rag_runner.run_debug(query, session_id=…)`.
- **Enrichment runs inside each RAG worker process**, immediately **before** `run_debug`, so:
  - Each worker loads the dictionary JSON **once per worker** (e.g. at worker startup or lazy first-use cache inside that process).
  - The **same OpenAI-compatible endpoint / model** used for that worker’s RAG stack can be reused for the enrichment LLM call (same `api_base`, `test_model` pattern as today).
- **Testing entry points** (`Inference/test.py`, `rag_agent/test_standalone.py`) are **out of scope** for now; no requirement to change them initially.
- **Dictionary file placement (runtime):** For runs, the JSON can live under **`Inference/`** (e.g. next to `generate.py`) so paths can be resolved from `Path(__file__).resolve().parent` in `generate.py` and passed into workers (or reconstructed consistently in the worker). The **authoritative** dictionary build still lives under this pipeline; copy or symlink the built JSON into `Inference/` when running batch inference.

---

## Current data flow (relevant parts)

1. **`Generate.get_prompt(item)`** builds `prompt["user"]`, optionally prefixed with `[User location: …]\n\n` plus the question, and `location` for metadata.
2. **`Generate.generate()`** queues `(item_id, prompt["user"], location, attempt)` to `rag_request_q`.
3. **`rag_worker_process`** dequeues, sets `rag_agent.current_location = location`, then **`run_debug(query, session_id=…)`** with `query` from the queue.
4. On success, the main process builds the final prompt for the answer model as  
   `prompt["user"] + "\n\nadditional context: " + rag_answer`  
   (or fallback to `prompt["user"]` on soft fail).

**Important:** If enrichment stays **only in the worker**, the **main process still holds the original** `prompt["user"]` for the generation step unless you also update `prompt["user"]` in the parent **or** return/store the enriched query from the worker and merge downstream.

**Required design point for implementers:** The user previously considered enriching in the parent so `prompt["user"]` was updated everywhere. With **worker-only** enrichment, you must **explicitly decide**:

- Either **also** update what gets concatenated for generation (e.g. worker returns enriched query in the response tuple, or main process replaces `prompt["user"]` after receiving enriched text once), **or**
- Accept that **RAG** sees enriched text but **generation** still uses the **unenriched** `prompt["user"]` unless fixed.

**RAG and downstream generation should both use the same enriched user text** for consistency unless there is a documented reason not to.

---

## Implementation surface (likely files)

| Area | Role |
|------|------|
| **`Inference/generate.py`** | Primary file: extend `rag_worker_process` to load dict (or receive path), call `enrich_query(...)`, pass result to `run_debug`. Possibly extend queue/response protocol if enriched query must be sent back to the main process for `enhanced = …` construction. |
| **`rag_agent/main.py` (optional)** | Could host a small `enrich_query` helper or LLM wrapper **if** you want reuse; not strictly required if logic lives in `Inference/` only. |
| **New module under `Inference/` (optional)** | e.g. `query_enrichment.py` with `load_dictionary(path)`, `enrich_query(query, dict_data, api_base, model, …)` for clarity and testing. |

---

## Worker / multiprocessing notes

- Workers use **`multiprocessing` with `spawn`** (see `generate.py`). Each worker is a **separate process** → **each loads its own copy** of the JSON (acceptable; file is small compared to models).
- Pass **absolute path** to the dictionary into the worker (constructor args to `rag_worker_process` from `Generate.__init__` / CLI) so cwd differences do not break loading.

---

## LLM / API alignment

- RAG already uses **`api_base`** (vLLM/Ollama-style OpenAI-compatible) and **`test_model`** per worker.
- Enrichment should use the **same** base URL and model name unless there is a reason to use a different model (document if so).
- Implement as a **single completion/chat** call with a **strict output**: e.g. only the final query string (or JSON `{"query": "..."}`) to avoid parsing ambiguity.

---

## Edge cases to handle in the plan

- **`[User location: X]` prefix:** Decide whether the LLM sees the **full** `prompt["user"]` or only the **question** slice, and whether enrichment should **preserve** the prefix exactly if present.
- **Failures:** If enrichment LLM fails (timeout, parse error), **fallback** to original `query` and log.
- **Retries:** RAG retries re-use the same queued `query`; if enrichment is deterministic from that string, re-enriching on retry is redundant but harmless, or cache by `query` in-process if desired.

---

## Success criteria (for the plan)

1. Dictionary JSON is loaded **once per RAG worker process** (not per request unless required).
2. **Enriched** string is what **`run_debug`** receives.
3. **Generation** step uses an **enriched** user message where intended (resolve the parent vs worker split explicitly).
4. Path to JSON is robust (absolute path; runtime copy under `Inference/` when using batch inference).
5. Tests/scripts explicitly out of scope can remain unchanged for v1.

---

## Related assets in this repo

- **Dictionary build:** `preload_pipeline/Dict-Value-Database/scripts/build_crop_dictionary.py`
- **Sample / test output:** `preload_pipeline/Dict-Value-Database/output/` (e.g. sanity JSON)
