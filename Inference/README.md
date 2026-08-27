# Batch inference (`generate.py`)

Place the runtime crop dictionary JSON as **`CropDatabase.json`** in this directory (same folder as `generate.py`) for query enrichment, or pass `--crop_dictionary_path` with another path. Use `--disable_query_enrichment` or `--crop_dictionary_path ""` to turn enrichment off.

## Runtime architecture

`generate.py` uses a staged inference pipeline:

```text
Input dataset
    ↓
Shared bounded RAG request queue
    ↓
One RAG worker per detected GPU/model endpoint
    ↓
RAG response queue
    ↓
Independent multiprocessing generation pool
    ↓
Incremental JSONL output
```

RAG workers use OpenAI-compatible endpoints beginning at port `11434` (`11435`, `11436`, and so on). Generation parallelism is controlled independently with `--num_processes`. RAG soft failures fall back to generation with the effective query, while hard failures are retried and skipped after the retry limit.

## Qdrant collections

Before a normal run, start Qdrant and set `QDRANT_URL` (default: `http://127.0.0.1:6333`). Inference uses two collection roles:

- `mirage_base` is the curated preload collection. It is read-only during inference and is required when `--use_base_collection true` is selected.
- `mirage_runtime_<ablation_id>_<YYYYMMDD>_<HHMMSS>` is selected or created for the run. Runtime web/PDF ingestion writes only to this collection.

The runtime collection is shared by all RAG workers and all queries in one run. `--runtime_mode resume` reuses the newest matching interrupted runtime collection; `--runtime_mode fresh` deletes matching runtime collections and starts a new one. A successful run optionally creates a snapshot with `--snapshot_runtime` and then deletes its runtime collection. Interrupted runs preserve it for resumption. `--runtime_collection_override` can select an existing runtime collection when resuming.

Use `--use_base_collection false` only for runtime-only development/testing. This skips base verification, base retrieval, and base-side deduplication while retaining the same runtime lifecycle and RAG code path.
