# MIRAGE-RAG

This README is a concise entry point based on `Guide.md`.

MIRAGE-RAG is an end-to-end retrieval-augmented pipeline with:

- Offline ingestion into a persistent Chroma vector database
- Optional crop-dictionary query enrichment
- Multi-process batch inference with GPU-aware RAG workers
- OpenAI-compatible generation backends

For full implementation detail, read `Guide.md`.

## Repository Map

- `rag_agent/`: `MainAgent`, retrieval/confidence/web tools, embeddings, chunking, ablation configs/templates.
- `preload_pipeline/`: Manifest-driven preload pipeline for vector DB creation/refresh.
- `Inference/`: Batch runtime pipeline (`generate.py`) and run wrapper scripts.
- `chat_models/`: Generation/chat client utilities.
- `Datasets/`: Metadata/location reference data used by retrieval and web filtering.

## Core Artifacts

Two artifacts are intentionally separate:

- **Chroma DB**: chunk embeddings + metadata for retrieval.
- **Crop dictionary JSON**: optional query enrichment only (does not write to Chroma).

## Critical Consistency Rules

- Keep Chroma **persist path** and **collection name** aligned between preload and runtime.
- Default collection is `meta-mirage_collection`.
- Keep embedding model/device aligned between preload and runtime.
- Copy the **entire** Chroma persistence directory when moving environments.

## Quick Start

### 1) Build or Refresh Vector DB (Preload)

Run from `preload_pipeline/`:

```bash
pip install -r requirements.txt
python bootstrap.py \
  --manifest manifest.yaml \
  --persist-dir ../rag_agent/chroma_database/chroma_db \
  --collection meta-mirage_collection \
  --rag-agent-dir ../rag_agent
```

Key flags:

- `--manifest`: preload YAML source list
- `--persist-dir`: Chroma persistence directory
- `--collection`: collection name used by runtime
- `--rag-agent-dir`: path to `rag_agent` package
- `--embed-model`, `--device`: must match runtime settings
- `--dry-run`: execute pipeline logic without DB writes

### 2) Optional: Build Crop Dictionary (Query Enrichment)

Run from `preload_pipeline/Dict-Value-Database/`:

```bash
python scripts/generate_web_sources.py \
  --base-url "https://extension.illinois.edu/plant-problems/" \
  --names-file "../Ingestion/URLs/names/uiuc.txt" \
  --state "Illinois" \
  --category "disease" \
  --output "YAMLfilesForDict/uiuc.yaml"

python scripts/build_crop_dictionary.py \
  --config YAMLfilesForDict/uiuc.yaml \
  --csv ../../Datasets/county_crops_frequency_multi_year_cleaned.csv \
  --output output/crop_dictionary_output.json
```

Then place/copy output JSON to `Inference/CropDatabase.json` (or pass `--crop_dictionary_path`).

### 3) Run Batch Inference

Run from `Inference/`:

```bash
bash bash_generate.sh
```

`bash_generate.sh` is the standard entrypoint for batch runs. It sets run configuration (including `ABLATION_ID`) and then invokes `generate.py` with the expected arguments.

Runtime behavior (high level):

1. Loads dataset input and skips already-completed items.
2. Starts RAG workers (rank0 first), one endpoint per GPU.
3. Builds user prompt (including optional location prefix).
4. Applies optional query enrichment to produce `effective_query`.
5. Runs RAG; on success adds retrieval context to generation prompt.
6. Writes JSONL output with retry/failure handling.

## Runtime Notes

- The runtime expects an OpenAI-compatible API endpoint per worker/GPU.
- Default endpoint pattern starts at port `11434` (`11434 + gpu_index`).
- Rank0 may reset/recreate Chroma collection before other workers start.
- Retrieval has self-healing rebinding logic for stale collection handles.

## Metadata and Retrieval Model

- Metadata fields like `location`, `hardiness_zone`, `month_year`, and `title` affect retrieval strategy quality.
- Progressive retrieval evaluates multiple metadata filter strategies plus semantic-only retrieval.
- Confidence evaluation uses retrieval outputs and selected strategy scope.
- Consistent ingest-time metadata improves runtime precision and reduces unnecessary web search.

## Ablation Control

Ablations are controlled by three linked pieces:

- Run selector: `Inference/bash_generate.sh` (`ABLATION_ID`)
- Settings map: `rag_agent/ablation_configs.json`
- Instruction templates: `rag_agent/model_instructions.md`

Toggles can gate:

- Progressive filtering
- Confidence evaluation
- Web search
- Domain filtering
- Ingestion loop

## Working Directory Expectations

- Run preload commands from `preload_pipeline/`.
- Run dictionary-build commands from `preload_pipeline/Dict-Value-Database/`.
- Run batch jobs from `Inference/` (or ensure equivalent path assumptions).

## Operational Checklist

Before preload:

- Validate manifest `sources` and required location fields.
- Stop writers for the target Chroma persist dir.
- Ensure disk for backup + new ingestion output.

Before inference:

- Confirm LLM server endpoints are reachable.
- Confirm Chroma path, collection, embed model, and device alignment.
- Confirm ablation ID/template availability.
- Confirm crop dictionary placement if enrichment is enabled.

## Documentation Index

- `Guide.md` (primary): full architecture, command reference, operations, troubleshooting
- `Documentation.md`: runtime queue design, rank0 reset, self-healing behavior
- `preload_pipeline/docs/README.md`: preload manifest/schema and ingestion policy
- `Inference/README.md`: inference usage and enrichment flags
- `preload_pipeline/Dict-Value-Database/QUERY_ENRICHMENT_CONTEXT.md`: enrichment logic and safeguards
