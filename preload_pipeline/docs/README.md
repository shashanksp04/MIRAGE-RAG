# Preload Pipeline Design Documentation

## Overview

The **Preload Pipeline** is a manifest-driven ingestion system that pre-populates the Chroma vector database used by the `rag_agent`.

Its primary purpose is to:

* Seed the vector database with authoritative reference sources
* Prevent cold-start retrieval failures
* Ensure first queries have meaningful semantic context
* Maintain versioned backups before every ingestion run
* Reuse as much of the existing `rag_agent` ingestion logic as possible

This pipeline is designed to operate independently of the runtime RAG agent, while producing a fully compatible persistent Chroma database directory.

---

# Architectural Principles

The pipeline is built around five core principles:

1. Safety-first ingestion (automatic versioned backups)
2. Reuse rag_agent chunking and deduplication logic
3. Manifest-driven configuration
4. Source-type modular adapters
5. Idempotent ingestion behavior

---

# Directory Structure

```
parent/
  rag_agent/
    tools/
      pdf_addition.py
      web_addition.py
    utils/
      Embedding.py
      ContentUtils.py

  preload_pipeline/
    bootstrap.py
    manifest.yaml
    PRELOAD_PIPELINE_DESIGN.md
    preload/
      ...
```

The `preload_pipeline` and `rag_agent` directories exist as siblings under the same parent directory.

---

# High-Level Pipeline Stages

## Stage 0 — Lock + Backup (Safety Layer)

This stage runs before any ingestion occurs.

Steps:

1. Acquire a file lock to prevent concurrent preload runs
2. Copy the entire Chroma persistence directory to:

   ```
   <persist_parent>/backups/<timestamp>_before_preload/
   ```
3. Optionally prune older backups

Why this matters:

* Guarantees rollback capability
* Protects against partial ingestion failures
* Allows experimentation without risk

This stage makes the preload pipeline behave like a transactional system.

---

## Stage 1 — Manifest Loading

The pipeline reads `manifest.yaml`.

The manifest defines:

* Source name
* Source type
* Paths or URLs
* Optional metadata fields
* Optional location fields (`location` or `location_field` for CSV)
* Unique ID field (for CSV sources)

Example:

```yaml
sources:
  - name: usda_plants_csv
    type: csv
    path: data/seeds/usda_plants.csv
    entity_type: plant
    source_org: USDA
    location: "Illinois"
    tags: [plants, usa]
```

The manifest ensures ingestion is configuration-driven rather than hardcoded.

---

## Stage 2 — rag_agent Integration Layer

The pipeline intentionally reuses major parts of `rag_agent`.

### Reused Components

From `rag_agent.utils.Embedding`:

* `SentenceTransformerEmbeddingFunction`

From `rag_agent.utils.ContentUtils`:

* `chunk_by_tokens(...)`
* `compute_content_hash(...)`
* `content_hash_exists(...)`
* tokenizer configuration
* chunk_config

From `rag_agent.tools.web_addition`:

* `WebAddition.add_web_content(...)`

From `rag_agent.tools.pdf_addition`:

* `PDFAddition.add_pdf_content(...)`

This guarantees:

* Identical chunk boundaries
* Identical deduplication behavior
* Identical document formatting
* Identical embedding function
* Identical metadata structure (for web/pdf)

The preload pipeline creates the same Chroma client + collection as the rag agent:

```python
SentenceTransformerEmbeddingFunction(...)
chromadb.PersistentClient(...)
get_or_create_collection(...)
```

This ensures the produced persistence directory is fully compatible with the RAG agent.

---

## Stage 3 — Source Adapters

Each source type has its own adapter.

### 1) Web Sources

Adapter: `WebPageListAdapter`

Behavior:

* Calls `WebAddition.add_web_content(url)`
* Passes `location` and `month_year` metadata when present in source config
* `location` is required for web sources (validated at manifest load); the underlying `add_web_content` also auto-derives location from `.edu` domain when not provided (used by rag-agent when ingesting URLs discovered at runtime)
* Extraction handled via trafilatura (inside rag_agent)
* Chunking handled via `ContentUtils.chunk_by_tokens`
* Deduplication handled via content hash logic
* Documents written via `collection.add(...)`

This is full reuse of rag_agent ingestion logic.

---

### 2) PDF Sources

Adapter: `PDFDirAdapter`

Behavior:

* Iterates over PDFs in directory
* Calls `PDFAddition.add_pdf_content(...)`
* Passes `location` and `month_year` metadata when present in source config
* Extraction via `pypdf`
* Chunking via rag_agent token chunker
* Deduplication via content hash
* Writes via `collection.add(...)`

Again, full reuse.

---

### 3) CSV Sources

Adapter: `CSVAdapter`

Since rag_agent does not have CSV ingestion logic, preload implements:

Row → structured narrative text → rag_agent chunking → rag_agent dedupe → collection.add()

Key details:

* Uses `ContentUtils.chunk_by_tokens`
* Uses `compute_content_hash`
* Uses `content_hash_exists`
* Stores metadata including:

  * source_name
  * record_id
  * entity_type
  * location (from source-level `location` or per-row `location_field`)
  * tags
  * content_hash

This ensures CSV data behaves identically to web/pdf chunks.

---

## Stage 4 — Run Report

After ingestion completes, a JSON report is written:

```
preload_run_report_<timestamp>.json
```

Includes:

* Sources processed
* Items added
* Items skipped (duplicates)
* Failures
* Backup path
* Timestamps

This makes ingestion auditable.

---

# How We Reuse rag_agent Code

The preload pipeline is not a parallel ingestion system.

Instead, it:

* Imports rag_agent as a module
* Uses the same embedding class
* Uses the same ContentUtils
* Uses the same chunking config
* Uses the same deduplication method
* Uses the same document formatting conventions

This ensures:

If a document was ingested via preload, it is indistinguishable from one ingested dynamically during runtime by the rag agent.

There is zero divergence in behavior.

---

# Metadata Behavior

For:

* Web sources → metadata comes from rag_agent tool
* PDF sources → metadata comes from rag_agent tool
* CSV sources → metadata is explicitly attached

All ingestion paths populate canonical metadata fields, including `location` and `hardiness_zone`.
`hardiness_zone` is derived from `location` via `rag_agent.utils.metadata.extract_hardiness_zone_for_location`.
Preferred `location` format is `"State, County"` or `"State"` (state full name or 2-letter abbreviation).

If desired, you can modify rag_agent tools to accept `extra_metadata` for richer provenance.

---

# Metadata Policy

Assumptions and enforcement used in this project:

* **Location policy**: `location` is required for preload web/pdf sources, and CSV must provide source-level `location` or `location_field`.
* **Location rag-agent policy**: when the RAG agent adds web content without an explicit `location`, it auto-derives the state from the URL’s `.edu` domain using `Datasets/land_grant_universities.csv`. For land-grant-university URLs, location is therefore enforced in both the preload pipeline (via manifest) and the rag-agent (via URL-derived state).
* **Hardiness policy**: `hardiness_zone` is always present as a metadata key and is derived from `location`.
* **Hardiness non-null expectation**: expected for resolvable locations; unresolved values may still be empty if a location does not map to the county/state lookup.
* **Month-year preload policy**: for preload web/pdf sources, `month_year` is expected to be provided in the manifest for every source.
* **Month-year rag policy**: for web-search-driven rag ingestion, `month_year` is derived from `page_age` (fallback to provider `month_year`) and validated as `YYYY-MM` before ingestion.
* **CSV month-year policy**: CSV ingestion is allowed to keep `month_year` empty by design.

Null vs non-null cases under the policy:

* **`hardiness_zone` non-null**: when `location` is valid and resolvable.
* **`hardiness_zone` empty**: when location is missing/unresolvable or lookup data cannot resolve it.
* **`month_year` non-null (preload web/pdf)**: when provided in manifest (assumed operational requirement).
* **`month_year` non-null (rag web-search flow)**: when derived or supplied and passes `YYYY-MM` validation.
* **`month_year` empty (CSV)**: accepted by current policy.

---

# Persistence Directory Strategy

The pipeline writes directly to the Chroma persistence directory used by rag_agent.

You have two options:

Option A:

* Preload writes to a temp directory
* Copy full directory into rag_agent

Option B (recommended):

* Preload writes directly to rag_agent’s persistence directory

In both cases:
Always copy the entire directory, never just SQLite files.

---

# Running the Pipeline

## Step 1 — Install Requirements

From `preload_pipeline/`:

```
pip install -r requirements.txt
```

---

## Step 2 — Prepare Manifest

Create:

```
preload_pipeline/manifest.yaml
```

Based on the example.

If you need to generate many `web_page_list` sources from a list of names, use:

```
python scripts/generate_web_sources.py \
  --base-url "https://extension.illinois.edu/plant-problems/" \
  --names-file "scripts/input.txt" \
  --location "Illinois" \
  --output "generated_sources.yaml"
```

Notes:

* Input file should be plain text with one name per line.
* `--base-url` and `--names-file` are required.
* Optional flags: `--output`, `--name-prefix`, `--entity-type`, `--source-org`, `--location`, and repeatable `--tag`.
* Preferred `--location` format is `"State, County"` or `"State"` (state can be full name or 2-letter abbreviation).
* Optional fields are only added to generated source records when provided.
* For CSV manifest sources, use either source-level `location` (single value for all rows) or `location_field` (column name containing per-row location).
* CSV rows with missing resolved location are rejected (counted as failed rows) to keep `hardiness_zone` metadata reliable.

---

## Step 3 — Run

From `preload_pipeline/`:

```
python bootstrap.py \
  --manifest manifest.yaml \
  --persist-dir ../rag_agent/chroma_database/chroma_db \
  --collection meta-mirage_collection \
  --rag-agent-dir ../rag_agent
```

Arguments explained:

* `--manifest` → Path to manifest.yaml
* `--persist-dir` → Chroma persistence directory
* `--collection` → Chroma collection name (must match rag_agent)
* `--rag-agent-dir` → Path to rag_agent directory
* `--embed-model` → Must match rag_agent embedding model
* `--device` → Must match rag_agent device setting
* `--dry-run` → Runs pipeline without writing to DB

---

# Recommended Operational Workflow

1. Stop rag_agent if running
2. Run preload pipeline
3. Review run report
4. Restart rag_agent

---

# Safety Guarantees

Every run:

* Creates versioned backup
* Prevents concurrent runs
* Logs ingestion results
* Maintains deduplication
* Preserves chunk consistency

---

# Future Improvements

Possible enhancements:

* Add extra_metadata support to rag_agent tools
* Add incremental update mode
* Add source-level refresh policies
* Add validation queries after ingestion
* Add checksum validation of persistence directory

---

# Summary

The preload pipeline is:

A versioned, manifest-driven, safety-first ingestion system that fully reuses rag_agent chunking, deduplication, and embedding logic to ensure database consistency and high-quality retrieval from the very first query.

It avoids architectural drift and ensures the vector database built offline behaves identically to runtime ingestion.

---

## Example scripts:

python scripts/generate_web_sources.py --base-url "https://extension.illinois.edu/plant-problems/"   --names-file "./names/uiuc.txt"    --location "Illinois" --entity-type "disease"  --source-org "Illinois Extension" --output "./uiuc_generated_sources.yaml"

python bootstrap.py \
  --manifest uiuc_generated_sources.yaml \
  --persist-dir ./chroma_database_src/chroma_db \
  --collection meta-mirage_collection \
  --rag-agent-dir ../rag_agent