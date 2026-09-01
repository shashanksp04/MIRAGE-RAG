# MetaMIRAGE Preload Pipeline — Finalized Architecture

## 1. Purpose

This document defines the finalized architecture for the MetaMIRAGE offline database-building pipeline.

The pipeline is designed for:

- Very large source corpora (hundreds of GB).
- Repeated state-by-state processing across all 50 U.S. states.
- Avoiding duplicate extraction/parsing work.
- Strong restart/resume behavior.
- Global deduplication across runs.
- Qualification of documents before expensive RAG indexing.
- A hard metadata contract compatible with MetaMIRAGE priority retrieval.
- Batched embedding and Qdrant ingestion.
- Automated cumulative Qdrant snapshots and run manifests.
- Reuse of the same qualification outputs for the crop dictionary builder.
- Maintenance of one cumulative `crop_occurrences.json` that is enriched state-by-state rather than creating separate crop-dictionary files.

The Jupyter notebook is the **orchestrator** of this offline build pipeline.

---

# 2. High-Level Architecture

```text
Source Discovery
      ↓
Extraction + Normalization
      ↓
Canonical Store + SQLite Ledger
      ↓
Global Deduplication
      ↓
Qualification
      ├── Crop Dictionary Builder
      │         ↓
      │   update current state in
      │   cumulative crop_occurrences.json
      │
      └── accepted?
             ↓
        RAG Chunking
             ↓
Metadata Enrichment
             ↓
Hard Metadata Contract Validation
             ↓
Batch Embedding
             ↓
Batch Qdrant Upsert
             ↓
Retry Failed Units
             ↓
Terminal-State Validation
             ↓
Run Validation
             ↓
Cumulative Snapshot
             ↓
Manifest + Crop Dictionary
             ↓
runs/<build>/<sequence_state>/
```

The most important invariant is:

> **A raw source is extracted and normalized only once.**

All downstream work uses the persisted canonical representation.

---

# 3. Jupyter Notebook Role

The notebook is the pipeline orchestrator.

It is responsible for:

- Run configuration.
- Source discovery.
- Extraction and normalization.
- Canonical document persistence.
- Global duplicate detection.
- Qualification chunking.
- Qualification LLM calls.
- Retry orchestration.
- Acceptance/rejection decisions.
- Crop dictionary aggregation.
- RAG chunk construction.
- Metadata enrichment.
- Metadata validation.
- Batched embedding.
- Qdrant batch upserts.
- Run validation.
- Snapshot creation.
- Manifest generation.
- Resume behavior.

Qdrant runs as a separate process/server.

```text
Delta Compute Node

Jupyter Kernel
     │
     │ HTTP
     ▼
127.0.0.1:6333
     │
Qdrant Server
     │
     ▼
Persistent Qdrant Storage
```

---

# 4. Persistence Model

The system uses four different persistence layers, each with a separate purpose.

## 4.1 SQLite — Processing Ledger

SQLite answers:

- What has already happened?
- What still needs processing?
- What failed?
- What must be retried?
- What is terminal?
- What run/state owns the work?

Example location:

```text
/u/ssingh38/Database/pipeline_state.db
```

SQLite stores metadata and status only, not large document text.

---

## 4.2 Canonical Document Store

The canonical store answers:

> What exact normalized content did we extract from this source?

Example layout:

```text
/u/ssingh38/Database/
│
├── pipeline_state.db
│
├── canonical/
│   ├── ab/
│   │   └── abcdef.../
│   │       ├── content.txt.zst
│   │       └── metadata.json
│   └── ...
│
├── runs/
│   └── ...
│
└── qdrant/
```

The directory is sharded by the first two characters of `document_id` to avoid millions of files in a single directory.

Canonical content is immutable for a given extraction version.

---

## 4.3 Qdrant — Retrieval Index

Qdrant contains the derived retrieval representation:

- RAG chunks.
- Embeddings.
- Runtime-compatible metadata payloads.
- Deterministic point IDs.

Qdrant is not the sole copy of accepted content.

The canonical store remains the durable source for re-chunking/re-embedding later.

---

## 4.4 Qdrant Snapshots — Portable Build Checkpoints

Each completed state produces a cumulative snapshot.

Snapshots are the versioned build artifacts.

They allow:

- Restarting from a prior state.
- Reusing the prepared DB on another Delta allocation.
- Preserving exact cumulative DB state after each state.
- Avoiding re-embedding/re-indexing earlier states.

---

# 5. Build and Run Naming

Use one cumulative working Qdrant collection:

```text
mirage_base_build
```

Each 50-state processing pass belongs to a build:

```text
BUILD_ID = build_2026_08
```

Each state gets a sequential run ID:

```text
001_IL
002_IN
003_IA
...
```

The live Qdrant collection accumulates state by state:

```text
Run 001 → Illinois
snapshot = Illinois

Run 002 → Indiana
snapshot = Illinois + Indiana

Run 003 → Iowa
snapshot = Illinois + Indiana + Iowa
```

There is no need to keep 50 simultaneous live Qdrant collections.

The snapshots are the versions.

---

# 6. Automated Run Folder Layout

Each completed state automatically creates a cumulative Qdrant snapshot and a manifest:

```text
Database/
├── crop_occurrences.json
│
└── runs/
    └── build_2026_08/
        ├── 001_IL/
        │   ├── mirage_base_001_IL.snapshot
        │   └── manifest.json
        │
        ├── 002_IN/
        │   ├── mirage_base_002_IN.snapshot
        │   └── manifest.json
        │
        └── ...
```

`crop_occurrences.json` is intentionally **not duplicated into every run folder**. It is a single cumulative artifact at the build root and is enriched in place as each state completes.

Each snapshot is cumulative through that state.

Each manifest separately reports:

- Added this run.
- Cumulative totals.
- Retry/failure counts.
- Metadata-quality statistics.
- Snapshot checksum.
- Which state section of `crop_occurrences.json` was updated and how many crop entries it contains.

---

# 7. Canonical Document Identity

Document identity is based on normalized extracted content.

```text
canonical normalized content
        ↓
SHA-256
        ↓
content_hash
        ↓
document_id
```

For v1:

```text
document_id = content_hash
```

This means:

- Renamed PDF copies are the same document.
- Moved PDF copies are the same document.
- Duplicate URLs with the same extracted content are the same document.
- Duplicate CSV records can also be detected globally.

A separate raw-byte hash may also be retained when available for diagnostics.

---

# 8. Global Deduplication

Deduplication scope is global across all 50 states.

If a document is seen during a later state and its `content_hash` already exists:

```text
duplicate detected
      ↓
duplicate_skipped
      ↓
no qualification
no RAG chunking
no embedding
no Qdrant insertion
```

The duplicate encounter is still recorded in the run ledger and manifest.

This provides statistics such as:

```text
duplicates_skipped_this_run = 428
```

---

# 9. Source Semantics

The pipeline supports:

- PDFs.
- URLs/web pages.
- CSVs.

Canonical-document granularity:

```text
PDF       → one canonical document
Web page  → one canonical document
CSV       → one canonical document per row/record
```

For CSV rows, create a deterministic textual representation before hashing, qualification, or RAG chunking.

Example:

```text
crop: soybean
disease: soybean rust
state: Illinois
description: ...
```

This representation must be deterministic so reruns produce the same document hash.

---

# 10. Canonical Metadata Schema

Each canonical document has a small `metadata.json`.

Example:

```json
{
  "schema_version": "1.0",

  "document_id": "sha256...",
  "content_hash": "sha256...",
  "raw_hash": "sha256...",

  "source_type": "pdf",
  "source_uri": "/path/to/file.pdf",
  "source_name": "file.pdf",

  "source_record": {
    "csv_row_index": null,
    "csv_record_id": null
  },

  "run_discovered": "001_IL",
  "state_discovered": "Illinois",

  "title": "Optional extracted title",
  "language": "en",

  "canonical_text_path": "canonical/ab/abcdef/content.txt.zst",
  "canonical_text_chars": 123456,
  "canonical_text_bytes": 98765,

  "extraction": {
    "extractor_version": "v1",
    "page_count": 42,
    "extracted_at": "..."
  },

  "source_metadata": {}
}
```

Qualification output is intentionally not stored in this immutable canonical metadata artifact.

---

# 11. Qualification

Qualification and RAG chunking are separate systems.

Qualification uses its own larger chunks and early stopping behavior.

Current qualification settings:

```text
Chunk size: 7000 characters
Overlap:    700 characters
Maximum:    20 chunks per document
```

Qualification tags:

```text
crops
pest
disease
management
multi
msc
```

Acceptance rule:

```python
accepted = (
    qualification_status == "succeeded"
    and tag != "msc"
)
```

Therefore:

```text
crops       → ACCEPT
pest        → ACCEPT
disease     → ACCEPT
management  → ACCEPT
multi       → ACCEPT
msc         → REJECT
```

Rejected documents stop before RAG processing.

---

# 12. Crop Dictionary Builder and Cumulative `crop_occurrences.json`

The crop dictionary remains part of the notebook, but there is **one cumulative JSON file for all states**:

```text
crop_occurrences.json
```

The initial skeleton is generated from the county-level crop-frequency source by aggregating county crop frequencies into state-level occurrence totals.

Conceptual structure:

```json
{
  "illinois": {
    "corn": {
      "occurrence": 1234,
      "disease": {},
      "pests": {},
      "management": {}
    },
    "soybeans": {
      "occurrence": 987,
      "disease": {},
      "pests": {},
      "management": {}
    }
  },
  "indiana": {
    "...": {}
  }
}
```

The same qualification outputs feed both:

```text
Qualification Result
        │
        ├── accepted/rejected → database path
        │
        └── crop_entities → crop dictionary builder
                                  ↓
                          enrich current state only
                                  ↓
                         crop_occurrences.json
```

No additional qualification/LLM processing is performed solely for the crop dictionary.

For a state run, the notebook:

1. Loads only the current state's section as the crop baseline.
2. Preserves the occurrence values and any existing enrichment already in that state.
3. Adds newly discovered crops when needed with `occurrence = 0` and `added = true`.
4. Aggregates disease, pest, and management entities from successful qualification results.
5. Atomically replaces only the current state's section in the same cumulative JSON file.
6. Leaves every other state's section untouched.

The JSON update must be atomic so an interrupted write cannot corrupt the cumulative artifact.

The initialization script may safely be rerun: occurrence counts can be refreshed from the source CSV while existing enrichment fields are preserved.

---

# 13. Retry Strategy

Retries are scoped to the current run/state.

Default:

```python
MAX_STAGE_RETRIES = 2
```

Meaning:

```text
initial attempt
+ retry 1
+ retry 2
= maximum 3 total attempts
```

The initial pass processes all units.

Example:

```text
Process Illinois
      ↓
some succeed
some reject normally
some fail
      ↓
finish initial pass
      ↓
collect failed units for Illinois only
      ↓
retry pass #1
      ↓
retry pass #2
      ↓
remaining failures
      ↓
permanently_failed
```

A state is not interrupted every time one unit fails.

Failures are retried after the main pass.

---

# 14. Terminal States

`permanently_failed` is a terminal state.

This allows a state to complete even with explicitly documented unrecoverable failures.

Example:

```text
Illinois
────────
completed              48,921
rejected               17,204
permanently_failed         13

RUN STATUS = COMPLETE
```

Permanent failures are recorded in the manifest and attempt history.

---

# 15. Restartable Unit Granularity

Failures are tracked at the smallest restartable unit.

Examples:

- Extraction stage → document.
- Qualification stage → qualification chunk.
- RAG indexing stage → RAG chunk.

This prevents one broken chunk from forcing reprocessing of an otherwise successful document.

---

# 16. SQLite Processing Ledger

Primary tables:

```text
runs
documents
run_documents
qualification_chunks
rag_chunks
attempts
snapshots
```

---

# 17. `runs` Table

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL,

    sequence_number INTEGER NOT NULL,

    state_name TEXT NOT NULL,
    state_code TEXT NOT NULL,

    status TEXT NOT NULL,

    started_at TEXT,
    completed_at TEXT,

    max_stage_retries INTEGER NOT NULL DEFAULT 2,

    snapshot_path TEXT,
    manifest_path TEXT,

    error TEXT
);
```

Allowed statuses:

```text
pending
running
retrying
validating
snapshotting
complete
failed
```

---

# 18. `documents` Table

Global document registry across all states.

```sql
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,

    content_hash TEXT NOT NULL UNIQUE,
    raw_hash TEXT,

    source_type TEXT NOT NULL,

    canonical_text_path TEXT NOT NULL,
    canonical_metadata_path TEXT NOT NULL,

    canonical_text_chars INTEGER,
    canonical_text_bytes INTEGER,

    language TEXT,

    extractor_version TEXT NOT NULL,

    extraction_status TEXT NOT NULL,

    first_seen_run_id TEXT NOT NULL,
    first_seen_state TEXT NOT NULL,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(first_seen_run_id) REFERENCES runs(run_id)
);
```

---

# 19. `run_documents` Table

Tracks what happened when a specific run encountered a document.

```sql
CREATE TABLE run_documents (
    run_id TEXT NOT NULL,
    document_id TEXT NOT NULL,

    source_type TEXT NOT NULL,
    source_uri TEXT,

    discovery_order INTEGER,

    duplicate INTEGER NOT NULL DEFAULT 0,

    document_status TEXT NOT NULL,

    qualification_status TEXT,
    accepted INTEGER,

    qualification_tag TEXT,
    qualification_subtags_json TEXT,
    qualification_entities_json TEXT,
    qualification_reason TEXT,

    classifier_version TEXT,

    rag_status TEXT,
    qdrant_status TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    PRIMARY KEY(run_id, document_id),

    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(document_id) REFERENCES documents(document_id)
);
```

Document statuses:

```text
discovered
extracting
extracted
qualifying
accepted
rejected
rag_preparing
indexing
indexed
duplicate_skipped
permanently_failed
```

Terminal document states:

```text
indexed
rejected
duplicate_skipped
permanently_failed
```

---

# 20. `qualification_chunks` Table

```sql
CREATE TABLE qualification_chunks (
    qualification_chunk_id TEXT PRIMARY KEY,

    run_id TEXT NOT NULL,
    document_id TEXT NOT NULL,

    chunk_index INTEGER NOT NULL,

    chunk_hash TEXT NOT NULL,

    status TEXT NOT NULL,

    classification_result_json TEXT,

    attempt_count INTEGER NOT NULL DEFAULT 0,

    last_error TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(document_id) REFERENCES documents(document_id),

    UNIQUE(run_id, document_id, chunk_index)
);
```

Statuses:

```text
pending
processing
succeeded
failed
skipped_early_stop
permanently_failed
```

Terminal:

```text
succeeded
skipped_early_stop
permanently_failed
```

`failed` is retryable and not terminal.

---

# 21. Qualification Retry Lifecycle

```text
INITIAL PASS

chunk 1 ✓
chunk 2 ✓
chunk 3 ✗
chunk 4 ✓
chunk 5 ✗
...
      ↓
finish all initial work
      ↓
SELECT current-run chunks
WHERE status='failed'
      ↓
retry pass #1
      ↓
remaining failures
      ↓
retry pass #2
      ↓
remaining failures
      ↓
permanently_failed
```

Retry queues are always scoped by `run_id`.

---

# 22. RAG Chunking Contract

Tokenizer:

```text
BAAI/bge-base-en-v1.5
```

Settings:

```text
Maximum chunk size: 480 tokens
Overlap:             80 tokens
Hard safety cap:    512 tokens
```

Splitting algorithm:

```python
tokens = tokenizer.encode(
    text,
    add_special_tokens=False
)

start = 0

while start < len(tokens):
    end = min(start + 480, len(tokens))

    chunk_tokens = tokens[start:end]
    chunk_text = tokenizer.decode(chunk_tokens)

    save(chunk_text)

    if end == len(tokens):
        break

    start = end - 80
```

---

# 23. Web RAG Chunking

For web content:

```text
page <= 512 tokens
    → keep whole page as one chunk

page > 512 tokens
    → split into 480-token chunks
    → 80-token overlap
```

---

# 24. PDF RAG Chunking

Each PDF page is processed independently.

For every page:

```text
480 tokens per chunk
80 overlapping tokens
```

Even a PDF page below the 512-token web threshold follows the PDF page-based chunking path.

---

# 25. CSV RAG Chunking

Each CSV row is already one canonical logical document.

Use web-like behavior:

```text
row narrative <= 512 tokens
    → one RAG chunk

row narrative > 512 tokens
    → 480-token chunks
    → 80-token overlap
```

---

# 26. RAG Chunk Table

```sql
CREATE TABLE rag_chunks (
    rag_chunk_id TEXT PRIMARY KEY,

    run_id TEXT NOT NULL,
    document_id TEXT NOT NULL,

    chunk_index INTEGER NOT NULL,

    chunk_hash TEXT NOT NULL,

    text_start_offset INTEGER,
    text_end_offset INTEGER,

    chunker_version TEXT NOT NULL,

    status TEXT NOT NULL,

    embedding_status TEXT NOT NULL,
    qdrant_status TEXT NOT NULL,

    embedding_attempt_count INTEGER NOT NULL DEFAULT 0,
    qdrant_attempt_count INTEGER NOT NULL DEFAULT 0,

    last_error TEXT,

    qdrant_point_id TEXT NOT NULL,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY(run_id) REFERENCES runs(run_id),
    FOREIGN KEY(document_id) REFERENCES documents(document_id),

    UNIQUE(document_id, chunk_index, chunker_version)
);
```

Lifecycle:

```text
pending
   ↓
metadata_enriching
   ↓
metadata_validated
   ↓
embedding
   ↓
embedded
   ↓
qdrant_pending
   ↓
indexed
```

Failure is represented by:

```text
failed
```

and after retries:

```text
permanently_failed
```

The failure stage is recorded separately.

---

# 27. Deterministic RAG IDs

RAG chunk identity is based on:

```text
document_id
+
chunk_index
+
chunker_version
```

Example:

```python
qdrant_point_id = UUID5(
    namespace,
    f"{document_id}:{chunk_index}:{chunker_version}"
)
```

This ensures idempotency.

Same document + same chunking configuration = same Qdrant point.

---

# 28. Metadata Enrichment + Contract Validation

This is a hard ingestion gate.

No RAG chunk is embedded or written to Qdrant until its payload satisfies the metadata contract.

Pipeline:

```text
RAG Chunk
    ↓
Metadata Enrichment
    ↓
Metadata Contract Validation
    ↓
valid?
 ┌──┴───┐
 │      │
No     Yes
 │      │
retry   ↓
      Embedding
         ↓
      Qdrant
```

The Qdrant writer must never construct metadata itself.

It accepts only already validated `PreparedRAGChunk` objects.

---

# 29. Qdrant Metadata Contract

Every Qdrant point payload must contain:

```python
{
    "text": ...,
    "chunk_id": ...,

    "source_type": ...,
    "source_id": ...,
    "title": ...,
    "url": ...,
    "page": ...,
    "chunk_index": ...,
    "location": ...,
    "month_year": ...,
    "content_hash": ...,
    "language": ...,
    "hardiness_zone": ...
}
```

Required fields always exist.

Expected rules:

| Field | Must exist | May be empty? | Rule |
|---|---:|---:|---|
| `text` | Yes | No | RAG chunk content |
| `chunk_id` | Yes | No | Deterministic chunk ID |
| `source_type` | Yes | No | `pdf`, `web`, `csv`, etc. |
| `source_id` | Yes | No | Stable source/document ID |
| `title` | Yes | No | Real title or deterministic fallback |
| `url` | Yes | Yes | Empty if not applicable |
| `page` | Yes | Yes-ish | Sentinel such as `-1` where not applicable |
| `chunk_index` | Yes | No | Deterministic 0-based index |
| `location` | Yes | No | Current state or state+county |
| `month_year` | Yes | Yes | `YYYY-MM` when known, else `""` |
| `content_hash` | Yes | No | Hash of normalized chunk text |
| `language` | Yes | No | Usually `en` |
| `hardiness_zone` | Yes | Normally no | AK/HI documented exception |

---

# 30. Location + Hardiness-Zone Derivation

The preload notebook independently reconstructs the inference hardiness-zone logic using:

```text
county_state_hardiness_zone.csv
```

The system builds:

```text
(county, state) → hardiness_zone
state → modal hardiness_zone
```

Supported location forms:

```text
"State, County"
"State"
"ST"
"County, State"
"County, ST"
```

Resolution:

1. If state + county are available, try exact county zone.
2. If county is missing/not found, use state modal zone.
3. If state cannot be recognized, return `""`.

Typical state-level run:

```text
location = "Illinois"
        ↓
state modal lookup
        ↓
hardiness_zone
```

If county metadata is available:

```text
location = "Illinois, Champaign"
        ↓
exact county lookup
        ↓
hardiness_zone
```

---

# 31. Alaska + Hawaii Exception

The supplied hardiness mapping does not provide usable hardiness zones for Alaska or Hawaii.

Therefore:

```text
AK / HI
hardiness_zone = ""
```

is a documented exception to the otherwise strict hardiness-zone contract.

This exception must be counted separately in the run manifest.

---

# 32. `month_year` Contract

`month_year` must exist on every chunk.

Valid values:

```text
Known date:
"2024-07"

Unknown:
""
```

For URLs and PDFs:

- Attempt to determine source/publication month and year.
- Normalize to `YYYY-MM`.
- Do not fabricate an ingestion date.
- Leave as `""` if not determinable.

Retrieval can then skip month-based filters when the field is empty.

---

# 33. Title Fallbacks

`title` must not be empty.

Fallback strategy:

## PDF

```text
document metadata title
    ↓ fallback
filename without extension
```

## URL

```text
HTML <title>
    ↓ fallback
page heading
    ↓ fallback
normalized URL slug
```

## CSV

```text
configured title field(s)
    ↓ fallback
deterministic synthetic record title
```

Example:

```text
Illinois soybean disease record 000123
```

---

# 34. Metadata Validation Result

The validator should return structured information.

Example success:

```python
{
    "valid": True,
    "missing_fields": [],
    "invalid_fields": [],
    "month_year_available": False
}
```

Example failure:

```python
{
    "valid": False,
    "missing_fields": ["hardiness_zone"],
    "invalid_fields": [],
    "month_year_available": True
}
```

Validation failures are retryable within the current run.

After retries are exhausted, the RAG chunk becomes `permanently_failed`.

---

# 35. Metadata Quality Metrics

Every run manifest tracks metadata quality.

Example:

```json
{
  "metadata_quality": {
    "rag_chunks_total": 382911,

    "contract_valid": 382902,
    "contract_permanently_failed": 9,

    "location_present": 382911,
    "hardiness_zone_present": 382902,
    "hardiness_zone_ak_hi_exception": 0,

    "title_present": 382911,

    "month_year_field_present": 382911,
    "month_year_known": 310224,
    "month_year_unknown": 72687,

    "priority_full_metadata": 310215,

    "missing_or_invalid": {
      "source_type": 0,
      "source_id": 0,
      "title": 0,
      "url": 0,
      "page": 0,
      "chunk_index": 0,
      "location": 0,
      "month_year_field": 0,
      "content_hash": 0,
      "language": 0,
      "hardiness_zone": 9
    }
  }
}
```

`contract_valid` means the chunk obeys the database payload contract.

`priority_full_metadata` means the chunk has usable:

```text
title
month_year
hardiness_zone
```

and is eligible for the strongest priority-retrieval strategy.

---

# 36. Qdrant Payload Indexes

Create payload indexes for:

```text
hardiness_zone
month_year
title
content_hash
```

These support:

- Priority metadata retrieval.
- Metadata filtering.
- Deduplication checks.

---

# 37. Embedding

Embedding model:

```text
BAAI/bge-base-en-v1.5
```

Embedding should happen only after metadata validation.

Use batches rather than one chunk at a time.

Conceptual flow:

```text
validated chunks
       ↓
chunk queue
       ↓
large GPU embedding batches
       ↓
vectors
       ↓
Qdrant batch upsert
```

The actual content is preserved in:

- Canonical store.
- Qdrant payload `text`.

---

# 38. Qdrant Batch Writes

Avoid:

```text
chunk
→ embed
→ upsert
→ next
```

Prefer:

```text
many chunks
    ↓
batch embedding
    ↓
many vectors
    ↓
batch upsert
```

This improves throughput and GPU utilization.

---

# 39. `attempts` Table

Keep an audit trail of every retry/failure.

```sql
CREATE TABLE attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,

    unit_type TEXT NOT NULL,
    unit_id TEXT NOT NULL,

    stage TEXT NOT NULL,

    attempt_number INTEGER NOT NULL,

    status TEXT NOT NULL,

    started_at TEXT NOT NULL,
    completed_at TEXT,

    error_type TEXT,
    error_message TEXT
);
```

Possible units:

```text
document
qualification_chunk
rag_chunk
```

Possible stages:

```text
extract
classify
metadata
embed
qdrant_upsert
```

---

# 40. State Completion Condition

A run can move to validation only when no non-terminal work remains.

Examples:

No `run_documents` in active states.

No qualification chunks in:

```text
pending
processing
failed
```

No RAG chunks in:

```text
pending
metadata_enriching
embedding
embedded
qdrant_pending
failed
```

Everything must be either:

```text
successful
rejected
duplicate_skipped
skipped_early_stop
permanently_failed
```

Then:

```text
processing
    ↓
retrying
    ↓
terminal-state check
    ↓
validating
    ↓
snapshotting
    ↓
complete
```

---

# 41. Run Validation

Before snapshot creation, validate:

- No non-terminal units remain.
- Qdrant point count is consistent with indexed chunks.
- Required Qdrant metadata fields exist.
- Metadata quality statistics are calculated.
- Sample retrieval queries work.
- Permanent failures are explicitly counted.
- Duplicate counts are recorded.

---

# 42. Snapshot Automation

Run finalization should be one automated operation:

```python
finalize_run(...)
```

Conceptually:

```text
state processing finishes
        ↓
terminal-state check
        ↓
validate Qdrant counts
        ↓
sample retrieval tests
        ↓
create snapshot
        ↓
download snapshot
        ↓
calculate SHA-256 checksum
        ↓
generate manifest
        ↓
atomically update current state in
crop_occurrences.json
        ↓
create run directory
        ↓
mark run COMPLETE
```

The user should not manually create/move snapshots during the real 50-state build.

---

# 43. `snapshots` Table

```sql
CREATE TABLE snapshots (
    snapshot_id TEXT PRIMARY KEY,

    build_id TEXT NOT NULL,
    run_id TEXT NOT NULL,

    collection_name TEXT NOT NULL,

    snapshot_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,

    checksum_sha256 TEXT NOT NULL,

    qdrant_point_count INTEGER NOT NULL,

    created_at TEXT NOT NULL,

    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
```

---

# 44. Run Manifest

Example:

```json
{
  "schema_version": "1.0",

  "build_id": "build_2026_08",
  "run_id": "002_IN",

  "sequence_number": 2,

  "state_processed": {
    "name": "Indiana",
    "code": "IN"
  },

  "states_included": [
    "IL",
    "IN"
  ],

  "previous_run": "001_IL",

  "collection_name": "mirage_base_build",

  "versions": {
    "extractor": "v1",
    "classifier": "v1",
    "chunker": "v1",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "qdrant": "1.18.0"
  },

  "this_run": {
    "sources_discovered": 25000,
    "duplicates_skipped": 843,

    "documents_accepted": 14831,
    "documents_rejected": 9289,
    "documents_permanently_failed": 37,

    "qualification_chunks_processed": 104223,
    "qualification_chunks_permanently_failed": 11,

    "rag_chunks_created": 382911,
    "rag_chunks_indexed": 382902,
    "rag_chunks_permanently_failed": 9
  },

  "metadata_quality": {
    "contract_valid": 382902,
    "priority_full_metadata": 310215,
    "month_year_known": 310224,
    "month_year_unknown": 72687,
    "hardiness_zone_ak_hi_exception": 0
  },

  "cumulative": {
    "states": 2,
    "documents": 29122,
    "rag_chunks": 751338,
    "qdrant_points": 751338
  },

  "snapshot": {
    "file": "mirage_base_002_IN.snapshot",
    "sha256": "...",
    "size_bytes": 123456789
  },

  "crop_dictionary": {
    "file": "crop_occurrences.json",
    "mode": "cumulative_single_file",
    "state_key": "indiana",
    "crop_count": 62
  },

  "started_at": "...",
  "completed_at": "..."
}
```

---

# 45. Notebook Configuration and Relative Input Contract

The complete builder directory is portable. The notebook uses:

```python
BASE_DIR = Path.cwd().resolve()
```

so the entire directory may be moved as long as the same **relative layout and filename contracts** are preserved.

Typical directory:

```text
Database/
├── MetaMIRAGE_Cumulative_Qdrant_Preload.ipynb
│
├── Illinois-PDFS.zip
├── Illinois-CSV.zip
├── Illinois-URL.xlsx
│
├── county_state_hardiness_zone.csv
├── crop_occurrences.json
│
├── pipeline_state.db
├── canonical/
├── runs/
│
└── qdrant/
    ├── bin/
    │   └── qdrant
    └── storage/
```

The state itself is **explicitly configured**. It is not inferred from input filenames:

```python
BUILD_ID = "build_2026_08"

STATE_NAME = "Illinois"
STATE_CODE = "IL"
RUN_SEQUENCE = 1

RUN_ID = f"{RUN_SEQUENCE:03d}_{STATE_CODE.upper()}"
```

This explicit state drives:

```text
STATE_NAME / STATE_CODE
        │
        ├── run ID
        ├── SQLite run identity
        ├── location metadata
        ├── hardiness-zone derivation
        ├── crop_occurrences.json state section
        └── manifest state identity
```

## 45.1 Automatic Source Discovery

Source files are discovered directly under `BASE_DIR` using case-insensitive filename contracts.

At a given run there may be at most one matching source archive/file of each type:

```text
*PDF*.zip
    → PDF archive

*CSV*.zip
    → CSV archive

*URL*.txt
*URL*.xlsx
*URL*.xlsm
*URL*.xls
    → URL input file
```

Examples:

```text
Illinois-PDFS.zip
Illinois_CSV_Data.zip
Illinois-URL.xlsx
```

The exact state name is not required in the filename; only the `PDF`, `CSV`, or `URL` marker is used for source-type discovery.

If more than one matching file of the same type exists, the notebook must fail rather than guess.

All source types are optional individually. A run must have at least one of PDF, CSV, or URL input before preflight can pass.

## 45.2 PDF ZIP Behavior

There is at most one PDF ZIP for a run.

The notebook:

```text
detect *PDF*.zip
      ↓
extract archive into the current run workspace
      ↓
recursively discover all .pdf files
(case-insensitive extension)
      ↓
process every PDF
```

The original ZIP remains unchanged.

## 45.3 CSV ZIP Behavior

There is at most one CSV ZIP for a run.

The notebook:

```text
detect *CSV*.zip
      ↓
extract archive into the current run workspace
      ↓
recursively discover all .csv files
      ↓
register every CSV as an input
```

CSV files use generic automatic field detection for common:

```text
title
date
url
county
```

fields when available.

Every CSV row remains one canonical document.

## 45.4 URL File Behavior

The URL input is the single file whose filename contains `URL`.

Supported formats:

```text
.txt
.xlsx
.xlsm
.xls
```

For text files, HTTP/HTTPS URLs are extracted from non-comment lines.

For Excel files, URL strings and hyperlink targets are extracted across workbook sheets.

Duplicate URLs are removed before source registration.

## 45.5 Fixed Relative Support Files

These paths remain fixed relative to `BASE_DIR`:

```python
CROP_OCCURRENCE_JSON = BASE_DIR / "crop_occurrences.json"
HARDINESS_CSV = BASE_DIR / "county_state_hardiness_zone.csv"

QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_COLLECTION = "mirage_base_build"
```

The Qdrant server itself is a separate process and must already be running on the same compute node before the pipeline starts.

## 45.6 Run Configuration

Core settings remain:

```python
MAX_STAGE_RETRIES = 2

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

RAG_CHUNK_SIZE = 480
RAG_CHUNK_OVERLAP = 80
RAG_HARD_CAP = 512

QUALIFICATION_CHUNK_CHARS = 7000
QUALIFICATION_OVERLAP_CHARS = 700
MAX_QUALIFICATION_CHUNKS = 20
```

The Hugging Face token may come from:

```text
HF_TOKEN environment variable
```

or may be requested interactively when the qualification classifier is first loaded.

## 45.7 Run Safety Switch

`RUN_PIPELINE` is an explicit safety switch.

The intended execution order is:

```text
configuration
      ↓
input discovery / setup
      ↓
preflight
      ↓
RUN_PIPELINE = True
      ↓
execute pipeline
```

Changing the run switch must not reset already discovered inputs.

Operationally, the safest pattern is to set `RUN_PIPELINE = True` in a small run-control cell immediately before the execution cell rather than rerunning the entire configuration cell.

The preflight must reject a run if no source is configured:

```text
URL_FILE is None
AND PDF_ZIP_FILE is None
AND no CSV inputs
        ↓
fail before processing
```

It must also verify that Qdrant is reachable before expensive work begins.

---

# 46. Idempotency Requirement

The notebook must be safe to rerun.

Accidentally rerunning orchestration cells should not:

- Duplicate Qdrant points.
- Reclassify already completed chunks.
- Re-extract completed documents.
- Rebuild canonical content unnecessarily.

This is achieved with:

- SQLite status checks.
- Deterministic content hashes.
- Deterministic document IDs.
- Deterministic RAG chunk IDs.
- Idempotent Qdrant upserts.

---

# 47. Final Pipeline State Machine

```text
                     SOURCE
                       │
                       ▼
                   discovered
                       │
                       ▼
                    extract
                       │
              ┌────────┴────────┐
              │                 │
            failed            success
              │                 │
          retry later            ▼
                        canonical document
                                 │
                          global duplicate?
                           ┌─────┴─────┐
                           │           │
                          yes         no
                           │           │
                    duplicate_skipped  ▼
                                  qualification
                                       │
                                 qualification chunks
                                       │
                           ┌───────────┴────────────┐
                           │                        │
                         failed                  succeeded
                           │                        │
                     retry queue                    ▼
                                              document decision
                                            ┌───────┴────────┐
                                            │                │
                                         rejected          accepted
                                            │                │
                                         terminal            ▼
                                                        RAG chunks
                                                            │
                                                   metadata enrichment
                                                            │
                                                   contract validation
                                                            │
                                                        embedding
                                                            │
                                                       Qdrant upsert
                                                            │
                                                   ┌────────┴────────┐
                                                   │                 │
                                                 failed            indexed
                                                   │                 │
                                              retry queue          terminal
                                                   │
                                         retries exhausted
                                                   │
                                                   ▼
                                          permanently_failed
                                                   │
                                                 terminal
```

After no non-terminal work remains:

```text
validate
   ↓
snapshot
   ↓
manifest
   ↓
atomically update current state in
crop_occurrences.json
   ↓
run folder
   ↓
mark run COMPLETE
   ↓
next state
```

---

# 48. Locked Architecture Decisions

The following are considered finalized:

1. Jupyter is the offline pipeline orchestrator.
2. Raw content is extracted only once.
3. Canonical normalized content is persisted.
4. SQLite stores processing state.
5. Global deduplication is based on normalized content hash.
6. Moved/renamed identical PDFs are the same document.
7. CSV row = one canonical document.
8. Qualification chunks and RAG chunks remain separate.
9. `tag != "msc"` means accepted.
10. Qualification outputs also feed the crop dictionary builder.
11. Failed work is retried within the current state run.
12. Default maximum stage retries = 2 after the initial attempt.
13. Remaining failures become `permanently_failed`.
14. A state completes only after all work reaches a terminal state.
15. One cumulative live Qdrant collection grows across all 50 states.
16. Snapshots, not live collection names, are the build versions.
17. Each state automatically creates a run folder with its cumulative snapshot and manifest; crop enrichment is kept in one cumulative root-level `crop_occurrences.json`.
18. RAG point IDs are deterministic.
19. Embeddings and Qdrant writes are batched.
20. RAG chunking matches runtime:
    - tokenizer `BAAI/bge-base-en-v1.5`
    - 480 tokens
    - 80 overlap
    - 512-token hard cap
21. Web <=512 tokens stays one chunk.
22. PDF pages are chunked independently.
23. CSV rows use web-like chunking behavior.
24. Metadata enrichment and contract validation are a hard ingestion gate.
25. No chunk is embedded/upserted before metadata validation.
26. Qdrant payload matches the inference retrieval contract.
27. `location` derives from the current state, with county when available.
28. Hardiness-zone logic is rebuilt independently using the supplied mapping.
29. Alaska/Hawaii may use `hardiness_zone=""` as a documented exception.
30. `month_year` always exists and may be `""` when unknown.
31. Dates are populated when determinable and never fabricated.
32. Metadata quality is recorded in every run manifest.
33. Qdrant payload indexes include:
    - `hardiness_zone`
    - `month_year`
    - `title`
    - `content_hash`
34. Run finalization automatically validates, snapshots, checksums, manifests, atomically updates the current state's section in `crop_occurrences.json`, and stores outputs.
35. The notebook must be idempotent and resume from persisted state rather than reprocessing completed work.
36. Input paths are relative to the movable builder directory (`BASE_DIR`).
37. PDF input is the single ZIP whose filename contains `PDF`; the notebook extracts and processes all PDFs inside it.
38. CSV input is the single ZIP whose filename contains `CSV`; the notebook extracts and processes all CSVs inside it.
39. URL input is the single file whose filename contains `URL` and may be `.txt`, `.xlsx`, `.xlsm`, or `.xls`.
40. More than one matching PDF ZIP, CSV ZIP, or URL file is an error; the notebook never guesses.
41. The current state is explicitly configured and drives run identity, location metadata, hardiness-zone derivation, crop-dictionary state selection, and manifest identity.
42. `crop_occurrences.json` is a single cumulative state-keyed artifact that preserves occurrence baselines and successive disease/pest/management enrichment.
43. `RUN_PIPELINE` is a safety switch and should be changed without rerunning input-resetting configuration logic.

---

# 49. Current Implementation Status

The architecture is considered frozen and is implemented by the MetaMIRAGE cumulative Qdrant preload notebook.

Before a real state run:

1. Start the local Qdrant server and verify `http://127.0.0.1:6333/collections` is reachable.
2. Place the current run's PDF ZIP, CSV ZIP, and/or URL file directly under `BASE_DIR` using the filename contracts above.
3. Keep `county_state_hardiness_zone.csv` and the cumulative `crop_occurrences.json` at their fixed relative locations.
4. Configure the current state/run identity.
5. Run setup/input discovery and confirm the detected inputs.
6. Keep the pipeline off while reviewing/preflighting.
7. Enable `RUN_PIPELINE` only after discovery and preflight are correct.
8. Let SQLite/deterministic IDs handle restart and idempotency if execution is interrupted.
9. On completion, preserve the cumulative snapshot, manifest, canonical store, SQLite ledger, and updated `crop_occurrences.json` for the next state.

The cumulative build then proceeds state-by-state until all planned states have been processed.
