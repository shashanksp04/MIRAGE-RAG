# MetaMIRAGE Inference Database Architecture — Implementation Plan

## 0. Purpose

This document is the implementation plan for changing MetaMIRAGE inference from a **single mutable Qdrant collection** into a **base + runtime collection architecture**.

The normal/final inference path uses two collections:

1. **`mirage_base`**
   - Preloaded, curated knowledge.
   - Exists before normal/final inference starts.
   - Read-only from inference.
   - Never reset, recreated, mutated, or deleted by inference code.
   - May be disabled with `USE_BASE_COLLECTION=False` during development/testing while the curated base is still being built.

2. **`mirage_runtime_<ABLATION_ID>_<YYYYMMDD>_<HHMMSS>`**
   - Created specifically for one inference run.
   - Starts empty for a new run.
   - Receives runtime web/PDF knowledge discovered during that run.
   - Is shared across all successive queries in the same run.
   - Is preserved after an interrupted/failed run so inference can resume.
   - Is optionally snapshotted after successful completion.
   - Is deleted after successful completion.
   - Is deleted and recreated when the user explicitly requests a fresh restart.

The core goal is to prevent runtime web augmentation from contaminating the curated base database or leaking knowledge across independent experimental runs while still allowing knowledge accumulated earlier in the **same** inference run to benefit later queries.

`USE_BASE_COLLECTION=False` is **not a second architecture**. It is a narrow conditional mode of the same implementation: skip base verification, base retrieval, and base-side deduplication while keeping the same runtime lifecycle, retriever, ingestion path, confidence flow, and evaluation code.

This plan is intentionally implementation-oriented and should be used as context for an engineering agent modifying the current MetaMIRAGE codebase.

---

# 1. Current System Context

The current runtime RAG path already uses Qdrant server mode.

Important current components include:

```text
rag_agent/main.py
    MainAgent composition/root

rag_agent/utils/qdrant_store.py
    Qdrant collection management
    embedding + upsert
    retrieval
    content-hash deduplication
    payload indexes

rag_agent/utils/ContentUtils.py
    query embedding
    priority retrieval
    metadata-filter translation
    chunking/hash helpers

rag_agent/tools/web_addition.py
    runtime web ingestion

rag_agent/tools/pdf_addition.py
    runtime PDF ingestion

rag_agent/tools/confidence_evaluator.py
    confidence scoring

Inference/generate.py
    batch inference orchestration
    RAG workers
    query/evaluation checkpoint/resume behavior
    JSONL output

rag_agent/ablation_configs.json
    ablation-specific feature controls
```

The current runtime architecture has one important reproducibility problem:

```text
retrieve from Qdrant
      ↓
low confidence
      ↓
web search
      ↓
runtime ingestion
      ↓
same Qdrant collection is mutated
```

Therefore one experiment can alter the database subsequently seen by later experiments.

This update replaces that behavior with explicit **base/runtime isolation**.

---

# 2. Locked Architecture Decisions

The following decisions are considered final.

## 2.1 Base collection

Use:

```python
BASE_COLLECTION = "mirage_base"
```

This is a placeholder name and may be renamed later, but all implementation should assume `mirage_base` for now.

Inference code must treat it as immutable.

Allowed operations:

```text
collection existence check
search
scroll/read for deduplication
count/health inspection if needed
```

Forbidden operations:

```text
create
recreate
reset
delete
upsert
update
payload modification
runtime ingestion
```

No runtime content may ever be promoted into `mirage_base`.

The base database is curated offline.

Base participation is controlled by:

```python
USE_BASE_COLLECTION = True
```

Semantics:

```text
True
→ base collection is mandatory
→ verify mirage_base exists
→ retrieve from base + runtime
→ deduplicate runtime ingestion against base + runtime

False
→ skip base existence verification
→ do not instantiate/use a base store
→ retrieve from runtime only
→ deduplicate runtime ingestion against runtime only
```

`False` is intended for development/testing while the curated preload database is unavailable. It must use the **same code path with conditional base participation**, not a separate RAG architecture.

---

## 2.2 Runtime collection

Each new inference run gets:

```text
mirage_runtime_<ABLATION_ID>_<YYYYMMDD>_<HHMMSS>
```

Example:

```text
mirage_runtime_full_system_20260824_145102
```

The timestamp is generated only when a genuinely **new** runtime collection is required.

A resumed run reuses the existing collection name and must not generate a new timestamp.

---

## 2.3 Runtime knowledge scope

Runtime knowledge is **run-scoped**, not query-scoped.

This is intentional:

```text
Run A
│
├── query 0 → web content A
├── query 1 → can retrieve content A
├── query 2 → adds content B
└── query 3 → can retrieve A + B
```

However:

```text
Run B
```

must never retrieve runtime knowledge from Run A.

Every new run therefore begins with:

```text
USE_BASE_COLLECTION=True
    mirage_base
    +
    empty runtime collection

USE_BASE_COLLECTION=False
    empty runtime collection only
```

unless it is explicitly resuming an interrupted run.

Do not add a mandatory `query_id` filter to runtime retrieval.

---

## 2.4 Runtime modes

Use one explicit mode instead of multiple potentially conflicting booleans:

```python
RUNTIME_MODE = "resume"
```

Allowed values:

```text
"resume"
"fresh"
```

Optional manual override:

```python
RUNTIME_COLLECTION_OVERRIDE = None
```

Snapshot toggle:

```python
SNAPSHOT_RUNTIME = False
```

---

## 2.5 Base participation mode

Use:

```python
USE_BASE_COLLECTION = True
```

for normal/final runs.

Use:

```python
USE_BASE_COLLECTION = False
```

only when the curated base collection is not yet available or when deliberately testing runtime-only behavior.

Important:

- This flag changes only whether the base branch participates.
- Runtime collection lifecycle is unchanged.
- Runtime collection naming is unchanged.
- `resume` / `fresh` semantics are unchanged.
- Confidence evaluation is unchanged.
- Web/PDF ingestion is unchanged except that base-side duplicate checking is skipped.
- Do not create a separate retriever/agent stack for this mode.
- Never auto-create an empty `mirage_base` when the flag is `False`.

---

## 2.6 Frozen embedding / Qdrant compatibility contract

The new-architecture preload database uses:

```text
Embedding model:        BAAI/bge-base-en-v1.5
Vector dimension:       768
Distance metric:        COSINE
Preload batch size:     64
Preload device:         CUDA if available, otherwise CPU
Preload normalization:  normalize_embeddings=False
Payload indexes:
    hardiness_zone
    month_year
    title
    content_hash
```

The runtime RAG uses the same:

```text
Embedding model:   BAAI/bge-base-en-v1.5
Vector dimension:  768
Distance metric:   COSINE
```

but currently embeds with:

```python
normalize_embeddings=True
```

This is compatible with Qdrant cosine distance because cosine comparison is scale-invariant and Qdrant normalizes vectors for cosine scoring.

Therefore:

- **Do not change the runtime normalization behavior solely to match preload.**
- **Do not change the preload vectors solely to match runtime normalization.**
- The model name and vector dimension are strict invariants and must remain unchanged.
- Runtime collection creation must use `size=768` and `Distance.COSINE`.
- Runtime payload indexes must include `hardiness_zone`, `month_year`, `title`, and `content_hash`.

Embedding batch size and CUDA/CPU choice are execution details, not collection-schema compatibility requirements.

---

# 3. Runtime Mode Semantics

## 3.1 `RUNTIME_MODE = "resume"`

This is the normal/default mode.

Behavior:

```text
look for runtime collections matching current ABLATION_ID
        ↓
matching interrupted runtime exists?
        │
   ┌────┴────┐
   │         │
  yes        no
   │         │
   ▼         ▼
reuse       generate timestamp
latest      create fresh runtime
runtime
   │         │
   └────┬────┘
        ↓
continue inference
```

A matching collection is:

```text
mirage_runtime_<current ABLATION_ID>_<YYYYMMDD>_<HHMMSS>
```

If several matching collections exist unexpectedly:

1. Parse their timestamp suffix.
2. Select the newest one for default resume.
3. Log all candidates.
4. Log which one was selected.

Do not create a new runtime collection when a valid resumable one exists.

The existing inference query/evaluation checkpoint logic must also resume as it does today.

Do **not** add another progress/checkpoint file just for database state.

---

## 3.2 `RUNTIME_MODE = "fresh"`

This means:

> Abandon prior interrupted state for this ablation and start again from query 0 with an empty runtime database.

Behavior:

```text
find matching runtime collection(s)
        ↓
delete old runtime collection(s)
        ↓
reset/restart existing inference/eval progress
        ↓
generate NEW timestamp
        ↓
create new empty runtime collection
        ↓
start inference from query 0
```

The base collection must never be touched.

For a clean semantic implementation, delete any live runtime collections matching:

```text
mirage_runtime_<current ABLATION_ID>_*
```

before the new one is created.

This is safe because successful runtime collections are deleted automatically; therefore matching live collections represent interrupted/abandoned state.

Snapshot artifacts, if any, are not live collections and are not affected by this cleanup.

---

## 3.3 Override semantics

If:

```python
RUNTIME_COLLECTION_OVERRIDE = "mirage_runtime_..."
```

then this is an explicit manual resume target.

Rules:

```text
RUNTIME_MODE == "resume"
    override allowed

RUNTIME_MODE == "fresh"
    override forbidden
```

If an override is supplied with `fresh`, fail configuration validation rather than guessing or deleting the explicitly named debugging collection.

When override is used:

1. Verify the collection exists.
2. Verify its name starts with `mirage_runtime_`.
3. Preferably verify its embedded ablation ID matches the current ablation.
4. Use exactly that collection.

---

# 4. High-Level Target Architecture

```text
                              QDRANT SERVER
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
                    ▼                              ▼
             ┌─────────────┐              ┌──────────────────────┐
             │ mirage_base │              │ mirage_runtime_<ID>  │
             │             │              │                      │
             │ READ ONLY   │              │ READ + WRITE         │
             │ CURATED     │              │ RUN-SCOPED           │
             └──────┬──────┘              └──────────┬───────────┘
                    │                                │
                    └──────────────┬─────────────────┘
                                   ▼
                         DualCollectionRetriever
                                   │
                         merge + deduplicate
                                   │
                          progressive strategy
                                   │
                              confidence
                                   │
                    ┌──────────────┴─────────────┐
                    │                            │
                 sufficient                    low
                    │                            │
                    │                        web search
                    │                            ↓
                    │                    extract/normalize
                    │                            ↓
                    │                  cross-collection dedupe
                    │                            ↓
                    │                  metadata validation
                    │                            ↓
                    │                  runtime collection only
                    │                            ↓
                    │                       retrieve again
                    └──────────────┬─────────────┘
                                   ▼
                             final evidence
                                   ↓
                              generation
```

---

# 5. Qdrant Server Responsibility

MetaMIRAGE inference does **not** own the base database storage directory.

The Qdrant server will be started externally by the user from the correct persistent location.

Therefore inference must not:

```text
discover Qdrant storage directories
restore the base snapshot
copy the base database
look for preload files
read database files directly
```

Inference only needs:

```python
QDRANT_URL
QDRANT_API_KEY  # optional
```

and communicates with the already-running Qdrant server over HTTP.

Startup contract:

```text
Qdrant is already running
        ↓
inference connects
        ↓
USE_BASE_COLLECTION?
   ┌────┴────┐
  True      False
   │          │
verify        │
mirage_base   │
exists        │
   │          │
   └────┬─────┘
        ↓
create/resume runtime collection
```

When:

```python
USE_BASE_COLLECTION = True
```

`mirage_base` must exist or inference fails before workers begin processing queries.

When:

```python
USE_BASE_COLLECTION = False
```

skip base existence verification entirely and continue with runtime-only retrieval. Do not create a placeholder/empty base collection.

---

# 6. Hardcoded Vector / Embedding Contract

Do not dynamically infer the vector configuration from the base collection.

Use the known preload/inference contract directly.

## 6.1 Qdrant schema invariants

```python
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768
DISTANCE = Distance.COSINE
```

Runtime collection payload indexes:

```text
hardiness_zone
month_year
title
content_hash
```

These are the collection-level invariants that must remain compatible between the curated base and runtime collection.

## 6.2 Preload embedding behavior

The offline preload pipeline currently uses:

```text
Embedding model:       BAAI/bge-base-en-v1.5
Vector size:           768
Distance:              COSINE
Embedding batch size:  64
Device:                CUDA if available, otherwise CPU
normalize_embeddings:  False
```

## 6.3 Runtime embedding behavior

The runtime RAG uses:

```text
Embedding model:       BAAI/bge-base-en-v1.5
Vector size:           768
Distance:              COSINE
normalize_embeddings:  True
```

Do not change this simply to make the normalization flag identical to preload.

This remains compatible with Qdrant cosine distance:

```text
preload vectors may be unnormalized
runtime query/runtime-ingestion vectors may be normalized
        ↓
Qdrant COSINE comparison
        ↓
vector magnitude does not alter cosine relevance ordering
```

The strict compatibility requirements are:

```text
same embedding model
same vector dimension
same cosine distance metric
same retrieval metadata semantics
```

The following are execution details and do **not** need to match:

```text
embedding batch size
CUDA vs CPU
normalize_embeddings=True/False
```

The model name and dimension must remain unchanged.

---

# 7. Chunking and Canonical Metadata Contract

Do not redesign chunking as part of this change.

Runtime ingestion must preserve the same inference/preload behavior already established.

Embedding/tokenizer:

```text
BAAI/bge-base-en-v1.5
```

RAG chunking:

```text
480-token chunk
80-token overlap
512-token hard cap
```

Canonical payload keys must remain compatible with base retrieval:

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

Do not fabricate `month_year`.

Unknown values should continue using the project's canonical unknown representation.

---

# 8. Minimal Runtime-Specific Provenance

Do not replicate large amounts of run metadata on every point.

Because the runtime collection itself encodes the run identity, these are unnecessary per point:

```text
runtime_run_id
ablation_id
query_id
full config
model config
snapshot settings
```

If runtime provenance is useful, keep additions minimal:

```python
{
    "ingested_at": "...",
    "search_query": "..."
}
```

Only add these if they fit naturally into the existing metadata builder.

Do not require them for retrieval.

Run-wide provenance belongs in logs/run reports/manifests, not thousands of repeated Qdrant payloads.

---

# 9. New Core Abstraction: `InferenceDatabaseManager`

Introduce one lifecycle owner for base/runtime database setup.

Suggested responsibility:

```text
InferenceDatabaseManager
│
├── connect()
├── validate_configuration()
├── verify_base_collection_if_enabled()
├── list_matching_runtime_collections()
├── choose_runtime_collection()
├── create_runtime_collection()
├── delete_matching_runtime_collections()
├── create_runtime_payload_indexes()
├── snapshot_runtime_if_enabled()
├── finalize_success()
└── preserve_failure()
```

This component should own **collection lifecycle**, not retrieval semantics.

It may live in a new file such as:

```text
rag_agent/utils/inference_database_manager.py
```

or another location consistent with the existing project structure.

Avoid putting the entire lifecycle directly into `MainAgent`.

---

# 10. Database Manager Configuration

Suggested constructor/config inputs:

```python
InferenceDatabaseManager(
    qdrant_client=client,
    base_collection="mirage_base",
    use_base_collection=USE_BASE_COLLECTION,
    ablation_id=ABLATION_ID,
    runtime_mode=RUNTIME_MODE,
    runtime_collection_override=RUNTIME_COLLECTION_OVERRIDE,
    snapshot_runtime=SNAPSHOT_RUNTIME,
    vector_size=768,
    distance=Distance.COSINE,
)
```

Validation:

```text
USE_BASE_COLLECTION is boolean

if USE_BASE_COLLECTION=True:
    base collection name must be non-empty

ablation ID non-empty
runtime mode in {"resume", "fresh"}

if override is not None:
    runtime mode must be "resume"
```

Do not require `mirage_base` to exist when `USE_BASE_COLLECTION=False`.

Runtime collection names must be sanitized for Qdrant-safe naming.

If `ABLATION_ID` can contain spaces or punctuation, normalize it consistently:

```text
spaces → _
unsafe characters → _
collapse repeated _
```

Do not change the logical ablation ID used elsewhere; only sanitize the collection-name component.

---

# 11. Runtime Collection Naming

New collection:

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

runtime_collection = (
    f"mirage_runtime_{safe_ablation_id}_{timestamp}"
)
```

Example:

```text
mirage_runtime_full_system_20260824_145102
```

Timestamp generation occurs only in:

```text
new run
or
fresh restart
```

Never regenerate it for resume.

---

# 12. Runtime Collection Discovery

Implement a helper that lists Qdrant collections and finds names matching the current ablation.

Conceptually:

```python
prefix = f"mirage_runtime_{safe_ablation_id}_"
```

Matching collection:

```text
starts with prefix
AND suffix parses as YYYYMMDD_HHMMSS
```

Do not accidentally match:

```text
another ablation
mirage_base
unrelated developer collections
```

Sort candidates by parsed timestamp descending.

Return:

```python
[
    newest,
    ...,
    oldest,
]
```

This discovery replaces the need for an `active_run.json`.

---

# 13. Runtime Collection Creation

When a new runtime is needed:

1. Generate timestamp.
2. Construct name.
3. Assert the name does not already exist.
4. Create the Qdrant collection with hardcoded vector settings.
5. Create payload indexes required by current retrieval/deduplication.

Required runtime indexes are frozen and must be created on every new runtime collection:

```text
hardiness_zone
month_year
title
content_hash
```

This mirrors the preload collection indexing contract.

Do not add `query_id`.

Additional indexes should only be added if an existing runtime feature actually filters on them.

---

# 14. Base Store vs Runtime Store

The current system effectively exposes one Qdrant store.

Refactor so the inference runtime has a mandatory runtime store and an optional base store:

```python
base_store = (
    QdrantStore(
        ...,
        collection_name=BASE_COLLECTION,
    )
    if USE_BASE_COLLECTION
    else None
)

runtime_store = QdrantStore(
    ...,
    collection_name=runtime_collection,
)
```

This is still one architecture. The base branch is simply disabled when `base_store is None`.

However, do not allow both stores to have identical mutating responsibilities.

The base path should be read-only by architecture.

Preferred options:

### Option A — read-only wrapper

```text
ReadOnlyQdrantStore
    wraps base QdrantStore

    allowed:
        search
        content_hash_exists
        count/read

    no:
        upsert
        reset
        delete
```

### Option B — strict call-site separation

If adding a wrapper is unnecessarily invasive, enforce that:

```text
retriever receives base_store + runtime_store
ingesters receive runtime_store only
```

and no ingestion component receives `base_store` except a narrow read-only duplicate checker.

Option A is safer if easy to implement.

---

# 15. New Core Abstraction: `DualCollectionRetriever`

Do not scatter two separate searches throughout `MainAgent`.

Add one logical retriever responsible for:

```text
optional base search
runtime search
result provenance
merge
cross-collection deduplication
top-k behavior
priority-filter strategy evaluation
```

It must gracefully support:

```text
USE_BASE_COLLECTION=True
    → base + runtime search

USE_BASE_COLLECTION=False
    → runtime search only
```

Do not introduce a second `RuntimeOnlyRetriever`.

Suggested conceptual interface:

```python
retriever.retrieve_with_priority_filters(
    query_text=...,
    location=...,
    month_year=...,
    title=...,
    k=...,
    min_results=...,
)
```

The rest of the agent should think of this as one knowledge system.

---

# 16. Progressive Retrieval With Two Collections

The current priority ladder should remain:

```text
1. hardiness_zone + month_year + title
2. hardiness_zone + title
3. title
4. month_year
5. hardiness_zone + month_year
6. hardiness_zone
7. semantic_only
```

For **each strategy**, always search the runtime collection and search the base collection only when `USE_BASE_COLLECTION=True`.

Current conceptual behavior:

```text
strategy
   ↓
search one collection
   ↓
score strategy
```

Target behavior:

```text
strategy
   │
   ├── if USE_BASE_COLLECTION=True:
   │       search mirage_base
   │
   └── always:
           search active runtime
              ↓
        merge available result sets
              ↓
       cross-collection dedupe
              ↓
         global top-k
              ↓
      calculate strategy score
```

Evaluate all candidate strategies as the existing code does.

Do not change the meaning of confidence-scope strategy names.

---

# 17. Retrieval Result Provenance

Every result returned by the dual retriever should indicate where it came from:

```python
{
    ...
    "retrieval_source": "base"
}
```

or:

```python
{
    ...
    "retrieval_source": "runtime"
}
```

This field can be attached in memory by the retriever.

It does **not** need to be persisted in Qdrant.

This will support:

```text
debugging
experiment reporting
base/runtime contribution statistics
```

without adding redundant storage.

---

# 18. Merge and Score Behavior

Because both collections use the same embedding model and cosine metric, treat scores as directly comparable.

For each strategy:

```python
combined = base_results + runtime_results
```

Then:

```text
deduplicate
sort by similarity
take global top-k
```

Do not introduce base/runtime weighting in this update.

No:

```text
base bonus
runtime penalty
collection-specific score normalization
```

A chunk should win based on relevance, not collection origin.

Preserve the existing downstream score/distance compatibility contract.

---

# 19. Cross-Collection Result Deduplication

After retrieving from both collections, duplicate evidence may theoretically appear in both.

Deduplicate merged hits using:

```text
content_hash
```

Preferred tie behavior when base is enabled:

```text
same content_hash appears in base + runtime
    ↓
keep base copy
```

When `USE_BASE_COLLECTION=False`, merged-result deduplication simply operates within runtime results.

Reason:

```text
base is curated
base provenance should be preferred
runtime duplicate should normally have been blocked at ingestion anyway
```

If duplicate points occur within the same collection, keep the highest-scoring result unless existing behavior already defines another deterministic rule.

After dedupe, sort and take final top-k.

---

# 20. Runtime Ingestion Must Write Only to Runtime

Refactor:

```text
WebAddition
PDFAddition
other runtime ingestion paths
```

so all upserts target only:

```text
active runtime collection
```

Never pass the base store as their write target.

Desired dependency flow:

```text
MainAgent
│
├── DualCollectionRetriever
│     ├── base_store (read)
│     └── runtime_store (read)
│
├── WebAddition
│     └── runtime_store (write)
│
└── PDFAddition
      └── runtime_store (write)
```

This dependency graph is a critical safety invariant.

---

# 21. Cross-Collection Ingestion Deduplication

Before runtime content is inserted:

```text
source/document
      ↓
normalize
      ↓
content_hash
      ↓
USE_BASE_COLLECTION=True?
      │
   yes│ → check hash in base
      │       │
      │    found → skip runtime ingestion
      │
      └── otherwise continue
              ↓
check hash in runtime
      │
   found → skip runtime ingestion
      │
 not found
      ↓
chunk/embed/upsert runtime
```

When `USE_BASE_COLLECTION=False`, skip the base hash lookup entirely; do not replace it with another architecture.

The existing `content_hash_exists` Qdrant behavior should be reused.

Avoid embedding content unnecessarily if a document/chunk can be rejected by hash before embedding.

Track/log two duplicate reasons separately when practical:

```text
duplicate_against_base
duplicate_against_runtime
```

---

# 22. Important Deduplication Granularity

Preserve the project's existing content-hash semantics.

Do not silently redesign whether `content_hash` represents:

```text
whole document
or
individual chunk
```

as part of this architecture update.

Use the current ingestion/hash helpers consistently for both base checking and runtime checking.

The implementation agent should inspect the current `ContentUtils.compute_content_hash` and ingestion paths and wire the same value into both collection checks.

---

# 23. Confidence Evaluation

Confidence must evaluate the **merged evidence**.

Incorrect:

```text
base retrieve → base confidence
runtime retrieve → runtime confidence
```

Correct:

```text
base + runtime
      ↓
DualCollectionRetriever
      ↓
merged winning strategy
      ↓
ConfidenceEvaluator
```

Keep the existing confidence formula and thresholds unchanged.

Current model:

```text
Similarity    50%
Coverage      20%
Consistency   20%
Scope         10%
```

Thresholds:

```text
>= 0.75       high
>= 0.50       medium
<  0.50       low
```

This update changes where results come from, not how confidence itself is mathematically defined.

---

# 24. Web-Augmentation Loop

Preserve the current conceptual loop:

```text
query
  ↓
dual retrieval
  ↓
merged evidence
  ↓
confidence
  ↓
low?
  │
  ├── no → continue
  │
  └── yes
         ↓
      web search
         ↓
      source filtering
         ↓
      extraction
         ↓
      cross-collection duplicate check
         ↓
      canonical metadata
         ↓
      RAG chunking
         ↓
      embedding
         ↓
      runtime upsert
         ↓
      dual retrieval again
         ↓
      merged evidence
```

A page added because of query N is intentionally available to query N+1 within the same inference run.

---

# 25. MainAgent Refactor

Current `MainAgent` owns the Qdrant connection/store and passes it to retrieval/ingestion tools.

Refactor its composition to something conceptually like:

```text
MainAgent
│
├── embedding_function
│
├── base_store (optional; None when base disabled)
│
├── runtime_store
│
├── dual_retriever
│
├── confidence_evaluator
│
├── web_search
│
├── web_addition
│
├── pdf_addition
│
├── crop_query_enrichment
│
└── ADK agent/tools
```

`MainAgent` should receive or be told the **already selected active runtime collection name**.

Prefer lifecycle creation/selection outside individual workers so all RAG workers use the same collection.

---

# 26. Multi-Process / Rank-0 Ownership

`Inference/generate.py` runs multiple RAG workers.

Collection lifecycle must happen exactly once.

Recommended sequence:

```text
Inference driver / rank0 setup
        ↓
connect Qdrant
        ↓
verify mirage_base only if USE_BASE_COLLECTION=True
        ↓
resolve runtime collection
        ↓
create/delete/resume runtime as required
        ↓
signal runtime collection name
        ↓
start/initialize all RAG workers
        ↓
every worker connects to:
    mirage_base
    same active runtime collection
```

Do not let every worker independently run:

```text
fresh cleanup
runtime discovery
runtime creation
snapshot
deletion
```

Those operations must have a single owner.

The current rank-0-ready barrier can be adapted for this purpose.

---

# 27. Remove Old Single-Collection Reset Behavior

Any logic that resets/recreates the one RAG collection at inference startup must be replaced.

Old conceptual behavior:

```text
reset Qdrant collection
→ workers use it
```

Target behavior:

```text
verify mirage_base exists
→ NEVER reset base
→ resolve/create runtime
→ workers use both
```

Audit:

```text
MainAgent.reset_collection()
do_reset_collection
rank0 reset calls
old default collection names
```

Any reset functionality retained for tests must be runtime-only and must never accept `mirage_base`.

A safety guard in delete/reset helpers is strongly recommended:

```python
if collection_name == BASE_COLLECTION:
    raise RuntimeError("Refusing to mutate immutable base collection")
```

---

# 28. Integration With Existing Inference Checkpoint Logic

The current inference system already skips/completes previously processed query/evaluation items.

Reuse/copy that logic.

Do not add:

```text
active_run.json
database_progress.json
another SQLite database
another checkpoint file
```

Database lifecycle must align with existing inference progress.

Required coupling:

### Resume

```text
RUNTIME_MODE = "resume"
        ↓
reuse existing runtime collection
        ↓
resume existing query/eval progress
```

### Fresh

```text
RUNTIME_MODE = "fresh"
        ↓
delete old runtime collection(s)
        ↓
reset/ignore old query/eval progress
        ↓
start query 0
```

The implementation must prevent the inconsistent state:

```text
new empty runtime collection
+
checkpoint starts at query 500
```

or:

```text
old populated runtime collection
+
query progress restarted at 0 unintentionally
```

---

# 29. Crash / Interruption Behavior

On abnormal termination:

```text
Ctrl-C
worker crash
job preemption
exception
node/session termination
```

do **not** delete the active runtime collection.

The best-effort failure handler should print:

```text
Inference interrupted.
Runtime collection preserved:

mirage_runtime_<...>

Restart using RUNTIME_MODE="resume"
to continue this run.
```

Do not attempt final-success snapshot semantics after failure.

The collection itself is the database-side resume state.

If the process is killed so abruptly that no handler executes, Qdrant persistence still retains the collection, and next startup discovery can find it.

---

# 30. Fresh Restart After a Crash

Example state:

```text
mirage_base
mirage_runtime_full_system_20260824_145102
```

User chooses:

```python
RUNTIME_MODE = "fresh"
```

Startup must:

```text
find old matching runtime
        ↓
DELETE old matching runtime
        ↓
reset inference progress
        ↓
generate new timestamp
        ↓
create:
mirage_runtime_full_system_20260824_153044
        ↓
start query 0
```

The deleted runtime is intentionally not preserved.

Do not delete:

```text
mirage_base
runtime collections belonging to another ablation
snapshot files
```

---

# 31. Successful Completion Behavior

Only successful completion triggers automatic cleanup.

```text
all intended inference queries complete
        ↓
SNAPSHOT_RUNTIME?
   ┌────┴────┐
  yes       no
   │         │
   ▼         │
snapshot     │
runtime      │
   │         │
   └────┬────┘
        ↓
delete live runtime collection
        ↓
finish
```

This ordering matters.

If snapshotting is enabled and snapshot creation fails:

```text
do NOT delete runtime collection
```

Treat finalization as failed and preserve the runtime collection for inspection/retry.

---

# 32. Runtime Snapshot Toggle

Configuration:

```python
SNAPSHOT_RUNTIME = False
```

Development/testing default:

```text
False
```

Reason:

```text
avoid unnecessary storage
avoid snapshot overhead
rapid iteration
```

Final experiment:

```text
True
```

When enabled:

1. Request Qdrant snapshot for active runtime collection.
2. Wait for successful creation.
3. Log snapshot identity/location returned by Qdrant.
4. Include it in existing run reporting/manifest if such output exists.
5. Only then delete the live runtime collection.

Snapshot support should be isolated in the database manager.

---

# 33. Snapshot Storage Responsibility

The Qdrant server owns its snapshot/storage location.

The inference code should not assume the base database directory is inside the MetaMIRAGE repository.

For runtime snapshots:

```text
request snapshot through Qdrant API
record returned metadata/path/name
```

Do not hardcode the external storage filesystem unless the current Qdrant snapshot API already requires an explicit path.

The primary requirement is the toggle + successful snapshot-before-delete behavior.

---

# 34. Runtime Contribution Statistics

Where convenient, expose counts that distinguish base/runtime use.

Useful statistics:

```text
base_results_returned
runtime_results_returned

runtime_web_pages_ingested
runtime_pdf_documents_ingested
runtime_chunks_ingested

duplicates_against_base
duplicates_against_runtime

runtime_initial_points
runtime_final_points
```

Do not make these statistics block the architectural refactor.

Prefer integrating into the project's existing run reporting rather than creating another standalone file solely for these metrics.

---

# 35. Optional Run-Level Manifest / Report Integration

If the existing inference run report has a natural place for database context, record:

```json
{
  "base_collection": "mirage_base",
  "runtime_collection": "mirage_runtime_full_system_20260824_145102",
  "runtime_mode": "resume",
  "runtime_snapshot_enabled": false
}
```

Potential final values:

```json
{
  "runtime_final_points": 2941,
  "duplicates_against_base": 72,
  "duplicates_against_runtime": 28
}
```

Do not create a completely separate manifest infrastructure unless needed.

Use existing run/report outputs when possible.

---

# 36. Suggested Configuration Surface

The database-specific runtime configuration should remain small.

Target:

```python
BASE_COLLECTION = "mirage_base"

# Normal/final inference: True
# Development/runtime-only testing while base is unavailable: False
USE_BASE_COLLECTION = True

RUNTIME_MODE = "resume"       # "resume" | "fresh"

RUNTIME_COLLECTION_OVERRIDE = None

SNAPSHOT_RUNTIME = False
```

Existing:

```python
ABLATION_ID = "..."
QDRANT_URL = ...
QDRANT_API_KEY = ...
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
```

Hardcoded/runtime constants:

```python
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768
DISTANCE = Distance.COSINE

RUNTIME_NORMALIZE_EMBEDDINGS = True
```

The preload pipeline uses batch size `64`, CUDA when available (otherwise CPU), and `normalize_embeddings=False`. Those preload execution details do not require the runtime code to copy the same normalization flag.

Avoid adding unnecessary configuration knobs.

---

# 37. Command-Line Integration

If `Inference/generate.py` already uses argparse, expose these settings through CLI flags or wire them through the existing experiment/launcher configuration.

Suggested flags:

```text
--use_base_collection true|false
--runtime_mode resume|fresh
--runtime_collection_override <name>
--snapshot_runtime
```

If the project already has a cleaner boolean-flag convention, follow it; the important requirement is that the user can explicitly disable base participation without changing the rest of the runtime architecture.

Base collection may be:

```text
--base_collection mirage_base
```

or kept as a code/config default if that better matches the project's experiment conventions.

Do not require the user to manually construct runtime collection names.

The program should generate them.

---

# 38. Ablation ID and Runtime Isolation

`ABLATION_ID` is part of the runtime collection namespace.

Examples:

```text
mirage_runtime_static_rag_20260824_145102

mirage_runtime_uncertainty_aware_20260824_150622

mirage_runtime_full_domain_filtered_20260824_155011
```

Resume discovery must use the **current** ablation ID.

Therefore:

```text
Run A / ablation X
```

cannot resume:

```text
Run B / ablation Y
```

by default.

This gives experiment isolation without an extra state file.

---

# 39. Retrieval APIs

Current code likely assumes one collection inside `ContentUtils.retrieve_with_priority_filters`.

Refactor carefully.

Preferred implementation:

Move multi-collection orchestration into `DualCollectionRetriever`.

Pseudo-interface:

```python
class DualCollectionRetriever:
    def __init__(self, base_store, runtime_store, content_utils):
        ...

    def retrieve_with_priority_filters(
        self,
        query,
        location=None,
        month_year=None,
        title=None,
        k=5,
        min_results=1,
    ):
        ...
```

Reuse current helper logic for:

```text
query embedding
hardiness-zone derivation
filter attempt construction
similarity calculation
strategy winner selection
```

Do not duplicate these formulas in two locations if avoidable.

---

# 40. Search Store API

Ensure `QdrantStore` can search a named store without lifecycle side effects.

Ideal separation:

```text
collection management:
    ensure/create/delete

search:
    search(query_vector, filter, limit)

dedupe:
    content_hash_exists(hash)

mutation:
    upsert(...)
```

Creating a `QdrantStore` for `mirage_base` must not silently create the collection if it is missing.

This is critical.

Base setup should use:

```text
require_existing=True
```

or equivalent behavior.

Runtime setup may use:

```text
create explicitly through DatabaseManager
```

Avoid constructors with surprising create-if-missing side effects.

---

# 41. Base Availability Behavior

## `USE_BASE_COLLECTION=True`

If:

```text
mirage_base
```

does not exist:

fail before starting RAG workers.

Error should clearly say:

```text
Required base Qdrant collection 'mirage_base' was not found.

Start Qdrant using the database location containing the curated base
collection, or explicitly run with USE_BASE_COLLECTION=False for
runtime-only development/testing.
```

Do not silently create an empty `mirage_base`.

## `USE_BASE_COLLECTION=False`

Do not check whether `mirage_base` exists.

Expected behavior:

```text
connect to Qdrant
→ skip base verification
→ create/resume runtime collection
→ initialize the same retriever with base_store=None
→ run runtime-only retrieval
```

This mode is intentionally supported while the curated base is still being built.

It must remain a conditional branch of the normal architecture, not a separate code path with duplicate retrieval or ingestion implementations.

---

# 42. Base Immutability Safety Guard

Add defensive code against accidental mutation.

At minimum, any centralized:

```text
delete_collection
reset_collection
upsert target selection
```

should reject:

```text
mirage_base
```

when invoked from inference lifecycle code.

For example:

```python
def assert_mutable_collection(name):
    if name == BASE_COLLECTION:
        raise RuntimeError(
            "Inference is not allowed to mutate mirage_base."
        )
```

Use this before destructive collection operations.

---

# 43. Web Addition Refactor

Current runtime web ingestion writes through the Qdrant store passed into `WebAddition`.

Change its construction so it receives:

```text
runtime_store
```

as the insertion destination.

For duplicate detection it needs read access to:

```text
base_store
runtime_store
```

Prefer a narrow dependency:

```text
CrossCollectionDeduplicator
```

rather than giving the web ingester general mutation access to base.

The deduplicator must accept an optional base store so the same object works when `USE_BASE_COLLECTION=False`.

Conceptually:

```text
WebAddition
│
├── runtime_store.upsert()
└── deduplicator.exists_in_either()
      ├── base_store.content_hash_exists()
      └── runtime_store.content_hash_exists()
```

---

# 44. PDF Addition Refactor

Apply the same pattern to runtime PDF ingestion.

```text
PDFAddition
│
├── runtime_store.upsert()
└── cross-collection duplicate checks
```

Do not create a separate architectural path for PDFs unless current code requires it.

Both runtime web and PDF ingestion share the same isolation rule:

```text
READ base
READ/WRITE runtime
```

---

# 45. Cross-Collection Deduplicator

A small helper may reduce duplicated logic.

Suggested:

```python
class CrossCollectionDeduplicator:
    def __init__(self, base_store, runtime_store):
        self.base_store = base_store
        self.runtime_store = runtime_store

    def find_duplicate(self, content_hash):
        if (
            self.base_store is not None
            and self.base_store.content_hash_exists(content_hash)
        ):
            return "base"

        if self.runtime_store.content_hash_exists(content_hash):
            return "runtime"

        return None
```

Return the duplicate source so reporting can distinguish both cases.

This helper must not upsert anything.

---

# 46. Concurrency and Runtime Upserts

Multiple RAG workers may search and ingest into the same runtime collection.

This is supported by Qdrant server mode.

Still preserve deterministic IDs / existing duplicate logic.

Potential race:

```text
worker A checks hash → missing
worker B checks hash → missing
both insert
```

If deterministic point IDs already collapse the same chunk safely, retain that protection.

Do not introduce heavy cross-process locking unless testing proves necessary.

The current migration to Qdrant was specifically intended to support concurrent workers through the central server.

---

# 47. Startup State Machine

Target:

```text
START
  ↓
parse configuration
  ↓
connect to Qdrant
  ↓
USE_BASE_COLLECTION?
  ├── True  → verify mirage_base exists
  └── False → skip base verification
  ↓
validate runtime mode
  ↓

RUNTIME_MODE == resume?
  │
  ├── override provided?
  │       │
  │       ├── yes → verify + use override
  │       │
  │       └── no
  │             ↓
  │       discover current-ablation runtimes
  │             ↓
  │       any?
  │       ├── yes → use newest
  │       └── no  → create timestamped runtime
  │
  └── no, mode == fresh
          ↓
     discover current-ablation runtimes
          ↓
     delete matching old runtimes
          ↓
     reset inference/eval progress
          ↓
     create timestamped runtime

  ↓
create/verify runtime payload indexes
  ↓
start workers using same two collections
  ↓
RUN
```

---

# 48. Success State Machine

```text
all inference items completed successfully
        ↓
runtime point count
        ↓
SNAPSHOT_RUNTIME?
        │
   ┌────┴─────┐
   │          │
  yes        no
   │          │
snapshot      │
   │          │
snapshot OK?  │
   │          │
 yes          │
   └────┬─────┘
        ↓
delete runtime collection
        ↓
record successful cleanup
        ↓
END
```

If snapshot fails:

```text
preserve runtime
report finalization failure
do not delete
```

---

# 49. Failure State Machine

```text
exception / interrupt / failed job
        ↓
do not delete active runtime
        ↓
do not run success cleanup
        ↓
best-effort log:
    runtime collection name
    ablation ID
    resume instructions
        ↓
END / propagate failure
```

Next run with:

```python
RUNTIME_MODE = "resume"
```

finds and uses the same runtime collection.

---

# 50. Fresh-State Cleanup Scope

Fresh mode must only delete runtime state associated with the **current ablation**.

Example Qdrant:

```text
mirage_base

mirage_runtime_ablation_A_20260824_100000
mirage_runtime_ablation_B_20260824_110000
```

Run:

```text
ABLATION_ID = ablation_A
RUNTIME_MODE = fresh
```

Result:

```text
mirage_base                                  KEEP
mirage_runtime_ablation_B_20260824_110000    KEEP

mirage_runtime_ablation_A_20260824_100000    DELETE
mirage_runtime_ablation_A_<new timestamp>    CREATE
```

Never use a broad `mirage_runtime_*` delete unless the user explicitly requests global cleanup outside normal inference.

---

# 51. Existing Query/Eval Resume Integration

Before implementation, inspect current `Inference/generate.py` logic for:

```text
already-successful output item detection
JSONL continuation
answer-model keys
failed item retry behavior
```

Keep this behavior.

The database update should only provide a clear signal:

```text
is_fresh_run = runtime_mode == "fresh"
```

The existing progress subsystem should interpret that appropriately.

Do not rewrite evaluation checkpointing unless required.

---

# 52. MainAgent Restart Every N Requests

The current inference architecture may re-instantiate `MainAgent` periodically.

Any such restart must preserve the active runtime collection.

Therefore worker/agent construction must receive:

```text
base_collection
runtime_collection
```

from stable worker configuration.

It must **not** rediscover or create a runtime collection on every `MainAgent` reconstruction.

Lifecycle selection belongs above `MainAgent`.

---

# 53. Runtime Collection Name Propagation

Once selected, the active collection name should be treated as immutable for the duration of the run.

Driver resolves:

```python
active_runtime_collection
```

Then passes it to all workers.

Potential path:

```text
Inference/generate.py
    ↓
rag_worker(...)
    ↓
MainAgent(
    base_collection=...,
    runtime_collection=...
)
```

Do not rely on each spawned worker to discover the latest collection independently.

That could race during startup.

---

# 54. Environment / CLI Propagation

Because workers may use Python multiprocessing `spawn`, ensure collection names are explicitly passed as arguments/config rather than relying on mutation of parent-process globals after worker start.

Values that should be propagated explicitly:

```text
BASE_COLLECTION
active runtime collection name
QDRANT_URL
QDRANT_API_KEY
embedding model
relevant ablation toggles
```

---

# 55. Logging Requirements

Startup should log clearly:

```text
Qdrant URL: ...
Base collection enabled: True|False
Base collection: mirage_base | DISABLED
Ablation ID: ...
Runtime mode: resume|fresh
Runtime collection: ...
Runtime action: CREATED|RESUMED|RECREATED
Snapshot runtime: True|False
```

Fresh mode should log deleted collection names.

Resume mode should log detected candidates and selected collection if multiple exist.

Failure should log:

```text
Runtime collection preserved: <name>
```

Success should log:

```text
Runtime collection deleted after successful completion.
```

---

# 56. Testing Strategy

The following tests will be run manually after implementation.

## Test 1 — Base existence

Setup:

```text
Qdrant running
mirage_base absent
```

Expected:

```text
inference refuses to start
runtime not created
```

---

## Test 2 — New resume-mode run

Setup:

```text
mirage_base exists
no runtime for current ablation
RUNTIME_MODE=resume
```

Expected:

```text
new timestamped runtime created
base unchanged
```

---

## Test 3 — Resume after simulated crash

Setup:

```text
runtime contains some ingested points
inference checkpoint is partially complete
```

Terminate inference before success cleanup.

Restart:

```python
RUNTIME_MODE = "resume"
```

Expected:

```text
same runtime collection selected
same points remain
existing query/eval resume logic continues
no new runtime created
```

---

## Test 4 — Fresh after simulated crash

Setup:

```text
old interrupted runtime exists
partial query/eval progress exists
```

Restart:

```python
RUNTIME_MODE = "fresh"
```

Expected:

```text
old current-ablation runtime deleted
new timestamped empty runtime created
progress starts from query 0
base untouched
```

---

## Test 5 — Cross-run isolation

Run A:

```text
ingest unique runtime content
```

Finish or interrupt.

Start genuinely fresh Run B.

Expected:

```text
Run B does not retrieve Run A runtime content
```

---

## Test 6 — Intra-run reuse

During one run:

```text
query 0 causes web ingestion of unique content
query 1 searches for that content
```

Expected:

```text
query 1 can retrieve runtime content added by query 0
```

This confirms runtime is run-scoped rather than query-scoped.

---

## Test 7 — Base immutability

Record:

```text
base point count
base representative payload
```

Run inference that triggers web ingestion.

Expected:

```text
base point count unchanged
base payload unchanged
runtime point count increases
```

---

## Test 8 — Duplicate against base

Insert known content into base.

Attempt runtime ingest of same content.

Expected:

```text
runtime insertion skipped
duplicate source reported as base
```

---

## Test 9 — Duplicate against runtime

Ingest same runtime source twice.

Expected:

```text
second insertion skipped/idempotent
runtime does not duplicate chunks
```

---

## Test 10 — Dual retrieval merge

Prepare:

```text
relevant result in base
more relevant result in runtime
```

Expected:

```text
both searched
global ranking reflects similarity
retrieval_source attached correctly
```

---

## Test 11 — Progressive retrieval

Verify each enabled strategy searches both collections.

Expected:

```text
strategy winner uses merged result set
existing strategy scoring semantics unchanged
```

---

## Test 12 — Confidence

Prepare mixed base/runtime top-k.

Expected:

```text
confidence sees merged evidence
not separate collection confidences
```

---

## Test 13 — Runtime snapshot off

```python
SNAPSHOT_RUNTIME = False
```

Successful run.

Expected:

```text
no snapshot created
runtime deleted
base kept
```

---

## Test 14 — Runtime snapshot on

```python
SNAPSHOT_RUNTIME = True
```

Successful run.

Expected:

```text
snapshot succeeds
snapshot information logged/reported
runtime deleted only after snapshot success
```

---

## Test 15 — Snapshot failure

Force snapshot failure.

Expected:

```text
runtime NOT deleted
error reported
collection available for inspection/resume
```

---

## Test 16 — Override

Set:

```python
RUNTIME_MODE = "resume"
RUNTIME_COLLECTION_OVERRIDE = "<existing runtime>"
```

Expected:

```text
exact collection used
```

Set:

```python
RUNTIME_MODE = "fresh"
RUNTIME_COLLECTION_OVERRIDE = "..."
```

Expected:

```text
configuration error
nothing deleted
```

---

## Test 17 — Multiple interrupted runtimes

Create:

```text
mirage_runtime_X_20260824_100000
mirage_runtime_X_20260824_120000
```

Resume.

Expected:

```text
12:00 collection selected
both candidates logged
```

Fresh.

Expected:

```text
both X runtime collections deleted
new X runtime created
```

---

## Test 18 — Different ablations

Create:

```text
mirage_runtime_A_...
mirage_runtime_B_...
```

Run A fresh.

Expected:

```text
A old runtime deleted
B untouched
mirage_base untouched
```

---

## Test 19 — Runtime-only mode with no base collection

Setup:

```python
USE_BASE_COLLECTION = False
```

and ensure:

```text
mirage_base does not exist
```

Expected:

```text
startup succeeds
no base collection is created
runtime collection is created/resumed normally
retrieval searches runtime only
runtime ingestion deduplicates against runtime only
confidence receives runtime results normally
```

---

## Test 20 — Base-enabled mode with missing base

Setup:

```python
USE_BASE_COLLECTION = True
```

and ensure:

```text
mirage_base does not exist
```

Expected:

```text
startup fails clearly
runtime workers do not start
empty mirage_base is NOT created
```

---

## Test 21 — Toggle uses same architecture

Run equivalent runtime-only data through:

```python
USE_BASE_COLLECTION=False
```

Expected implementation behavior:

```text
same InferenceDatabaseManager
same DualCollectionRetriever
same runtime store
same WebAddition/PDFAddition
same confidence evaluator
same runtime lifecycle
```

Only the base calls are skipped.

No separate `RuntimeOnlyRetriever`, `RuntimeOnlyAgent`, or duplicate ingestion pipeline should exist.

---

# 59. Implementation Sequence

Recommended engineering order:

## Phase 1 — Collection lifecycle

1. Add configuration values.
2. Implement runtime name sanitizer/generator/parser.
3. Implement `InferenceDatabaseManager`.
4. Add `USE_BASE_COLLECTION` and conditionally verify base existence.
5. Implement resume discovery.
6. Implement fresh deletion.
7. Implement runtime creation + indexes.
8. Add immutable-base guards.
9. Wire startup in `Inference/generate.py`.
10. Ensure all workers receive the same active runtime name.

Do not change retrieval yet.

---

## Phase 2 — Store separation

1. Allow `QdrantStore` to bind cleanly to an existing named collection.
2. Prevent base auto-create.
3. Instantiate runtime store always and base store only when enabled.
4. Ensure runtime store is the only write target.
5. Remove/disable old one-collection reset path.

---

## Phase 3 — Dual retrieval

1. Extract/reuse current priority-filter logic.
2. Implement `DualCollectionRetriever`.
3. Search runtime for every strategy and base additionally when enabled.
4. Attach result provenance.
5. Merge.
6. Dedupe by `content_hash`.
7. Prefer base on cross-collection duplicates.
8. Rank globally.
9. Preserve existing strategy scoring.
10. Route confidence through merged retrieval.

---

## Phase 4 — Runtime ingestion isolation

1. Refactor `WebAddition`.
2. Refactor `PDFAddition`.
3. Implement conditional cross-collection duplicate checks:
   - base + runtime when enabled
   - runtime only when base disabled
4. Upsert only to runtime.
5. Keep canonical metadata/chunking unchanged.
6. Add duplicate-source statistics/logging where convenient.

---

## Phase 5 — Resume/fresh alignment

1. Reuse current inference checkpoint logic.
2. On resume, continue existing progress.
3. On fresh, reset/restart progress.
4. Test crash recovery.
5. Test fresh after crash.

---

## Phase 6 — Finalization

1. Add `SNAPSHOT_RUNTIME`.
2. Snapshot only on successful run.
3. Delete runtime after successful snapshot or directly when snapshot disabled.
4. Preserve runtime when snapshot fails.
5. Preserve runtime on inference failure.
6. Print runtime name on failure.

---

## Phase 7 — Reporting and cleanup

1. Add startup database logs.
2. Add base/runtime retrieval-source counters if practical.
3. Add runtime ingestion/duplicate counts if practical.
4. Update docs/comments naming the old single collection.
5. Remove stale assumptions that rank0 resets the RAG/base collection.
6. Add tests.

---

# 60. File-by-File Implementation Map

## `Inference/generate.py`

Expected changes:

```text
add USE_BASE_COLLECTION + runtime-mode CLI/config
create one Qdrant lifecycle manager before worker processing
verify mirage_base only when enabled
resolve active runtime collection
integrate fresh with existing progress reset
pass base/runtime names to workers
preserve runtime on failure
finalize runtime on full success
snapshot toggle
replace old reset assumptions
```

This file should remain the inference-run lifecycle owner.

---

## `rag_agent/main.py`

Expected changes:

```text
accept base_collection
accept use_base_collection
accept runtime_collection

construct:
    optional base store
    runtime store
    same dual retriever

pass runtime-only store to:
    web ingestion
    PDF ingestion

pass merged retriever to:
    retrieve tool
    confidence evaluator

remove assumption that one collection is both preload + runtime
remove base-reset behavior
```

---

## `rag_agent/utils/qdrant_store.py`

Expected changes:

```text
support explicit named collections cleanly
separate require-existing behavior from create behavior
ensure search can operate on base/runtime independently
retain deterministic IDs
retain content_hash_exists
retain payload indexes for runtime creation
add safety guard where destructive operations are centralized
```

Avoid large rewrites if current store already supports most of this.

---

## `rag_agent/utils/ContentUtils.py`

Expected changes:

```text
reuse filter-building logic
reuse query embedding
reuse strategy score calculations

move multi-collection search/merge orchestration into
DualCollectionRetriever or refactor helpers so it can call both stores
```

Do not change:

```text
priority ladder meaning
similarity formula
metadata semantics
chunking contract
```

---

## `rag_agent/tools/web_addition.py`

Expected changes:

```text
runtime-only write target
cross-collection duplicate checking
no base mutation
```

---

## `rag_agent/tools/pdf_addition.py`

Expected changes:

```text
runtime-only write target
cross-collection duplicate checking
no base mutation
```

---

## `rag_agent/tools/confidence_evaluator.py`

Likely minimal change.

Requirement:

```text
its retrieval input must now be the merged dual-collection result path
```

Do not change confidence weights/thresholds unless needed for compatibility.

---

## `rag_agent/ablation_configs.json`

Do not encode collection names in every ablation unless necessary.

`ABLATION_ID` already selects the experiment and should be used to construct the runtime namespace.

Database lifecycle settings are run configuration, not scientific ablation semantics.

---

# 61. Non-Goals

Do not include the following in this implementation unless absolutely required:

```text
moving Qdrant storage into MetaMIRAGE repo
base snapshot restore
offline base promotion
runtime-to-base synchronization
query-scoped runtime collections
one runtime collection per query
automatic curation
dynamic vector-schema discovery
new checkpoint database
active_run.json
new database registry
separate runtime-only retriever/agent architecture
base/runtime score weighting
changing embedding model
changing RAG chunking
changing confidence formula
rewriting evaluation architecture
rewriting preload
```

The preload side is already responsible for curated base construction.

This task is specifically the **inference database separation and lifecycle update**.

---

# 62. Safety Invariants

These should be treated as assertions during implementation.

1. `mirage_base` is never created by inference.
2. `mirage_base` is never deleted by inference.
3. `mirage_base` is never reset by inference.
4. `mirage_base` is never upserted into by inference.
5. When `USE_BASE_COLLECTION=True`, `mirage_base` must exist before workers start.
6. When `USE_BASE_COLLECTION=False`, base verification/retrieval/deduplication are skipped and no empty base is created.
7. `USE_BASE_COLLECTION=False` uses the same lifecycle/retriever/ingestion architecture with an optional base branch; no parallel runtime-only architecture is introduced.
8. Web/PDF runtime content is written only to the active runtime collection.
9. Every RAG worker uses the same runtime collection for one run.
10. Different inference runs never share runtime collections.
11. Success deletes runtime after optional snapshot.
12. Failure preserves runtime.
13. Resume reuses runtime.
14. Fresh deletes old current-ablation runtime state and creates an empty collection.
15. Fresh starts inference progress from query 0.
16. Resume uses existing inference progress.
17. Runtime content from earlier queries in the same run remains searchable by later queries.
18. With base enabled, runtime ingestion dedupe checks base before runtime.
19. With base disabled, runtime ingestion dedupe checks runtime only.
20. Base and runtime retrieval results compete under the same similarity semantics when base is enabled.
21. Confidence sees the available merged evidence (base + runtime, or runtime-only when base disabled).
22. Runtime data never gets automatically promoted into base.
23. Embedding model remains `BAAI/bge-base-en-v1.5`.
24. Vector dimension remains `768`.
25. Qdrant distance remains `COSINE`.
26. Runtime payload indexes include `hardiness_zone`, `month_year`, `title`, and `content_hash`.
27. Runtime `normalize_embeddings=True` and preload `normalize_embeddings=False` are intentionally allowed under cosine distance.

---

# 63. Acceptance Criteria

The implementation is complete when all of the following are true.

### Startup

```text
Qdrant unavailable
→ clear failure
```

With:

```python
USE_BASE_COLLECTION = True
```

```text
mirage_base missing
→ clear failure
→ no empty base auto-created
```

With:

```python
USE_BASE_COLLECTION = False
```

```text
mirage_base may be absent
→ startup continues
→ no empty base auto-created
→ runtime collection lifecycle works normally
```

### New run

```text
no resumable runtime
RUNTIME_MODE=resume
→ timestamped runtime created
```

### Resume

```text
interrupted runtime exists
RUNTIME_MODE=resume
→ same runtime reused
→ no new timestamp
→ previous runtime points visible
→ existing inference progress resumes
```

### Fresh

```text
interrupted runtime exists
RUNTIME_MODE=fresh
→ old current-ablation runtime deleted
→ fresh runtime created
→ query/eval progress starts at 0
```

### Retrieval

With base enabled:

```text
every priority strategy searches base + runtime
→ results merged
→ content-hash deduped
→ globally ranked
→ top-k returned
→ source provenance available
```

With base disabled:

```text
same retriever searches runtime only
→ no base request is made
→ runtime results are ranked normally
→ top-k returned
→ confidence path remains unchanged
```

### Ingestion

With base enabled:

```text
runtime web/PDF addition
→ checks base duplicate
→ checks runtime duplicate
→ writes only runtime
```

With base disabled:

```text
runtime web/PDF addition
→ skips base duplicate lookup
→ checks runtime duplicate
→ writes only runtime
```

### Isolation

```text
base point count unchanged through inference
```

and:

```text
new independent run starts without previous runtime knowledge
```

### Same-run accumulation

```text
later queries can retrieve runtime content found by earlier queries
```

### Failure

```text
inference interrupted
→ runtime preserved
→ runtime name printed
```

### Success without snapshot

```text
SNAPSHOT_RUNTIME=False
→ runtime deleted
```

### Success with snapshot

```text
SNAPSHOT_RUNTIME=True
→ snapshot created
→ snapshot recorded/logged
→ runtime deleted
```

### Snapshot failure

```text
snapshot fails
→ runtime preserved
```

---

# 64. Final Target Mental Model

```text
                   CURATED OFFLINE DATABASE
                           │
                           ▼
                    ┌─────────────┐
                    │ mirage_base │
                    │ READ ONLY   │
                    │ optional at │
                    │ dev/runtime │
                    │ test time   │
                    └──────┬──────┘
                           │
                      if enabled
                           │
                    ┌──────┴───────┐
                    │ DualRetriever│
                    └──────┬───────┘
                           │
                           │
                    ┌──────┴──────────────────────────┐
                    │                                 │
                    │                    ┌──────────────────────────┐
                    │                    │ mirage_runtime_<RUN_ID>  │
                    │                    │ READ + WRITE             │
                    │                    │ shared within one run    │
                    │                    └────────────┬─────────────┘
                    │                                 │
                    └────────────────┬────────────────┘
                                     ↓
                               merged evidence
                                     ↓
                                confidence
                                     ↓
                          low → web/PDF augmentation
                                     ↓
                              runtime only
```

Base toggle:

```text
USE_BASE_COLLECTION=True
→ normal/final architecture
→ base + runtime

USE_BASE_COLLECTION=False
→ same architecture with base branch disabled
→ runtime only
→ intended for development/testing while base is unavailable
```

Lifecycle:

```text
NEW
→ base + empty runtime when base enabled
→ empty runtime only when base disabled

CRASH
→ preserve runtime

RESUME
→ reuse same runtime + existing inference progress

FRESH
→ delete old current-ablation runtime
→ reset inference progress
→ create new empty runtime
→ start query 0

SUCCESS
→ optional runtime snapshot
→ delete runtime

BASE
→ curated offline
→ immutable forever during inference
```

---

# 65. Implementation Principle

The desired change is **not** a new RAG algorithm.

The retrieval strategy, confidence logic, embedding model, metadata semantics, web-search behavior, chunking, and existing inference checkpoint system should remain as stable as possible.

The implementation should primarily introduce:

```text
collection isolation
collection lifecycle management
one flexible retriever with optional base participation
conditional cross-collection deduplication
runtime-only mutation
crash-safe resume/fresh semantics
optional runtime snapshotting
```

`USE_BASE_COLLECTION=False` must be implemented as a small conditional around base participation, not as a new architecture.

Prefer small, testable abstractions over a broad rewrite of the RAG stack.
