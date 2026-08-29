# MetaMIRAGE — Concurrent Preload Pipeline Architecture

**Status:** Finalized design
**Purpose:** Replace the current sequential, one-state-at-a-time preload execution model with a concurrency-safe multi-state architecture that can use multiple independent 1-GPU Jupyter notebook allocations while preserving deterministic persistence, global deduplication, resumability, cumulative Qdrant state, crop-occurrence state, and reproducible snapshots.

---

## 1. Motivation

The current MetaMIRAGE preload notebook is designed around a sequential cumulative workflow:

```text
State 1
  ↓
extract → qualify → chunk → embed → Qdrant
  ↓
snapshot
  ↓
State 2
  ↓
restore/verify previous cumulative state
  ↓
extract → qualify → chunk → embed → Qdrant
  ↓
snapshot
  ↓
State 3
  ↓
...
```

This architecture works correctly when one state is processed at a time, but it cannot safely be extended by simply opening several copies of the notebook and pointing all of them at the same Qdrant storage.

The new architecture is designed around the actual compute environment available for preload:

- Long-lived Jupyter notebook allocations are easier to obtain with **one GPU per notebook**.
- Several independent notebook allocations may be available at the same time.
- Each notebook may run on a **different compute node** and have its own Python environment, GPU, kernel, and local variables.
- The preload should therefore use **one state per notebook**, with multiple states processed concurrently.
- All workers should contribute to a **single cumulative Qdrant build**.
- The number of concurrent states is variable. Five generic notebooks are intended as the normal maximum, but a wave may contain 1, 2, 3, 4, or 5 states depending on available compute.

The main goal is therefore:

> Allow several independent state notebooks to process concurrently while preserving all important correctness guarantees of the existing cumulative preload architecture.

---

## 2. Design Principles

### 2.1 One worker notebook processes one state

Each worker notebook:

- runs independently;
- has its own GPU;
- has its own model instances;
- has its own variables and Python environment;
- processes exactly one state at a time;
- writes its state-local persistence artifacts separately from every other state;
- communicates with shared services over HTTP.

### 2.2 Worker notebooks are generic

There will ideally be five generic preload worker folders/notebooks available.

The notebook code in all five workers is the same.

Only the configuration block and input folder contents change.

For example:

```python
STATE_NAME = "Illinois"
STATE_CODE = "IL"

BUILD_ID = "mirage_base_2026"
WAVE_ID = "wave_01"

RUN_MODE = "resume"

COORDINATOR_URL = "http://<service-node>:8001"
QDRANT_URL = "http://<service-node>:6333"
```

A worker folder may process Illinois in one wave and Wisconsin in a later wave.

The worker folder itself is only execution space. Persistent state is stored by `BUILD_ID` and `STATE_CODE`, not by worker number.

### 2.3 Qdrant is a shared network service

Only one Qdrant server owns the Qdrant database files.

Worker notebooks do **not** start their own Qdrant servers against the same storage path.

Instead:

```text
Worker 1 ─┐
Worker 2 ─┤
Worker 3 ─┼── HTTP ──→ ONE Qdrant server
Worker 4 ─┤
Worker 5 ─┘
```

The Qdrant server is the single owner of the on-disk Qdrant storage and safely handles concurrent client writes.

### 2.4 Coordination is separate from the data path

The system uses a small **Preload Coordinator API** for:

- global deduplication claims;
- state execution locks;
- state status;
- claim leases;
- wave status.

The coordinator does **not**:

- receive PDFs or CSV files;
- run qualification;
- chunk documents;
- compute embeddings;
- forward vectors to Qdrant.

Workers process data locally and write vectors directly to Qdrant.

This separates:

```text
CONTROL PLANE
Coordinator API
```

from:

```text
DATA PLANE
Qdrant
```

### 2.5 State completion and build checkpointing are separate

A worker finishing a state does **not** create a Qdrant snapshot.

A Qdrant snapshot is created only after all states explicitly assigned to a wave have completed successfully.

The checkpoint unit therefore changes from:

```text
one state
```

to:

```text
one wave
```

---

# 3. High-Level Topology

The intended deployment is:

```text
                     SERVICE NODE
        ┌─────────────────────────────────┐
        │                                 │
        │  Qdrant Server                  │
        │  :6333                          │
        │                                 │
        │  Preload Coordinator API        │
        │  :8001                          │
        │                                 │
        │  coordinator.db                 │
        │                                 │
        └────────────────┬────────────────┘
                         │
                         │ HTTP
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ↓                 ↓                  ↓
   Worker Node 1     Worker Node 2      Worker Node 3
   1 GPU             1 GPU              1 GPU
   preload.ipynb     preload.ipynb      preload.ipynb
   STATE=IL          STATE=IA           STATE=IN

       ↓                 ↓                  ↓

   Worker Node 4     Worker Node 5
   1 GPU             1 GPU
   preload.ipynb     preload.ipynb
   STATE=MI          STATE=OH
```

The service node does not need a GPU.

A wave does not need to use all five workers.

For example:

```text
wave_01:
  IL
  IA
  IN
```

may use only three workers.

A later wave may use five:

```text
wave_02:
  MI
  OH
  WI
  MN
  MO
```

No part of the architecture assumes a fixed wave size.

---

# 4. Shared Service Node

The shared service node runs two long-lived processes.

## 4.1 Qdrant Server

Example:

```text
http://<service-node>:6333
```

Responsibilities:

- own the Qdrant storage directory;
- host the cumulative preload collection;
- accept concurrent vector upserts from workers;
- support validation queries;
- create cumulative snapshots during finalization.

Typical persistent storage remains conceptually:

```text
/work/nvme/.../qdrant_database
```

The exact cluster path may be adjusted operationally.

### Important invariant

There must never be multiple Qdrant server processes simultaneously opening the same Qdrant storage directory.

Correct:

```text
ONE Qdrant process
       ↓
shared persistent Qdrant storage
```

Incorrect:

```text
Qdrant server A ─┐
Qdrant server B ─┼→ same storage directory
Qdrant server C ─┘
```

---

## 4.2 Preload Coordinator API

Example:

```text
http://<service-node>:8001
```

The coordinator should be a small internal service, such as a FastAPI application served by Uvicorn.

Conceptually:

```bash
uvicorn preload_coordinator:app \
  --host 0.0.0.0 \
  --port 8001
```

The coordinator uses a small persistent registry such as:

```text
/work/nvme/.../preload_coordinator.db
```

Only the coordinator process accesses this SQLite database directly.

Workers communicate with it through HTTP.

This avoids having several Python processes on several compute nodes directly opening the same SQLite file.

---

# 5. Why a Coordinator Is Needed

Qdrant can safely handle concurrent vector writes, but Qdrant alone does not solve every preload concurrency problem.

The preload architecture also requires:

1. strict global document/content deduplication;
2. prevention of two notebooks processing the same state simultaneously;
3. recovery when a worker crashes after claiming work;
4. centralized visibility into state/wave progress.

The coordinator handles exactly these control-plane concerns.

It is deliberately kept small so that it does not become a data-processing bottleneck.

---

# 6. Coordinator Responsibilities

## 6.1 State locking

Only one worker may process a given state for a given build at a time.

Valid:

```text
IL + IA + IN + MI + OH
```

Invalid:

```text
IL worker A
IL worker B
```

The worker starts by acquiring a state lock.

Conceptually:

```http
POST /state/acquire
```

Payload:

```json
{
  "build_id": "mirage_base_2026",
  "wave_id": "wave_01",
  "state_code": "IL",
  "worker_id": "worker_1"
}
```

A successful response grants the worker a lease.

If another active worker already owns the IL lock, the coordinator rejects the request.

## 6.2 State lock leases

State locks must not remain active forever if a notebook dies.

A state lock therefore has a lease, for example:

```text
build_id
state_code
worker_id
status
acquired_at
lease_until
last_heartbeat
```

The worker periodically refreshes its lease.

If the notebook crashes and stops heartbeating, the lease eventually expires and the state can be resumed by another notebook.

## 6.3 Global deduplication claims

The existing preload architecture treats canonical content hashing as a global deduplication mechanism.

Concurrent notebooks create a race if they independently do:

```text
check hash
↓
not found
↓
process
```

at the same time.

Therefore a worker must atomically claim a canonical content hash through the coordinator.

Example:

```http
POST /content/claim
```

Payload:

```json
{
  "build_id": "mirage_base_2026",
  "wave_id": "wave_01",
  "state_code": "IL",
  "content_hash": "abc123..."
}
```

Possible responses:

```text
CLAIMED
ALREADY_COMPLETE
CLAIMED_BY_OTHER_WORKER
```

Because the coordinator performs the claim atomically, two workers cannot both become the canonical owner of the same content hash.

## 6.4 Content claim leases

A content claim must also use a lease.

Example record:

```text
build_id      = mirage_base_2026
content_hash  = abc123
owner_state   = IL
worker_id     = worker_1
status        = CLAIMED
claimed_at    = ...
lease_until   = ...
```

If the worker completes processing:

```http
POST /content/complete
```

the coordinator records:

```text
status = COMPLETE
```

If the worker crashes before completion and the lease expires, another worker may claim the content later.

This prevents permanent dead claims.

## 6.5 State completion

When a worker has successfully completed all processing and local validation for its state, it reports:

```http
POST /state/complete
```

The coordinator records:

```text
IL = COMPLETE
```

This status is used by the finalizer.

A worker should only report `COMPLETE` after:

- expected inputs have been processed;
- terminal-state validation passes;
- state-local crop output is written;
- state manifest is written;
- all required Qdrant upserts are complete.

## 6.6 Wave status

The coordinator may expose:

```http
GET /wave/status?build_id=...&wave_id=...
```

for observability.

However, the finalizer should still use explicit `EXPECTED_STATES` and state manifests as correctness inputs rather than assuming that every active worker automatically belongs to the wave.

---

# 7. Worker Notebook Architecture

All worker notebooks should be identical copies.

The worker notebook is responsible for the actual preload pipeline.

Its conceptual lifecycle becomes:

```text
load configuration
      ↓
validate state inputs
      ↓
acquire state lock from coordinator
      ↓
load state-local persisted state
      ↓
discover/extract sources
      ↓
canonical normalization
      ↓
compute content hash
      ↓
claim content hash from coordinator
      ↓
      ├── duplicate/already complete → skip
      │
      └── claimed
             ↓
          qualify
             ↓
          accept/reject
             ↓
          chunk
             ↓
          validate metadata
             ↓
          embed locally on worker GPU
             ↓
          direct Qdrant upsert
             ↓
          mark content claim complete
             ↓
          continue
      ↓
write state crop JSON
      ↓
validate terminal states
      ↓
write state manifest
      ↓
mark state COMPLETE
```

---

# 8. Worker Configuration Block

Every worker uses the same notebook but edits its configuration block manually.

Example:

```python
# ------------------------------------------------------------------
# STATE CONFIGURATION
# ------------------------------------------------------------------

STATE_NAME = "Illinois"
STATE_CODE = "IL"

BUILD_ID = "mirage_base_2026"
WAVE_ID = "wave_01"

RUN_MODE = "resume"

COORDINATOR_URL = "http://<service-node>:8001"
QDRANT_URL = "http://<service-node>:6333"

QDRANT_COLLECTION = "mirage_base_build"
```

The notebook should not derive concurrency behavior from:

```text
RUN_SEQUENCE
PREVIOUS_RUN_DIR
```

Those concepts belong to the old sequential architecture and should no longer be correctness dependencies.

---

# 9. Generic Worker Folders

The intended working layout is approximately:

```text
preload_pipeline/
│
├── workers/
│   ├── worker_1/
│   │   ├── preload.ipynb
│   │   └── input/
│   │       ├── <PDF ZIP if present>
│   │       ├── <CSV ZIP if present>
│   │       └── <URL file if present>
│   │
│   ├── worker_2/
│   │   ├── preload.ipynb
│   │   └── input/
│   │
│   ├── worker_3/
│   ├── worker_4/
│   └── worker_5/
│
├── persistent_state/
│   └── <BUILD_ID>/
│       ├── IL/
│       ├── IA/
│       ├── IN/
│       └── ...
│
├── shared/
│   └── crop_occurrences.json
│
└── finalizer/
    └── finalize_wave.ipynb
```

The exact directory names may change during implementation, but the ownership boundaries should not.

---

# 10. State-Local Persistence

Persistent processing state is organized by state, not by generic worker number.

Example:

```text
persistent_state/
└── mirage_base_2026/
    ├── IL/
    │   ├── pipeline_state.db
    │   ├── canonical/
    │   ├── crop_occurrences_state.json
    │   ├── state_manifest.json
    │   └── logs/
    │
    ├── IA/
    │   ├── pipeline_state.db
    │   ├── canonical/
    │   ├── crop_occurrences_state.json
    │   ├── state_manifest.json
    │   └── logs/
    │
    └── IN/
        └── ...
```

This is critical because generic workers are reusable.

Example:

```text
worker_1
  wave_01 → Illinois

worker_1
  wave_02 → Wisconsin
```

Illinois and Wisconsin must never share:

- `pipeline_state.db`;
- canonical files;
- state crop output;
- state manifests;
- retry state.

Their persistent directories are selected from the configuration:

```text
BUILD_ID + STATE_CODE
```

rather than:

```text
worker_1
```

---

# 11. State-Local SQLite Ledger

Each state maintains its own processing ledger.

The ledger continues to provide:

- persisted extraction/processing state;
- terminal-state tracking;
- retry state;
- resume support;
- prevention of unnecessary reprocessing.

Because each state now owns a separate SQLite database, workers on separate nodes do not directly contend over one shared state ledger.

The coordinator owns a different small global registry used only for cross-state coordination.

These databases have different purposes:

| Database | Ownership | Purpose |
|---|---|---|
| State `pipeline_state.db` | One state worker at a time | Detailed state-local pipeline progress |
| Coordinator registry | Coordinator process only | Global dedupe, locks, leases, status |

---

# 12. Global Deduplication

Global deduplication remains a required invariant.

The new architecture implements it as an atomic claim instead of a distributed:

```text
check → then act
```

pattern.

Example:

```text
Illinois discovers content X
Iowa discovers content X
```

Both compute:

```text
content_hash = abc123
```

Then:

```text
IL ──┐
     ├──→ Coordinator
IA ──┘
```

The coordinator atomically grants exactly one claim.

Example result:

```text
IL → CLAIMED
IA → ALREADY_CLAIMED
```

IL becomes the canonical processor for that content.

IA records/skips the duplicate according to the existing preload semantics.

---

# 13. Qdrant Point Identity

Global content deduplication and Qdrant point identity are related but not identical concerns.

The Qdrant point ID should preserve provenance-aware chunk identity.

The agreed logical identifier is:

```text
source_id
+ page
+ chunk_index
+ content_hash
```

Conceptually:

```python
logical_id = (
    f"{source_id}|"
    f"{page}|"
    f"{chunk_index}|"
    f"{content_hash}"
)
```

The logical ID is converted to a deterministic UUID using UUID5:

```python
point_id = uuid.uuid5(NAMESPACE, logical_id)
```

This gives two important properties:

### Determinism

The same logical chunk always produces the same Qdrant point ID.

### Idempotent retries

If a worker crashes after Qdrant accepts an upsert but before the local ledger records completion, retrying the same chunk generates the same point ID.

Therefore:

```text
upsert ABC
upsert ABC again
```

does not create:

```text
ABC
XYZ
```

for the same logical chunk.

This is essential to reliable resume behavior.

---

# 14. Qdrant Payload Requirements

The existing canonical metadata contract should remain compatible with runtime retrieval.

Typical fields include:

```text
source_type
source_id
title
url
page
chunk_index
location
month_year
content_hash
language
hardiness_zone
```

The new preload architecture should additionally consider recording operational provenance such as:

```text
build_id
ingest_state
wave_id
```

These fields are useful for:

- debugging;
- reproducibility;
- validating wave output;
- auditing which build introduced a point.

They should not change retrieval semantics unless explicitly intended.

---

# 15. Direct Qdrant Ingestion

The coordinator is never in the vector data path.

Correct:

```text
Worker
   ├──→ Coordinator: claim/status calls
   │
   └──→ Qdrant: vectors + payload
```

Incorrect:

```text
Worker
   ↓
Coordinator
   ↓
Qdrant
```

The latter would unnecessarily:

- serialize large vectors through the coordinator;
- increase memory requirements;
- increase network overhead;
- create a bottleneck;
- add another failure point.

Worker notebooks compute embeddings locally and call Qdrant directly.

---

# 16. GPU Model Ownership

Each worker notebook owns its own model instances.

For example:

```text
Worker IL
  own qualification model
  own embedding model
  own GPU

Worker IA
  own qualification model
  own embedding model
  own GPU
```

Because workers are separate Jupyter allocations, each notebook may see its assigned GPU as:

```text
CUDA device 0
```

even though the physical GPUs belong to different nodes/jobs.

The service node requires no GPU.

---

# 17. Wave Model

A **wave** is an explicit group of states that are intended to complete before a cumulative checkpoint is finalized.

Wave size is variable.

Examples:

```text
wave_01
  IL
  IA
  IN
```

```text
wave_02
  MI
  OH
  WI
  MN
  MO
```

```text
wave_03
  KY
  TN
```

There is no hardcoded assumption that a wave contains five states.

Five is simply the desired maximum number of simultaneously available generic workers.

---

# 18. Explicit Wave Membership

The finalizer does not infer wave membership from active workers.

The finalization notebook is manually configured with the exact expected states.

Example:

```python
BUILD_ID = "mirage_base_2026"
WAVE_ID = "wave_01"

EXPECTED_STATES = [
    "IL",
    "IA",
    "IN",
]
```

For a five-state wave:

```python
EXPECTED_STATES = [
    "MI",
    "OH",
    "WI",
    "MN",
    "MO",
]
```

This makes variable compute availability explicit and reproducible.

---

# 19. State Completion Is Not a Snapshot

Worker notebooks never:

- create cumulative Qdrant snapshots;
- restore Qdrant snapshots;
- delete the shared Qdrant collection;
- reset the collection;
- compare Qdrant point counts against a previous-state snapshot and restore on mismatch.

Those behaviors are incompatible with concurrent writers.

A worker only marks its own state complete.

Example:

```text
IL  COMPLETE
IA  COMPLETE
IN  COMPLETE
```

No snapshot occurs until finalization.

---

# 20. Why Per-State Snapshots Are Removed

Suppose five states run concurrently:

```text
IL  100%
IA   70%
IN   40%
MI   85%
OH   20%
```

If Illinois created a snapshot when it completed, the snapshot could contain:

```text
IL  100%
IA   70%
IN   40%
MI   85%
OH   20%
```

Such a snapshot does not represent a valid cumulative state boundary.

Therefore per-state snapshot semantics are removed.

---

# 21. Cumulative Per-Wave Snapshots

Snapshots remain cumulative, but the checkpoint boundary is the wave.

Example:

```text
Wave 1:
IL + IA + IN
```

After all three complete:

```text
Snapshot 1 =
IL + IA + IN
```

Wave 2:

```text
MI + OH + WI + MN
```

After completion:

```text
Snapshot 2 =
IL + IA + IN
+ MI + OH + WI + MN
```

Thus snapshots remain cumulative exactly as before, but now at a concurrency-safe boundary.

Suggested naming:

```text
mirage_base_<BUILD_ID>_<WAVE_ID>.snapshot
```

Example:

```text
mirage_base_mirage_base_2026_wave_02.snapshot
```

The final naming convention may be cleaned up during implementation.

---

# 22. Finalization Notebook

Global finalization is moved into a completely separate notebook:

```text
finalizer/finalize_wave.ipynb
```

It is not one of the state worker notebooks.

This notebook performs all operations that affect global build state.

---

# 23. Finalization Preconditions

The finalizer first verifies all explicitly expected states.

Example:

```text
EXPECTED_STATES = IL, IA, IN

IL → COMPLETE
IA → COMPLETE
IN → FAILED
```

Result:

```text
DO NOT FINALIZE
DO NOT SNAPSHOT
```

After IN is resumed and completes:

```text
IL → COMPLETE
IA → COMPLETE
IN → COMPLETE
```

Finalization may proceed.

---

# 24. Finalization Workflow

The intended order is:

```text
load BUILD_ID / WAVE_ID / EXPECTED_STATES
      ↓
validate all expected state manifests
      ↓
validate coordinator state statuses
      ↓
validate state-local terminal-state invariants
      ↓
merge state crop JSON outputs
      ↓
atomically update global crop_occurrences.json
      ↓
validate Qdrant collection
      ↓
run point-count / schema / sample retrieval checks
      ↓
create cumulative Qdrant snapshot
      ↓
verify snapshot creation
      ↓
write wave manifest LAST
```

The wave manifest being written last acts as the final commit marker for a successfully finalized wave.

---

# 25. Finalizer Idempotency

`finalize_wave.ipynb` must be safe to rerun after interruption.

Example failure:

```text
crop merge completed
Qdrant snapshot completed
kernel dies before wave manifest
```

A rerun should:

- detect already completed safe steps;
- verify their outputs;
- avoid corrupting or duplicating global state;
- finish remaining work;
- write the final manifest.

This is especially important because finalization is the only stage that modifies cumulative global artifacts.

---

# 26. Crop Occurrence Architecture

The crop occurrence artifact has two levels.

## State-local crop output

Each worker creates:

```text
<STATE_DIR>/crop_occurrences_state.json
```

For example:

```text
persistent_state/.../IL/crop_occurrences_state.json
persistent_state/.../IA/crop_occurrences_state.json
persistent_state/.../IN/crop_occurrences_state.json
```

Workers never edit the shared global crop JSON.

## Global cumulative crop output

One persistent global file is maintained across all waves:

```text
shared/crop_occurrences.json
```

This is the canonical cumulative crop dictionary artifact.

---

# 27. Crop Merge Semantics

Suppose before wave 2 the global file contains:

```json
{
  "illinois": {},
  "iowa": {},
  "indiana": {}
}
```

Wave 2 produces:

```text
MI/crop_occurrences_state.json
OH/crop_occurrences_state.json
WI/crop_occurrences_state.json
```

The finalizer:

1. reads the existing global file;
2. reads crop outputs from only the expected wave states;
3. updates/adds those state keys;
4. validates the merged result;
5. writes a temporary file;
6. atomically replaces the canonical global file.

Result:

```json
{
  "illinois": {},
  "iowa": {},
  "indiana": {},
  "michigan": {},
  "ohio": {},
  "wisconsin": {}
}
```

The global file is not rebuilt from scratch each wave.

It is maintained cumulatively.

---

# 28. Crop JSON Safety

Before modifying the global crop file, the finalizer should keep a pre-wave backup.

Example:

```text
crop_occurrences.before_wave_03.json
```

Recommended update pattern:

```text
read canonical global file
      ↓
merge expected states
      ↓
write crop_occurrences.tmp
      ↓
validate JSON and required structure
      ↓
backup current global file
      ↓
atomic replace
```

This avoids partially written global state.

---

# 29. Worker Resume Mode

The default run mode is:

```python
RUN_MODE = "resume"
```

This mode is for:

- notebook kernel crash;
- GPU allocation ending;
- temporary Qdrant network failure;
- temporary coordinator failure;
- model/download error;
- OOM fixed without changing semantic processing;
- other transient execution failures.

Resume means:

```text
keep existing state-local ledger
keep existing canonical state
keep existing Qdrant points
recover interrupted state entries
skip completed items
retry incomplete items
continue processing
```

---

# 30. No Global Rollback on Worker Failure

Consider:

```text
IL → COMPLETE
IA → COMPLETE
IN → crashes at 47%
MI → COMPLETE
OH → COMPLETE
```

The shared Qdrant may contain:

```text
IL 100%
IA 100%
IN 47%
MI 100%
OH 100%
```

This is acceptable during an active wave.

The system must **not** restore an earlier Qdrant snapshot.

Doing so would destroy valid writes from other states.

Instead:

```text
restart IN
      ↓
resume IN
      ↓
finish remaining work
      ↓
IN COMPLETE
      ↓
wave can finalize
```

---

# 31. Crash Between Qdrant Upsert and Ledger Update

A critical failure case is:

```text
1. Worker creates point ID ABC
2. Worker upserts ABC to Qdrant
3. Qdrant successfully commits
4. Worker crashes
5. Local ledger never records completion
```

On resume, the worker may attempt the same chunk again.

Because the point ID is deterministic:

```text
source_id + page + chunk_index + content_hash
→ UUID5
→ ABC
```

the retry upserts `ABC` again rather than creating a new point.

This makes the operation idempotent.

---

# 32. Crash After Global Content Claim

Another important case:

```text
IL claims content_hash abc123
      ↓
IL crashes before completion
```

Without leases, `abc123` could remain permanently unavailable.

With claim leases:

```text
CLAIMED by IL
      ↓
heartbeat stops
      ↓
lease expires
      ↓
claim becomes recoverable
```

A resumed IL worker or another appropriate worker can reclaim the content.

---

# 33. Failed State and Wave Finalization

A wave may not finalize until every expected state reaches a valid terminal completion state.

Example:

```text
IL ✅
IA ✅
IN ❌
MI ✅
OH ✅
```

No snapshot.

No wave manifest.

No global crop merge for that wave.

After IN resumes:

```text
IL ✅
IA ✅
IN ✅
MI ✅
OH ✅
```

Then:

```text
finalize_wave.ipynb
```

may run.

---

# 34. Fresh-State Rebuild Semantics

A distinction is made between:

```text
transient execution failure
```

and:

```text
semantic/correctness error
```

## Transient failure

Use:

```text
RUN_MODE = resume
```

## Semantic bug / wrong metadata / wrong configuration

For the first concurrent architecture version, the system does **not** promise surgical removal and rebuild of one already-ingested state from the shared build.

Reason:

- global deduplication may cause one state to be the canonical owner of content also discovered by another state;
- blindly deleting all points where `ingest_state = IN` could remove content whose existence is logically shared;
- reconstructing shared ownership correctly adds significant complexity.

Therefore:

> Resume transient failures in place. If a state was processed with semantically incorrect inputs/configuration and its indexed content cannot be trusted, begin a corrected build/checkpoint rather than relying on unsafe state-local deletion.

Surgical state rebuild may be added later as a separate capability if necessary.

---

# 35. Build Identity

Every preload run belongs to an explicit build.

Example:

```text
BUILD_ID = mirage_base_2026
```

`BUILD_ID` should be included in:

- state manifests;
- coordinator records;
- wave manifests;
- Qdrant operational metadata where useful;
- snapshot naming;
- persistent state paths.

This prevents accidental mixing between independent preload builds.

---

# 36. Wave Identity

Every concurrent batch belongs to an explicit wave.

Example:

```text
WAVE_ID = wave_01
```

Wave identity should be included in:

- worker configuration;
- state manifest;
- coordinator state status;
- crop merge records;
- snapshot name;
- wave manifest.

Wave size is not encoded into the architecture.

---

# 37. State Manifest

Each worker should produce a state manifest containing at least:

```text
build_id
wave_id
state_name
state_code
status
started_at
completed_at
input file identities/fingerprints
qualification model/config
embedding model/config
chunking configuration
Qdrant collection
Qdrant endpoint used
coordinator endpoint used
accepted document count
rejected document count
duplicate count
chunk count
Qdrant upsert count
crop output path
pipeline version / code revision if available
errors / retry summary
```

A manifest should only be marked:

```text
COMPLETE
```

after state-local validation passes.

---

# 38. Wave Manifest

The finalizer creates a cumulative wave manifest containing:

```text
build_id
wave_id
expected_states
completed_states
state manifest references
global crop JSON version/path
Qdrant collection name
Qdrant point count before wave
Qdrant point count after wave
snapshot identifier/path
snapshot creation timestamp
validation results
finalization status
```

The wave manifest is written last.

---

# 39. Input Handling

Each generic worker folder receives the current state's inputs manually.

Conceptually:

```text
worker_1/input/
  state_pdf.zip
  state_csv.zip
  state_urls.txt
```

Only the intended current state's files should be present.

The notebook's source discovery behavior should operate relative to that worker's current `input/` directory rather than a global preload directory containing files for several states.

This preserves the current easy manual workflow while making each worker independent.

---

# 40. Manual Worker Reuse

Example operational sequence:

```text
Wave 1
worker_1 → Illinois
worker_2 → Iowa
worker_3 → Indiana
```

After wave 1 finalizes:

```text
Wave 2
worker_1 → Michigan
worker_2 → Ohio
worker_3 → Wisconsin
worker_4 → Minnesota
worker_5 → Missouri
```

For each worker:

1. change the settings block;
2. replace the contents of its input folder;
3. run the same notebook.

The worker notebook itself does not need to be specialized by state.

---

# 41. Variable Compute Availability

The architecture explicitly supports fewer than five available GPU notebooks.

Example:

```text
Available GPU notebooks = 3

Wave:
IL
IA
IN
```

Finalizer configuration:

```python
EXPECTED_STATES = ["IL", "IA", "IN"]
```

Nothing waits for worker 4 or worker 5.

A later wave can contain five states if more compute is available.

---

# 42. Coordinator Availability

The coordinator and Qdrant should run for the duration of active worker processing.

If the service-node allocation ends:

```text
Qdrant process stops
Coordinator process stops
```

but persistent storage remains:

```text
qdrant_database/
preload_coordinator.db
persistent_state/
crop_occurrences.json
```

A new service-node allocation can restart:

```text
Qdrant
      ↓
same Qdrant storage

Coordinator
      ↓
same coordinator registry
```

Workers then update their endpoint configuration if the service-node hostname changes.

---

# 43. Network Requirement

The architecture depends on worker nodes being able to reach the service node over the required internal ports.

Required connectivity:

```text
worker → service-node:6333
worker → service-node:8001
```

Before implementing the full concurrent workflow, verify from a separate GPU notebook node:

```bash
curl http://<service-node>:6333/collections
```

and:

```bash
curl http://<service-node>:8001/health
```

If cross-job node networking is blocked by cluster policy, the architecture will need an alternative deployment arrangement.

---

# 44. Recommended Coordinator API Surface

The exact API may evolve, but the initial service should remain intentionally small.

Suggested endpoints:

```text
GET  /health

POST /state/acquire
POST /state/heartbeat
POST /state/complete
POST /state/release
GET  /state/status

POST /content/claim
POST /content/heartbeat
POST /content/complete
POST /content/release

GET  /wave/status
```

The coordinator should not expose arbitrary data-processing endpoints.

---

# 45. Suggested Coordinator Registry Tables

Conceptually:

## State leases

```text
state_leases
-----------
build_id
wave_id
state_code
worker_id
status
acquired_at
last_heartbeat
lease_until
completed_at
```

Unique active identity:

```text
(build_id, state_code)
```

## Content claims

```text
content_claims
--------------
build_id
content_hash
owner_state
worker_id
status
claimed_at
last_heartbeat
lease_until
completed_at
```

Global dedupe identity:

```text
(build_id, content_hash)
```

Additional audit tables may be added later but are not required for the first version.

---

# 46. Coordinator Atomicity

Claim operations must be transactional.

The coordinator must not implement:

```text
SELECT
if absent:
    INSERT
```

without transactional protection.

Instead the database constraint and transaction should make acquisition atomic.

For example, conceptually:

```text
INSERT OR IGNORE
```

or an equivalent transactional pattern.

The same rule applies to state locks.

---

# 47. Heartbeat Policy

Leases need heartbeats so long-running processing does not appear abandoned.

The exact timeout may be configured later.

Conceptually:

```text
worker heartbeat every N minutes
lease expires after M minutes without heartbeat
```

The timeout should be long enough to tolerate individual expensive qualification/embedding batches without constant coordination traffic.

---

# 48. Coordinator Failure Behavior

If the coordinator is temporarily unavailable:

- workers should not assume they own new global claims;
- existing local work may be allowed to pause/retry;
- Qdrant should not be reset;
- workers should retry coordinator calls with bounded backoff;
- completion should not be declared until coordinator state is consistent.

The coordinator is small enough that it should be easier to restart than any GPU worker.

---

# 49. Qdrant Failure Behavior

If Qdrant becomes temporarily unavailable:

- workers retain state-local progress;
- no rollback is performed;
- workers retry bounded operations;
- deterministic point IDs make re-upsert safe;
- the wave remains unfinalized until all workers complete.

A Qdrant outage is therefore recoverable without discarding completed preprocessing.

---

# 50. Snapshot Ownership

Only `finalize_wave.ipynb` owns preload snapshot creation.

Worker notebooks must never call snapshot restore/create logic as part of normal state completion.

The finalizer is the only place permitted to perform build-level checkpoint operations.

---

# 51. Snapshot Restore Ownership

Similarly, snapshot restore must never occur while concurrent worker notebooks are actively writing.

A restore is a build-level administrative action.

If a restore is ever required:

1. stop/pause all workers;
2. confirm no active Qdrant writes;
3. restore the selected cumulative snapshot;
4. reconcile coordinator/build state;
5. resume workers.

This is intentionally outside ordinary worker execution.

---

# 52. Old Sequential Concepts to Remove or Rework

The following concepts from the current one-state-at-a-time notebook should not remain as concurrency correctness assumptions:

```text
RUN_SEQUENCE determines database state
PREVIOUS_RUN_DIR determines current database validity
worker restores previous snapshot
worker validates live count against previous state snapshot
worker creates one cumulative snapshot per state
worker updates global crop_occurrences.json directly
global thread-only lock protects multi-process persistence
```

They are replaced by:

```text
BUILD_ID
WAVE_ID
EXPECTED_STATES
state-local persistence
Coordinator API
shared Qdrant server
wave-level finalizer
cumulative per-wave snapshots
```

---

# 53. Existing Components That Should Be Preserved

The architecture is intentionally a refactor of orchestration, not a rewrite of preload logic.

The following existing behavior should be preserved wherever possible:

- source extraction and normalization;
- canonical content construction;
- metadata validation;
- state/county location handling;
- hardiness-zone derivation;
- qualification logic;
- acceptance/rejection semantics;
- chunking logic;
- embedding model and vector dimensions;
- Qdrant payload schema;
- retry behavior;
- terminal-state validation;
- state-local ledger semantics;
- deterministic persistence;
- crop occurrence extraction logic.

The primary changes concern:

- ownership;
- coordination;
- concurrency;
- snapshot lifecycle;
- global artifact writes.

---

# 54. End-to-End Example

Assume three workers are available.

## Wave configuration

```text
BUILD_ID = mirage_base_2026
WAVE_ID = wave_01

EXPECTED_STATES:
IL
IA
IN
```

## Worker 1

```text
STATE=IL
↓
acquire IL lock
↓
process
↓
global content claims
↓
qualification/chunking/embedding
↓
Qdrant upserts
↓
IL crop JSON
↓
IL manifest COMPLETE
↓
state complete
```

## Worker 2

Same flow for IA.

## Worker 3

Same flow for IN.

## Failure

IN crashes at 47%.

Current state:

```text
IL COMPLETE
IA COMPLETE
IN FAILED/lease expired
```

Qdrant remains untouched.

IN is restarted:

```text
RUN_MODE=resume
```

The new worker:

- reacquires the state lock after expiry;
- loads IN's ledger;
- skips completed units;
- resumes incomplete units;
- safely repeats any ambiguous Qdrant upserts using deterministic point IDs;
- completes IN.

Now:

```text
IL COMPLETE
IA COMPLETE
IN COMPLETE
```

## Finalization

`finalize_wave.ipynb` is configured:

```python
BUILD_ID = "mirage_base_2026"
WAVE_ID = "wave_01"
EXPECTED_STATES = ["IL", "IA", "IN"]
```

It then:

```text
validates IL/IA/IN
↓
merges their crop state into global crop_occurrences.json
↓
validates Qdrant
↓
creates cumulative wave_01 snapshot
↓
writes wave_01 manifest
```

Wave 1 is complete.

---

# 55. Subsequent Wave Example

Later five workers become available.

```text
wave_02:
MI
OH
WI
MN
MO
```

All five notebooks process concurrently.

When all five complete, the finalizer creates a snapshot containing:

```text
wave_01 states
+
wave_02 states
```

The global crop JSON now contains all states processed across both waves.

---

# 56. Invariants

The following invariants define correctness.

## Qdrant invariants

1. Exactly one Qdrant server owns a storage directory at a time.
2. Workers write only through Qdrant's network API.
3. Workers never reset, delete, or restore the cumulative collection.
4. Point IDs are deterministic from:
   `source_id + page + chunk_index + content_hash`.
5. Embedding configuration remains compatible across all workers and waves.

## Worker invariants

1. One worker processes one state at a time.
2. Two active workers may not hold the same state lock.
3. Each state owns its own persistent ledger/canonical area.
4. Worker code is generic; configuration determines state/build/wave.
5. Workers never modify global crop JSON.
6. Workers never create cumulative snapshots.

## Coordinator invariants

1. State acquisition is atomic.
2. Content-hash claims are atomic.
3. Claims use leases.
4. State locks use leases.
5. Only the coordinator process accesses its registry database directly.

## Wave invariants

1. Wave size is variable.
2. Wave membership is explicit.
3. All expected states must complete before finalization.
4. A snapshot represents a complete cumulative wave boundary.
5. A failed state does not roll back other states.

## Crop invariants

1. Each worker emits one state-local crop artifact.
2. Only the finalizer modifies the global crop file.
3. The global crop file persists across waves.
4. Global updates are atomic and backed up.

---

# 57. Implementation Sequence

A low-risk implementation sequence is:

## Phase 1 — Refactor worker persistence

- make state persistence paths depend on `BUILD_ID + STATE_CODE`;
- remove dependence on generic worker directory for persistent state;
- add `BUILD_ID`, `WAVE_ID`, `RUN_MODE`, `COORDINATOR_URL`, and `QDRANT_URL`;
- move input discovery to each worker's local input directory.

## Phase 2 — Deterministic Qdrant IDs

- define canonical logical chunk ID;
- construct:
  `source_id + page + chunk_index + content_hash`;
- convert using UUID5;
- verify repeated upserts are idempotent.

## Phase 3 — Coordinator

- create minimal FastAPI service;
- implement state leases;
- implement content claims;
- implement heartbeats;
- implement completion/status endpoints;
- persist coordinator state.

## Phase 4 — Worker coordinator integration

- acquire/release state lease;
- claim canonical content before global processing;
- complete claims;
- handle expired/retry claims;
- report state completion.

## Phase 5 — Remove unsafe per-state global operations

Remove from workers:

- Qdrant restore;
- Qdrant reset/delete;
- per-state snapshots;
- global crop JSON updates;
- previous-run point-count restoration assumptions.

## Phase 6 — State crop artifacts

- emit `crop_occurrences_state.json`;
- validate state output;
- include path/hash in state manifest.

## Phase 7 — Finalizer notebook

Create:

```text
finalize_wave.ipynb
```

Implement:

- explicit `EXPECTED_STATES`;
- state/wave validation;
- crop merge;
- atomic crop update;
- Qdrant validation;
- cumulative Qdrant snapshot;
- final wave manifest.

## Phase 8 — Failure testing

Test:

- one worker crashes during extraction;
- one crashes after global claim;
- one crashes after Qdrant upsert but before ledger update;
- coordinator restarts;
- Qdrant restarts;
- wave has only 3 of 5 workers;
- duplicate content discovered concurrently;
- duplicate state worker launch;
- finalizer crashes halfway and is rerun.

## Phase 9 — Cross-node network smoke test

Before full deployment, verify from separate notebook nodes:

```text
Coordinator health reachable
Qdrant collections reachable
```

If that succeeds, run a two-state small-scale concurrent preload before scaling to five workers.

---

# 58. Final Architecture Summary

The finalized preload architecture is:

```text
                       SHARED SERVICE NODE
               ┌──────────────────────────┐
               │                          │
               │ Qdrant Server :6333      │
               │                          │
               │ Coordinator API :8001    │
               │   ├─ state leases        │
               │   ├─ global dedupe       │
               │   └─ wave/status state   │
               │                          │
               └────────────┬─────────────┘
                            │
            ┌───────────────┼────────────────┐
            │               │                │
            ↓               ↓                ↓
        Worker 1         Worker 2         Worker N
        1 GPU            1 GPU            1 GPU
        STATE=IL         STATE=IA         STATE=IN
            │               │                │
            ↓               ↓                ↓
        state-local      state-local      state-local
        persistence      persistence      persistence
            │               │                │
            └───────────────┼────────────────┘
                            │
                    direct Qdrant writes
                            │
                            ↓
                  cumulative build collection

                    ALL EXPECTED STATES
                         COMPLETE
                            │
                            ↓
                  finalize_wave.ipynb
                            │
                 ┌──────────┼──────────┐
                 ↓          ↓          ↓
             validate    merge crop   Qdrant
              states       JSON       validate
                 └──────────┼──────────┘
                            ↓
                    cumulative snapshot
                            ↓
                      wave manifest
```

The key architectural shift is:

> **Concurrency is state-level, persistence is state-local, coordination is centralized, vector storage is shared, and checkpointing is wave-level.**

This preserves the strengths of the current deterministic cumulative preload architecture while allowing independent 1-GPU Jupyter notebooks on separate nodes to safely build the same curated MetaMIRAGE Qdrant database in parallel.

---

# 59. Decisions Finalized During Design

The following decisions are considered finalized:

1. Use **state-local** processing ledgers and canonical persistence rather than one shared state-processing SQLite database.
2. Preserve **strict global deduplication** using a centralized atomic coordinator claim registry.
3. Define Qdrant logical chunk identity as:
   `source_id + page + chunk_index + content_hash`, converted deterministically through UUID5.
4. Use explicit `BUILD_ID`, `WAVE_ID`, and manually configured `EXPECTED_STATES`.
5. Allow variable wave sizes; five workers is a practical maximum, not a requirement.
6. Keep five generic, identical worker notebooks/folders and manually update the state configuration/input files.
7. Run exactly one shared Qdrant server.
8. Run the Preload Coordinator API on the same service node as Qdrant.
9. Keep the coordinator outside the data path.
10. Workers send vectors directly to Qdrant.
11. Workers do not snapshot, restore, reset, or delete the shared Qdrant build collection.
12. Create cumulative Qdrant snapshots only at complete wave boundaries.
13. Use a separate `finalize_wave.ipynb`.
14. Each worker writes a state-local crop JSON.
15. The finalizer merges state crop outputs into one persistent global `crop_occurrences.json`.
16. The global crop JSON is maintained cumulatively across waves.
17. Use `resume` for transient failures.
18. Do not initially support unsafe surgical deletion/rebuild of one semantically corrupted state from a shared build.
19. Use leases for both global content claims and state locks.
20. Preserve existing preload extraction, qualification, chunking, embedding, metadata, and validation logic wherever possible.

---

*End of architecture specification.*
