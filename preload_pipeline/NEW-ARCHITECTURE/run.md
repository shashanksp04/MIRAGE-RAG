# MetaMIRAGE Concurrent Preload — Complete Run Guide

**Status:** Current operational guide for the concurrent preload architecture
**Supersedes:** The old sequential one-state-at-a-time `run.md` workflow

This guide explains everything required to run the concurrent MetaMIRAGE preload system end to end:

- one shared **Qdrant server**;
- one shared **Preload Coordinator API**;
- one to five independent **1-GPU worker notebooks**;
- state-local persistence and resume;
- concurrent global deduplication;
- explicit waves with variable numbers of states;
- a separate **wave finalizer notebook**;
- cumulative `crop_occurrences.json`;
- cumulative Qdrant snapshots.

## 1.1 Storage-cache setup

Set these variables before starting workers so model and compiler caches use the
1 TB storage volume rather than the node's limited home/tmp storage:

```bash
mkdir -p /projects/bfox/ssingh38/huggingface_cache/hub
mkdir -p /projects/bfox/ssingh38/triton_cache
export HF_HOME=/projects/bfox/ssingh38/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/projects/bfox/ssingh38/huggingface_cache/hub
export TRITON_CACHE_DIR=/projects/bfox/ssingh38/triton_cache

echo $HF_HOME
echo $HUGGINGFACE_HUB_CACHE
echo $TRITON_CACHE_DIR
```

The architecture source of truth is:

```text
new-architecture-preload-pipeline.md
```

The main runtime files are:

```text
MetaMIRAGE_Concurrent_Preload_Worker.ipynb
preload_coordinator.py
finalize_wave.ipynb
MetaMIRAGE_Cross_Node_Connectivity_Test.ipynb
```

---

# 1. Mental Model

The system is split into a **shared service node** and independent **worker nodes**.

```text
                       SHARED SERVICE NODE
                ┌────────────────────────────┐
                │                            │
                │ Qdrant Server :6333        │
                │                            │
                │ Coordinator API :8001      │
                │                            │
                └─────────────┬──────────────┘
                              │
                              │ HTTP
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ↓                    ↓                    ↓
   GPU Worker Node 1    GPU Worker Node 2    GPU Worker Node 3
   worker_1             worker_2             worker_3
   STATE=IL             STATE=IA             STATE=IN

         ↓                    ↓                    ↓

   GPU Worker Node 4    GPU Worker Node 5
   worker_4             worker_5
   STATE=MI             STATE=OH
```

A normal deployment therefore needs:

```text
1 CPU/service allocation
+
1–5 independent GPU/Jupyter allocations
```

The number of workers is **not fixed**.

A wave can contain:

```text
1 state
2 states
3 states
4 states
or 5 states
```

depending on available compute.

---

# 2. Critical Architecture Rules

These rules must always hold.

## 2.1 One Qdrant server

Exactly one Qdrant server process owns the preload Qdrant storage directory.

Correct:

```text
Workers → HTTP → ONE Qdrant server → one Qdrant storage directory
```

Incorrect:

```text
Qdrant A ─┐
Qdrant B ─┼→ same storage directory
Qdrant C ─┘
```

Never run multiple Qdrant server processes against the same Qdrant storage.

---

## 2.2 Workers never own global Qdrant lifecycle

Worker notebooks may:

- query the shared collection;
- embed chunks;
- upsert points;
- ensure payload indexes exist.

Worker notebooks must never:

- reset the collection;
- delete the collection;
- restore a snapshot;
- create a cumulative wave snapshot.

Snapshot creation belongs only to:

```text
finalize_wave.ipynb
```

---

## 2.3 One state per worker

Each generic worker notebook processes one state at a time.

Example:

```text
worker_1 → Illinois
worker_2 → Iowa
worker_3 → Indiana
```

Do not process two states inside one worker at the same time.

---

## 2.4 One active worker per state

Two workers must never process the same `(BUILD_ID, STATE_CODE)` simultaneously.

The coordinator enforces this through state leases.

---

## 2.5 Waves are explicit and variable-sized

Example:

```text
wave_01
  IL
  IA
  IN
```

Next wave:

```text
wave_02
  MI
  OH
  WI
  MN
  MO
```

The finalizer is manually configured with the exact states belonging to that wave.

---

## 2.6 Do not overlap waves

States **within** one wave may run concurrently.

Different waves should not overlap.

Correct:

```text
wave_01 workers
      ↓
all complete
      ↓
finalize wave_01
      ↓
global crop updated + snapshot created
      ↓
start wave_02 workers
```

Do not start `wave_02` workers before `wave_01` finalization completes.

This matters because every state in a wave must start from the same maintained global `crop_occurrences.json` seed.

---

# 3. Recommended Directory Layout

Keep the concurrent preload setup on storage visible from all notebook nodes.

Recommended layout:

```text
preload_pipeline/
│
├── run.md
├── new-architecture-preload-pipeline.md
├── preload_coordinator.py
├── finalize_wave.ipynb
│
├── workers/
│   ├── worker_1/
│   │   ├── MetaMIRAGE_Concurrent_Preload_Worker.ipynb
│   │   ├── input/
│   │   └── .python_packages/              # created by notebook
│   │
│   ├── worker_2/
│   │   ├── MetaMIRAGE_Concurrent_Preload_Worker.ipynb
│   │   └── input/
│   │
│   ├── worker_3/
│   ├── worker_4/
│   └── worker_5/
│
├── persistent_state/
│   └── <BUILD_ID>/
│       ├── IL/
│       │   ├── canonical/
│       │   ├── input_staging/
│       │   ├── pipeline_state.db
│       │   ├── crop_occurrences_state.json
│       │   └── state_manifest.json
│       │
│       ├── IA/
│       └── ...
│
├── shared/
│   ├── county_state_hardiness_zone.csv
│   ├── crop_occurrences.json
│   │
│   ├── finalization/
│   │   └── <BUILD_ID>/<WAVE_ID>/
│   │       ├── finalization_state.json
│   │       └── wave_manifest.json
│   │
│   ├── crop_backups/
│   │   └── <BUILD_ID>/
│   │
│   └── snapshots/
│       └── <BUILD_ID>/<WAVE_ID>/
│
└── finalizer/
    └── finalize_wave.ipynb
```

The five worker folders are generic execution slots.

Persistent state belongs to:

```text
BUILD_ID + STATE_CODE
```

not to:

```text
worker_1
worker_2
...
```

Therefore `worker_1` can process Illinois in one wave and Wisconsin in a later wave without mixing their durable state.

---

# 4. Required Shared Support Files

Before running any worker, place these files under:

```text
shared/
```

Required:

```text
shared/county_state_hardiness_zone.csv
shared/crop_occurrences.json
```

The worker notebook resolves:

```python
GLOBAL_CROP_OCCURRENCE_JSON = SHARED_ARTIFACTS_DIR / "crop_occurrences.json"
HARDINESS_CSV = SHARED_ARTIFACTS_DIR / "county_state_hardiness_zone.csv"
```

## 4.1 `county_state_hardiness_zone.csv`

Used to derive state/county hardiness-zone metadata.

Keep one shared immutable copy during the build.

---

## 4.2 `crop_occurrences.json`

This is the one maintained global crop JSON across waves.

Workers:

```text
READ it
```

Workers do **not** modify it.

Each worker produces:

```text
persistent_state/<BUILD_ID>/<STATE_CODE>/crop_occurrences_state.json
```

Then `finalize_wave.ipynb` merges the current wave's state outputs into:

```text
shared/crop_occurrences.json
```

The global file persists across all waves.

### Important

All workers in one wave must read the **same exact version** of the global crop file.

The worker stores its fingerprint in the state manifest, and the finalizer verifies that every expected state used the same seed.

---

# 5. Input Layout for Each Worker

Each worker auto-discovers the current state's inputs from:

```text
workers/worker_N/input/
```

For example:

```text
workers/worker_1/input/
├── Illinois-PDFS.zip
├── Illinois-CSV.zip
└── Illinois-URL.xlsx
```

A state does not need all three input types.

At least one source type must be present.

---

# 6. Input Naming Rules

Matching is case-insensitive.

## PDF

At most one `.zip` whose name contains:

```text
PDF
```

Examples:

```text
Illinois-PDFS.zip
IL_PDF.zip
PDF_documents.zip
```

---

## CSV

At most one `.zip` whose name contains:

```text
CSV
```

Examples:

```text
Illinois-CSV.zip
IL_CSV.zip
CSV_sources.zip
```

Every CSV inside the ZIP is discovered recursively.

---

## URL input

At most one URL file whose name contains:

```text
URL
```

Supported extensions:

```text
.txt
.xlsx
.xlsm
.xls
```

Examples:

```text
Illinois-URL.txt
Illinois_URLs.xlsx
URL_sources.xls
```

---

# 7. First-Time Qdrant Binary Setup

If Qdrant 1.18.0 is already installed at:

```text
~/bin/qdrant
```

skip this section.

Otherwise:

```bash
mkdir -p ~/bin
cd ~/bin

wget https://github.com/qdrant/qdrant/releases/download/v1.18.0/qdrant-x86_64-unknown-linux-musl.tar.gz

tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz

chmod +x qdrant

./qdrant --version
```

Expected:

```text
qdrant 1.18.0
```

The Python package `qdrant-client` is not the server. The binary above is the server process.

---

# 8. Allocate the Shared Service Node

For every active wave, obtain one sufficiently long **CPU allocation**.

This node runs:

```text
Qdrant
+
Preload Coordinator
```

It does not need a GPU.

Once on the allocated node:

```bash
hostname -f
```

Example:

```text
gpub053.delta.ncsa.illinois.edu
```

Save this hostname.

Every worker notebook needs it.

---

# 9. Persistent Service Storage

Do not run real preload using Qdrant's default:

```text
./storage
```

Use an explicit persistent location.

Example:

```bash
export PRELOAD_SERVICE_ROOT=/work/nvme/bfox/ssingh38/metamirage_preload_service

mkdir -p "$PRELOAD_SERVICE_ROOT/qdrant_database"
mkdir -p "$PRELOAD_SERVICE_ROOT/logs"
```

Then use:

```text
$PRELOAD_SERVICE_ROOT/qdrant_database
```

for Qdrant and:

```text
$PRELOAD_SERVICE_ROOT/preload_coordinator.db
```

for the coordinator registry.

Adjust the path for your allocation/project if necessary.

The important property is:

> The path must persist after the service-node allocation ends.

---

# 10. Start Qdrant on the Service Node

Set the explicit storage location:

```bash
export QDRANT__STORAGE__STORAGE_PATH="$PRELOAD_SERVICE_ROOT/qdrant_database"
```

Start Qdrant:

```bash
~/bin/qdrant
```

Leave it running.

Qdrant should report:

```text
Qdrant HTTP listening on 6333
```

and should bind to:

```text
0.0.0.0:6333
```

so other allocated nodes can reach it.

---

# 11. Verify Qdrant Locally

In another terminal on the same service node:

```bash
curl http://127.0.0.1:6333/collections
```

Expected:

```json
{
  "result": {
    "collections": []
  },
  "status": "ok"
}
```

or an existing collection list.

---

# 12. Initialize the Shared Preload Collection

The generic worker intentionally does **not** create the build-level collection.

It expects:

```text
mirage_base_build
```

to exist before workers start.

The embedding model is:

```text
BAAI/bge-base-en-v1.5
```

with a 768-dimensional embedding space and cosine distance.

## 12.1 Check whether it already exists

```bash
curl http://127.0.0.1:6333/collections
```

If:

```text
mirage_base_build
```

already exists for the current build, do not recreate it.

---

## 12.2 Create it for a new build

```bash
curl -X PUT \
  "http://127.0.0.1:6333/collections/mirage_base_build" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'
```

Verify:

```bash
curl http://127.0.0.1:6333/collections/mirage_base_build
```

---

## 12.3 Create runtime-required payload indexes

Workers also ensure these idempotently, but creating them centrally is recommended.

```bash
for field in hardiness_zone month_year title content_hash; do
  curl -X PUT \
    "http://127.0.0.1:6333/collections/mirage_base_build/index?wait=true" \
    -H "Content-Type: application/json" \
    -d "{
      \"field_name\": \"${field}\",
      \"field_schema\": \"keyword\"
    }"
done
```

Required indexed fields:

```text
hardiness_zone
month_year
title
content_hash
```

---

# 13. Coordinator First-Time Python Setup

The coordinator is a small FastAPI service.

On the service node, use an environment where you can install:

```text
fastapi
uvicorn
```

For example:

```bash
python3 -m venv ~/metamirage-coordinator-venv

source ~/metamirage-coordinator-venv/bin/activate

pip install --upgrade pip

pip install fastapi uvicorn
```

This only needs to be done once unless the environment is removed.

---

# 14. Start the Preload Coordinator

From the directory containing:

```text
preload_coordinator.py
```

activate the coordinator environment:

```bash
source ~/metamirage-coordinator-venv/bin/activate
```

Configure persistent registry storage:

```bash
export METAMIRAGE_COORDINATOR_DB="$PRELOAD_SERVICE_ROOT/preload_coordinator.db"
export METAMIRAGE_COORDINATOR_HOST=0.0.0.0
export METAMIRAGE_COORDINATOR_PORT=8001
```

Start it:

```bash
python preload_coordinator.py
```

Leave it running for the entire wave.

The coordinator uses:

```text
:8001
```

---

# 15. Verify the Coordinator Locally

On the service node:

```bash
curl http://127.0.0.1:8001/health
```

Expected:

```json
{
  "status": "ok",
  "service": "metamirage-preload-coordinator",
  ...
}
```

---

# 16. Service Node Process Layout

The service allocation should now have both:

```text
Qdrant      :6333
Coordinator :8001
```

Conceptually:

```text
Service Node
├── ~/bin/qdrant
│      └── storage → persistent qdrant_database/
│
└── python preload_coordinator.py
       └── state → persistent preload_coordinator.db
```

You may use two terminals, `tmux`, or background processes while the allocation remains active.

Example background logging:

```bash
export QDRANT__STORAGE__STORAGE_PATH="$PRELOAD_SERVICE_ROOT/qdrant_database"

nohup ~/bin/qdrant \
  > "$PRELOAD_SERVICE_ROOT/logs/qdrant.log" 2>&1 &
```

Coordinator:

```bash
source ~/metamirage-coordinator-venv/bin/activate

export METAMIRAGE_COORDINATOR_DB="$PRELOAD_SERVICE_ROOT/preload_coordinator.db"
export METAMIRAGE_COORDINATOR_HOST=0.0.0.0
export METAMIRAGE_COORDINATOR_PORT=8001

nohup python preload_coordinator.py \
  > "$PRELOAD_SERVICE_ROOT/logs/coordinator.log" 2>&1 &
```

Remember: when the CPU allocation ends, both processes end even though the persistent data remains.

---

# 17. Test Cross-Node Connectivity

Before a real wave, use:

```text
MetaMIRAGE_Cross_Node_Connectivity_Test.ipynb
```

from a separate Jupyter node.

Set:

```python
SERVICE_NODE = "<service-node-hostname>"
```

Example:

```python
SERVICE_NODE = "gpub053.delta.ncsa.illinois.edu"
```

The important URLs become:

```text
http://gpub053.delta.ncsa.illinois.edu:6333
http://gpub053.delta.ncsa.illinois.edu:8001
```

Required successful tests:

```text
TCP :6333
GET /collections
QdrantClient.get_collections()

TCP :8001
GET /health
```

We have already verified that Delta worker nodes can communicate with Qdrant on another allocated node using this pattern. Repeat the test when the service-node hostname changes if needed.

---

# 18. Allocate Worker Jupyter Nodes

Get between one and five 1-GPU Jupyter allocations.

Example wave with three available workers:

```text
worker_1 → IL
worker_2 → IA
worker_3 → IN
```

Unused generic workers simply remain unused.

---

# 19. Prepare Each Generic Worker Folder

For each worker:

```text
workers/worker_1/
workers/worker_2/
...
```

make sure the folder contains:

```text
MetaMIRAGE_Concurrent_Preload_Worker.ipynb
input/
```

Before assigning a new state, remove/move the previous state's contents from:

```text
input/
```

Then place only the current state's inputs there.

Example:

```text
workers/worker_1/input/
├── Illinois-PDFS.zip
├── Illinois-CSV.zip
└── Illinois-URL.xlsx
```

---

# 20. Worker Dependency Installation

The notebook contains its own installation cell.

It installs Python dependencies into:

```text
workers/worker_N/.python_packages/
```

rather than trying to modify Delta's read-only shared Python environment.

Run the install cell before the import/configuration cells.

The cell intentionally avoids reinstalling PyTorch.

Because every worker folder has a separate `.python_packages`, each generic worker can be independently prepared.

If the package directory already exists and works, the install cell may show many already-installed/cached packages; that is fine.

---

# 21. Hugging Face Access

The qualification model is:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

The worker checks:

```python
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
```

If it is missing when the classifier loads, the notebook prompts you.

Do **not** store an access token in `run.md`.

Recommended options:

### Notebook session

Before model loading:

```python
import os
os.environ["HF_TOKEN"] = "<your-token>"
```

or enter the token when prompted.

### Environment inherited by Jupyter

If your Jupyter environment already has `HF_TOKEN`, no extra action is required.

Verify without printing the token:

```python
import os
print("HF token present:", bool(os.environ.get("HF_TOKEN")))
```

---

# 22. Configure a Worker

In the worker configuration cell, edit:

```python
BUILD_ID = "build_2026_08"
WAVE_ID = "wave_01"

STATE_NAME = "Illinois"
STATE_CODE = "IL"

RUN_MODE = "resume"
```

Use the same:

```text
BUILD_ID
WAVE_ID
```

for every worker in the current wave.

Use different:

```text
STATE_NAME
STATE_CODE
```

for each worker.

---

# 23. `WORKER_ID` Is Automatic

Do **not** manually configure `WORKER_ID`.

The notebook automatically creates a session-specific value from:

```text
worker folder name
+
hostname
+
process ID
+
random session suffix
```

Example conceptually:

```text
worker_1:gpub123-45678-a1b2c3d4
```

This allows the same state to resume on another notebook/node while keeping the state identity stable through:

```text
RUN_ID = BUILD_ID + STATE_CODE
```

---

# 24. Configure Service URLs in Every Worker

The worker defaults are:

```text
127.0.0.1
```

which are wrong for remote service-node operation.

Every worker must point to the service node.

Example:

```python
COORDINATOR_URL = "http://gpub053.delta.ncsa.illinois.edu:8001"
QDRANT_URL = "http://gpub053.delta.ncsa.illinois.edu:6333"
```

You may either:

1. edit those values directly in the configuration cell; or
2. set environment variables before running the configuration cell.

For example:

```python
import os

os.environ["METAMIRAGE_COORDINATOR_URL"] = (
    "http://gpub053.delta.ncsa.illinois.edu:8001"
)

os.environ["QDRANT_URL"] = (
    "http://gpub053.delta.ncsa.illinois.edu:6333"
)
```

Then rerun the configuration cell.

Directly editing the notebook config is often simplest because separate Jupyter kernels do not automatically inherit environment variables exported from unrelated shell sessions.

---

# 25. Shared Collection Name

For the normal build, all workers use:

```python
QDRANT_COLLECTION = "mirage_base_build"
```

Every worker in the build must use the same collection.

The worker has:

```python
WORKER_MAY_CREATE_QDRANT_COLLECTION = False
```

by design.

If the collection is missing, the worker fails rather than creating an accidental per-worker database.

---

# 26. `RUN_MODE`

Current concurrent workers support:

```python
RUN_MODE = "resume"
```

only.

This is intentional.

Use `resume` for:

- kernel crash;
- allocation timeout;
- network error;
- transient Qdrant failure;
- transient coordinator failure;
- temporary model error;
- restart on another node.

Do not use the current shared build for a state that was processed with semantically wrong inputs/configuration.

For a correctness rebuild, plan a new build instead of trying to surgically remove one state's shared contributions.

---

# 27. Verify Worker Input Discovery

After running the configuration/state-layout cells, verify:

```python
print("URL_FILE =", URL_FILE)
print("PDF_ZIP_FILE =", PDF_ZIP_FILE)
print("CSV_ZIP_FILE =", CSV_ZIP_FILE)
print("CSV_INPUTS =", CSV_INPUTS)
```

At least one source must exist.

Expected PDF-only example:

```text
URL_FILE = None
PDF_ZIP_FILE = .../worker_1/input/Illinois-PDFS.zip
CSV_ZIP_FILE = None
CSV_INPUTS = []
```

If no sources are configured, the worker fails preflight intentionally.

---

# 28. Important Configuration Rerun Warning

The main configuration cell initializes:

```python
URL_FILE = None
PDF_ZIP_FILE = None
CSV_ZIP_FILE = None
CSV_INPUTS = []
```

The following layout/discovery cell populates them.

If you rerun only the configuration cell later, rerun the discovery/layout cell afterward as well.

Before starting expensive execution, verify the discovered input variables again.

---

# 29. Worker Safety Switch

Every worker defaults to:

```python
RUN_PIPELINE = False
```

This prevents accidental large runs.

Recommended workflow:

```text
run dependency/install cells
↓
run imports/configuration
↓
run state layout/input discovery
↓
run setup/helper cells
↓
review settings
↓
verify services + inputs
↓
set RUN_PIPELINE=True
↓
run orchestrator cell
```

The cleanest method is a tiny separate cell:

```python
RUN_PIPELINE = True
```

Then execute the final orchestrator cell.

---

# 30. What a Worker Does

The worker runs:

```text
1. Preflight
   ├── input exists
   ├── Qdrant reachable
   └── coordinator reachable

2. Acquire state lease

3. Bind/validate input fingerprints

4. Discover sources

5. Extract + canonicalize

6. Global document dedupe claim
   through coordinator

7. Qualification
   Meta-Llama-3.1-8B-Instruct

8. Qualification retries + accept/reject

9. Unload classifier model

10. RAG chunking + metadata contract

11. Validate shared Qdrant collection

12. Global RAG-chunk dedupe claim
    through coordinator

13. BGE embedding on worker GPU

14. Direct Qdrant upsert

15. Retry failed RAG units

16. Final state validation

17. Write:
    crop_occurrences_state.json
    state_manifest.json

18. Mark state COMPLETE in coordinator
```

The worker then stops.

It does **not** snapshot Qdrant or update the global crop file.

---

# 31. Global Deduplication

The coordinator handles two claim scopes:

```text
document
rag_chunk
```

This preserves the original preload semantics while multiple states run concurrently.

Example:

```text
IL discovers document hash abc
IA discovers document hash abc

           ↓

Coordinator atomically grants one owner.

IL → CLAIMED
IA → ALREADY_COMPLETE / CLAIMED_BY_OTHER
```

The coordinator's persistent state lives in:

```text
preload_coordinator.db
```

on the service node's persistent storage.

---

# 32. Deterministic Qdrant Point IDs

Every RAG chunk uses the agreed logical identity:

```text
source_id
+
page
+
chunk_index
+
content_hash
```

Conceptually:

```text
source_id|page|chunk_index|content_hash
        ↓
UUID5
        ↓
Qdrant point ID
```

This makes retries idempotent.

If Qdrant successfully stores a point but the notebook crashes before SQLite records completion, the resumed worker regenerates the same point ID.

---

# 33. State-Local Persistence

Each state writes only to:

```text
persistent_state/<BUILD_ID>/<STATE_CODE>/
```

Example:

```text
persistent_state/build_2026_08/IL/
├── canonical/
├── input_staging/
├── pipeline_state.db
├── crop_occurrences_state.json
└── state_manifest.json
```

Do not delete this directory after a transient failure.

It is the basis for resume.

---

# 34. Resume After Worker Failure

Example wave:

```text
IL ✅ complete
IA ✅ complete
IN ❌ crashed at 47%
MI ✅ complete
OH ✅ complete
```

Do **not**:

- roll back Qdrant;
- restore an old snapshot;
- rerun the other four states;
- delete IN's state directory.

Instead:

1. obtain/reuse a worker notebook;
2. configure the same:

```text
BUILD_ID
WAVE_ID
STATE_NAME
STATE_CODE
```

3. place/retain the exact same input files;
4. use:

```python
RUN_MODE = "resume"
```

5. point it to the same coordinator/Qdrant;
6. run again.

The coordinator lease and state-local ledger reconcile the interrupted work.

---

# 35. Input Fingerprints on Resume

The worker fingerprints its inputs.

If you try to resume a state but silently replace/change the source files, the worker should refuse to treat it as the same resumable state.

For a transient failure:

> Resume with the same inputs.

For a semantic/correctness change:

> Use a new planned build rather than pretending it is the same run.

---

# 36. State Completion

A state is considered finalizable only when both are true:

1. shared persistent storage contains a valid:

```text
state_manifest.json
```

with:

```text
status = complete
```

2. coordinator `/state/status` says:

```text
complete
```

This two-part check closes the crash window between writing a manifest and committing coordinator completion.

---

# 37. Do Not Snapshot When an Individual State Finishes

Suppose:

```text
IL 100%
IA 70%
IN 40%
MI 85%
OH 20%
```

An IL snapshot would capture a mixed partial wave.

Therefore state workers never snapshot.

Wait until all explicitly expected states complete.

---

# 38. Check Wave Status

Coordinator status endpoint:

```text
GET /wave/status
```

Example from any node with network access:

```bash
curl \
  "http://<service-node>:8001/wave/status?build_id=build_2026_08&wave_id=wave_01"
```

This is useful for visibility.

The finalizer still relies on its explicit `EXPECTED_STATES` list rather than assuming every coordinator record belongs to the intended wave.

---

# 39. Prepare the Wave Finalizer

After all states in the wave are complete, open:

```text
finalize_wave.ipynb
```

This is a separate notebook.

It does not require a GPU.

It may run from any Jupyter/CPU node that:

- can access the shared preload filesystem;
- can reach Qdrant;
- can reach the coordinator.

---

# 40. Configure the Finalizer

Edit:

```python
BUILD_ID = "build_2026_08"
WAVE_ID = "wave_01"

EXPECTED_STATES = [
    "IL",
    "IA",
    "IN",
]
```

The number of states is arbitrary.

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

Also configure:

```python
COORDINATOR_URL = "http://<service-node>:8001"
QDRANT_URL = "http://<service-node>:6333"

QDRANT_COLLECTION = "mirage_base_build"
```

---

# 41. Finalizer Safety Switch

The finalizer defaults to:

```python
RUN_FINALIZATION = False
```

Run the notebook through its validation and preview cells first.

It will check:

- coordinator reachable;
- Qdrant reachable;
- collection exists;
- every expected state manifest exists;
- every state manifest says complete;
- every coordinator state says complete;
- manifest SHA matches coordinator;
- every state crop artifact hash is valid;
- all states agree on build contracts;
- all states used the same global crop seed;
- Qdrant point count is nonzero;
- sample payload contract is readable.

Only after these pass should you change:

```python
RUN_FINALIZATION = True
```

and execute the finalization cell.

---

# 42. Finalizer Global Crop Merge

The finalizer merges only the current wave's state outputs into:

```text
shared/crop_occurrences.json
```

Example before wave:

```json
{
  "illinois": {},
  "iowa": {},
  "indiana": {}
}
```

Wave adds:

```text
MI
OH
WI
```

After finalization:

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

The global crop file is cumulative across waves.

Before replacement, the finalizer creates a backup under:

```text
shared/crop_backups/<BUILD_ID>/
```

The replacement is atomic.

---

# 43. Finalizer Qdrant Snapshot

After crop merge and Qdrant validation, the finalizer creates one cumulative Qdrant snapshot.

Snapshot semantics:

```text
wave_01 snapshot
=
all states through wave_01
```

```text
wave_02 snapshot
=
all wave_01 states
+
all wave_02 states
```

The snapshot is therefore cumulative.

If:

```python
DOWNLOAD_SNAPSHOT = True
```

the finalizer also downloads it to:

```text
shared/snapshots/<BUILD_ID>/<WAVE_ID>/
```

---

# 44. Finalization Journal

The finalizer maintains:

```text
shared/finalization/<BUILD_ID>/<WAVE_ID>/finalization_state.json
```

This supports recovery if the finalizer kernel dies partway through.

It records stages such as:

```text
started
crop_merging
crop_merged
snapshotting
complete
```

---

# 45. Wave Manifest

The finalizer writes:

```text
shared/finalization/<BUILD_ID>/<WAVE_ID>/wave_manifest.json
```

**last**.

This is the commit marker for a successfully finalized wave.

It includes:

- expected states;
- state manifests;
- state statistics;
- global crop hash;
- Qdrant point count;
- snapshot information;
- validation information.

---

# 46. Finalizer Lock

The finalizer creates:

```text
shared/.finalize_wave.lock
```

to prevent two finalizers from mutating global state concurrently.

If a finalizer crashes and leaves a stale lock:

1. confirm no finalizer process is still running;
2. inspect the finalization journal;
3. only then remove the stale lock manually.

Never run two finalizers concurrently.

---

# 47. After a Wave Finalizes

Only after `wave_manifest.json` exists with:

```text
status = complete
```

should you start the next wave.

Then:

1. choose the next:

```text
WAVE_ID
```

2. reuse the same:

```text
BUILD_ID
QDRANT_COLLECTION
```

3. assign new states to available generic workers;
4. replace each worker's `input/` contents;
5. configure the new state;
6. run concurrent workers;
7. finalize the new wave.

---

# 48. Reusing Generic Worker Folders

Example:

## Wave 1

```text
worker_1 → IL
worker_2 → IA
worker_3 → IN
```

## Wave 2

```text
worker_1 → MI
worker_2 → OH
worker_3 → WI
worker_4 → MN
worker_5 → MO
```

Before reuse:

- replace `input/` contents;
- edit `STATE_NAME`;
- edit `STATE_CODE`;
- edit `WAVE_ID`.

Do not delete the previous state's durable directory under:

```text
persistent_state/
```

---

# 49. Service Node Allocation Ends

When the CPU allocation ends:

```text
Qdrant process stops
Coordinator process stops
```

but persistent data should remain:

```text
qdrant_database/
preload_coordinator.db
```

On the next service allocation:

1. get the new hostname:

```bash
hostname -f
```

2. point Qdrant at the same persistent storage:

```bash
export QDRANT__STORAGE__STORAGE_PATH="$PRELOAD_SERVICE_ROOT/qdrant_database"
~/bin/qdrant
```

3. start coordinator using the same registry:

```bash
export METAMIRAGE_COORDINATOR_DB="$PRELOAD_SERVICE_ROOT/preload_coordinator.db"
python preload_coordinator.py
```

4. verify:

```bash
curl http://127.0.0.1:6333/collections
curl http://127.0.0.1:8001/health
```

5. update worker/finalizer service URLs to the new hostname.

---

# 50. Important: Never Run Two Service Qdrant Processes

Before starting Qdrant using the persistent storage path, make sure there is not another active allocation already running Qdrant against that same storage.

Only one server may own it at a time.

---

# 51. Qdrant Snapshot Restore

Snapshot restore is **not** part of ordinary worker or finalizer execution.

If a restore is ever required:

1. stop all worker notebooks;
2. stop all writes;
3. stop/reconcile the coordinator workflow as needed;
4. perform the restore as an administrative operation;
5. reconcile coordinator/build state;
6. restart workers.

Never restore a Qdrant snapshot while concurrent workers are writing.

---

# 52. Semantic Error / Wrong State Configuration

Transient errors use:

```text
resume
```

But suppose a state was processed with:

- wrong metadata;
- wrong state;
- wrong source inputs;
- a semantic bug in preprocessing.

Do not attempt to delete only that state's vectors from the current shared build.

Global deduplication means ownership may be shared conceptually across state discovery.

For the initial architecture:

> Treat semantic corruption as a new-build problem.

Do not silently continue the corrupted build.

---

# 53. New Build Policy

Normal waves should keep the same:

```text
BUILD_ID
QDRANT_COLLECTION
```

for the entire cumulative build.

A genuinely new build should be planned deliberately.

Safe options include:

- create a new `BUILD_ID` and new Qdrant collection name; or
- perform a coordinated full reset of Qdrant + coordinator + persistent build state before any worker starts.

Do not merely change `BUILD_ID` while continuing to write into an old unrelated Qdrant collection unless that is explicitly intended.

---

# 54. Debug / Smoke-Test Build

Before a large real build, run a tiny two-state integration test.

Use a separate debug identity:

```python
BUILD_ID = "debug_build_01"
WAVE_ID = "wave_01"

DEBUG_SOURCE_LIMIT = 5
QDRANT_COLLECTION = "mirage_debug_build"
```

Create the debug collection centrally first.

Run:

```text
worker_1 → small IL input
worker_2 → small IA input
```

Verify:

- both acquire different state leases;
- global document claims work;
- global chunk claims work;
- both write to one Qdrant collection;
- each creates its own state directory;
- each creates a state manifest;
- neither snapshots;
- neither edits global crop JSON.

Then test `finalize_wave.ipynb` with:

```python
EXPECTED_STATES = ["IL", "IA"]
```

After testing, remove the debug artifacts deliberately.

Do not use `DEBUG_SOURCE_LIMIT` on a real state's production `BUILD_ID`, because a state can become marked complete after only the limited sources.

---

# 55. Recommended Failure Tests Before Full Scale

Before relying on five concurrent workers, test:

1. Kill a worker during extraction.
2. Restart it with the same state and verify resume.
3. Kill a worker after some Qdrant writes.
4. Verify deterministic IDs prevent duplicates.
5. Launch two workers for the same state and confirm the second is rejected.
6. Feed the same source to two different states and confirm coordinator dedupe.
7. Restart coordinator using the same DB.
8. Restart Qdrant using the same storage.
9. Run a 3-state wave rather than 5 states.
10. Kill the finalizer after crop merge and confirm rerun recovery.

---

# 56. Useful Service Commands

## Service hostname

```bash
hostname -f
```

---

## Qdrant collections

Local service node:

```bash
curl http://127.0.0.1:6333/collections
```

Remote worker:

```bash
curl http://<service-node>:6333/collections
```

---

## Inspect preload collection

```bash
curl http://<service-node>:6333/collections/mirage_base_build
```

---

## Exact Qdrant count

```bash
curl -X POST \
  "http://<service-node>:6333/collections/mirage_base_build/points/count" \
  -H "Content-Type: application/json" \
  -d '{"exact": true}'
```

---

## Coordinator health

```bash
curl http://<service-node>:8001/health
```

---

## State status

```bash
curl \
  "http://<service-node>:8001/state/status?build_id=build_2026_08&state_code=IL"
```

---

## Wave status

```bash
curl \
  "http://<service-node>:8001/wave/status?build_id=build_2026_08&wave_id=wave_01"
```

---

## Qdrant process

```bash
ps aux | grep qdrant
```

---

## Coordinator process

```bash
ps aux | grep preload_coordinator
```

---

## Logs if launched in background

```bash
tail -f "$PRELOAD_SERVICE_ROOT/logs/qdrant.log"
```

```bash
tail -f "$PRELOAD_SERVICE_ROOT/logs/coordinator.log"
```

---

# 57. Full Wave Run — Condensed Procedure

## Step A — Service node

```text
1. Obtain CPU allocation.
2. hostname -f
3. Set PRELOAD_SERVICE_ROOT.
4. Start Qdrant with persistent storage.
5. Verify :6333.
6. Ensure mirage_base_build exists.
7. Start coordinator with persistent DB.
8. Verify :8001.
```

---

## Step B — Worker nodes

For each available state:

```text
1. Obtain 1-GPU Jupyter allocation.
2. Open one generic worker folder.
3. Replace worker input/ contents.
4. Run package-install cell.
5. Configure BUILD_ID/WAVE_ID/STATE.
6. Configure Qdrant + coordinator URLs.
7. Verify shared support files.
8. Verify input discovery.
9. Make HF token available.
10. Keep RUN_PIPELINE=False while reviewing.
11. Set RUN_PIPELINE=True.
12. Run pipeline.
13. Wait for state COMPLETE.
```

---

## Step C — Failed worker

```text
1. Do not rollback anything.
2. Keep state persistent directory.
3. Reopen/reassign a generic worker.
4. Use same BUILD_ID/WAVE_ID/STATE.
5. Use same inputs.
6. RUN_MODE="resume".
7. Run again.
```

---

## Step D — Finalize wave

```text
1. Confirm every EXPECTED_STATE is COMPLETE.
2. Open finalize_wave.ipynb.
3. Configure BUILD_ID/WAVE_ID/EXPECTED_STATES.
4. Configure service URLs.
5. RUN_FINALIZATION=False.
6. Run validation/preview.
7. Confirm no state workers are writing.
8. Set RUN_FINALIZATION=True.
9. Run finalization.
10. Verify wave_manifest.json.
11. Verify crop_occurrences.json update.
12. Verify cumulative Qdrant snapshot.
```

---

## Step E — Next wave

```text
1. Do not start until previous wave finalized.
2. Keep BUILD_ID.
3. Increment/change WAVE_ID.
4. Reuse generic workers.
5. Assign new states.
6. Repeat.
```

---

# 58. Full Operational Checklist

## Before the service starts

```text
[ ] Qdrant binary exists
[ ] preload_coordinator.py exists
[ ] persistent service storage directory exists
[ ] no other Qdrant process is using that storage
```

## Service node

```text
[ ] CPU allocation active
[ ] hostname -f recorded
[ ] QDRANT__STORAGE__STORAGE_PATH set
[ ] Qdrant started
[ ] Qdrant :6333 reachable locally
[ ] mirage_base_build exists
[ ] payload indexes exist / workers can ensure them
[ ] coordinator DB path set
[ ] coordinator started
[ ] coordinator :8001 health succeeds
```

## Shared preload filesystem

```text
[ ] shared/county_state_hardiness_zone.csv exists
[ ] shared/crop_occurrences.json exists
[ ] persistent_state/ accessible from worker nodes
[ ] finalizer/ accessible
```

## Every worker

```text
[ ] correct generic worker folder
[ ] only assigned state's current inputs under input/
[ ] BUILD_ID correct
[ ] WAVE_ID correct
[ ] STATE_NAME correct
[ ] STATE_CODE correct
[ ] RUN_MODE="resume"
[ ] QDRANT_URL points to service node
[ ] COORDINATOR_URL points to service node
[ ] QDRANT_COLLECTION correct
[ ] HF token available
[ ] input discovery correct
[ ] Qdrant reachable
[ ] coordinator reachable
[ ] RUN_PIPELINE reviewed before enabling
```

## During the wave

```text
[ ] do not snapshot manually
[ ] do not modify global crop JSON
[ ] do not restore Qdrant
[ ] do not start another worker for same state
[ ] monitor worker progress
[ ] resume failed states rather than rolling back
```

## Before finalization

```text
[ ] every EXPECTED_STATE completed
[ ] state manifests exist
[ ] coordinator shows COMPLETE
[ ] no workers in the wave are still writing
[ ] next wave has NOT started
```

## Finalizer

```text
[ ] BUILD_ID correct
[ ] WAVE_ID correct
[ ] EXPECTED_STATES exact
[ ] service URLs correct
[ ] RUN_FINALIZATION=False during validation
[ ] all validation cells pass
[ ] set RUN_FINALIZATION=True
[ ] global crop backup created
[ ] global crop merge succeeds
[ ] Qdrant snapshot succeeds
[ ] snapshot downloaded if enabled
[ ] wave_manifest.json written
```

## After finalization

```text
[ ] wave manifest status=complete
[ ] global crop JSON hash recorded
[ ] cumulative snapshot recorded
[ ] safe to start next wave
```

---

# 59. What Not to Do

Do not:

```text
❌ start one Qdrant server per worker
❌ point multiple Qdrant servers at the same storage
❌ let workers snapshot the shared collection
❌ restore snapshots while workers are active
❌ let two workers process the same state
❌ share one state pipeline_state.db across states
❌ edit global crop_occurrences.json from workers
❌ overlap wave N+1 with finalization of wave N
❌ change inputs during a resume run
❌ use a debug source limit on a real production state
❌ embed Hugging Face access tokens into committed documentation
```

---

# 60. Bottom Line

A complete production wave now looks like:

```text
                         CPU SERVICE NODE
                  ┌────────────────────────┐
                  │ Qdrant :6333           │
                  │ Coordinator :8001      │
                  └───────────┬────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
     Worker 1              Worker 2              Worker N
     1 GPU                 1 GPU                 1 GPU
     STATE A               STATE B               STATE N
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ↓
                    all expected states
                         COMPLETE
                              ↓
                    finalize_wave.ipynb
                              ↓
            global crop merge + Qdrant snapshot
                              ↓
                     wave_manifest.json
                              ↓
                         next wave
```

The key operational rule is:

> **Workers are concurrent within a wave; global finalization is serial between waves.**

This preserves crash recovery, global deduplication, state-local persistence, cumulative crop state, and cumulative Qdrant checkpoints while allowing independent 1-GPU notebook allocations to build MetaMIRAGE in parallel.

---

# 61. Startup Sequence: `build_2026_08` / `wave_01`

```bash
cd "/path/to/Mirage Database"
hostname -f
```

```bash
cd "/path/to/Mirage Database"
export QDRANT__STORAGE__STORAGE_PATH="$(pwd)/qdrant/storage"
./qdrant/bin/qdrant
```

```bash
cd "/path/to/Mirage Database"
source ~/metamirage-coordinator-venv/bin/activate
mkdir -p service
export METAMIRAGE_COORDINATOR_DB="$(pwd)/service/preload_coordinator.db"
export METAMIRAGE_COORDINATOR_HOST=0.0.0.0
export METAMIRAGE_COORDINATOR_PORT=8001
python preload_coordinator.py
```

```bash
curl http://127.0.0.1:6333/collections
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:6333/collections/mirage_base_build
```

```bash
curl -X PUT \
  "http://127.0.0.1:6333/collections/mirage_base_build" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'
```

```python
QDRANT_URL = "http://<SERVICE_HOSTNAME>:6333"
COORDINATOR_URL = "http://<SERVICE_HOSTNAME>:8001"
QDRANT_COLLECTION = "mirage_base_build"
BUILD_ID = "build_2026_08"
WAVE_ID = "wave_01"
RUN_MODE = "resume"
DEBUG_SOURCE_LIMIT = None
RUN_PIPELINE = False
```

```bash
mkdir -p workers/worker_1/input
mkdir -p workers/worker_2/input
```

```python
# worker_1
STATE_NAME = "Indiana"
STATE_CODE = "IN"
RUN_PIPELINE = True
```

```python
# worker_2
STATE_NAME = "New York"
STATE_CODE = "NY"
RUN_PIPELINE = True
```

```bash
curl \
  "http://<SERVICE_HOSTNAME>:8001/state/status?build_id=build_2026_08&state_code=IN"

curl \
  "http://<SERVICE_HOSTNAME>:8001/state/status?build_id=build_2026_08&state_code=NY"

curl \
  "http://<SERVICE_HOSTNAME>:8001/wave/status?build_id=build_2026_08&wave_id=wave_01"
```

```python
BUILD_ID = "build_2026_08"
WAVE_ID = "wave_01"
EXPECTED_STATES = ["IN", "NY"]
QDRANT_URL = "http://<SERVICE_HOSTNAME>:6333"
COORDINATOR_URL = "http://<SERVICE_HOSTNAME>:8001"
QDRANT_COLLECTION = "mirage_base_build"
DOWNLOAD_SNAPSHOT = True
RUN_FINALIZATION = False
RUN_FINALIZATION = True
```
