# MetaMIRAGE Preload — Run Guide

This guide explains how to set up and run the complete MetaMIRAGE cumulative Qdrant preload pipeline.

> **Important:** this file contains a Hugging Face access token below because it was requested for convenience. Treat this file as **local/private**. Do not commit it to a public repository. If the repository is shared, remove the token first or add `run.md` to `.gitignore`.

---

# 1. Expected Repository / Working Directory Layout

Keep the complete preload setup in one directory. The directory can be moved anywhere because the notebook uses relative paths through:

```python
BASE_DIR = Path.cwd().resolve()
```

Recommended layout:

```text
Database/
├── MetaMIRAGE_Cumulative_Qdrant_Preload_CONFIRMED.ipynb
├── run.md
│
├── county_state_hardiness_zone.csv
├── crop_occurrences.json
│
├── Illinois-PDFS.zip          # optional, one *PDF*.zip at a time
├── Illinois-CSV.zip           # optional, one *CSV*.zip at a time
├── Illinois-URL.xlsx          # optional, one *URL* input at a time
│
├── qdrant/
│   ├── bin/
│   │   └── qdrant
│   └── storage/
│
├── canonical/                 # created/used automatically
├── runs/                      # created/used automatically
├── pipeline_state.db          # created automatically
├── pipeline_state.db-shm      # SQLite runtime file; may appear
├── pipeline_state.db-wal      # SQLite runtime file; may appear
└── .python_packages/          # notebook-local packages
```

You do **not** need PDF, CSV, and URL inputs at the same time. A state run may use any combination of them, but at least one source type must be present.

---

# 2. Input File Naming Rules

The notebook auto-discovers files directly under `BASE_DIR`.

The matching is case-insensitive.

## PDF input

There must be at most one ZIP file whose filename contains `PDF`.

Examples:

```text
Illinois-PDFS.zip
IL_PDF_Data.zip
PDF_documents.zip
```

The notebook automatically:

```text
detects the ZIP
→ extracts it
→ recursively finds all .pdf files
→ processes them
```

The PDF ZIP may contain nested folders.

---

## CSV input

There must be at most one ZIP file whose filename contains `CSV`.

Examples:

```text
Illinois-CSV.zip
IL_CSV_Data.zip
CSV_sources.zip
```

The notebook automatically:

```text
detects the ZIP
→ extracts it
→ recursively finds all .csv files
→ registers every CSV
```

Each CSV row becomes one canonical document.

---

## URL input

There must be at most one URL input file whose filename contains `URL`.

Supported formats:

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

The notebook extracts HTTP/HTTPS URLs from the file and removes duplicates.

---

# 3. Required Support Files

These two files must remain directly under `BASE_DIR`:

```text
county_state_hardiness_zone.csv
crop_occurrences.json
```

The notebook expects:

```python
HARDINESS_CSV = BASE_DIR / "county_state_hardiness_zone.csv"
CROP_OCCURRENCE_JSON = BASE_DIR / "crop_occurrences.json"
```

## `county_state_hardiness_zone.csv`

Used to construct the hardiness-zone lookup:

```text
(county, state) -> hardiness_zone
state -> modal hardiness_zone
```

For normal state-level preload runs:

```text
STATE_NAME
→ location
→ state modal hardiness zone
```

If county data exists, the notebook tries the county-specific zone first.

Alaska and Hawaii are allowed to use:

```text
hardiness_zone = ""
```

as the documented exception.

---

## `crop_occurrences.json`

This is one cumulative file for all states.

It is initialized from the county crop-frequency source using the separate skeleton-builder script.

Example structure:

```json
{
  "illinois": {
    "corn": {
      "occurrence": 1234,
      "disease": {},
      "pests": {},
      "management": {}
    }
  }
}
```

Each successful state run updates only that state's section.

The notebook does **not** create a new crop dictionary file for every state.

---

# 4. First-Time Qdrant Setup

The Qdrant Python client is not the Qdrant server. The server runs as a separate process.

For this pipeline, use the Qdrant Linux musl binary.

From the repository directory:

```bash
cd /u/ssingh38/Database
```

Create the Qdrant directories:

```bash
mkdir -p qdrant/bin
mkdir -p qdrant/storage
```

Download Qdrant 1.18.0:

```bash
cd qdrant/bin

wget https://github.com/qdrant/qdrant/releases/download/v1.18.0/qdrant-x86_64-unknown-linux-musl.tar.gz
```

Extract it:

```bash
tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz
```

Make the binary executable:

```bash
chmod +x qdrant
```

Check the binary:

```bash
./qdrant --version
```

Then return to the repository root:

```bash
cd ../..
```

This setup only needs to be done once unless the Qdrant binary is removed.

---

# 5. Start the Qdrant Server

Qdrant must run on the **same compute node** as the Jupyter notebook.

Open a terminal in the same Jupyter/compute session.

From the repository directory:

```bash
cd /u/ssingh38/Database
```

Set the storage path:

```bash
export QDRANT__STORAGE__STORAGE_PATH="$(pwd)/qdrant/storage"
```

Start Qdrant:

```bash
./qdrant/bin/qdrant
```

Leave this terminal running.

The server should expose:

```text
REST: 127.0.0.1:6333
gRPC: 127.0.0.1:6334
```

The notebook uses:

```python
QDRANT_URL = "http://127.0.0.1:6333"
```

---

# 6. Test Qdrant with `curl`

Open a **second terminal** on the same compute node.

Run:

```bash
curl http://127.0.0.1:6333/collections
```

A working Qdrant server should return JSON similar to:

```json
{
  "result": {
    "collections": []
  },
  "status": "ok",
  "time": 0.0001
}
```

The exact timing fields may differ.

If collections already exist, they will appear in the list.

You can also inspect the preload collection directly once it exists:

```bash
curl http://127.0.0.1:6333/collections/mirage_base_build
```

---

# 7. Optional: Clear the Qdrant Preload Collection

To delete only the cumulative preload collection:

```bash
curl -X DELETE \
  "http://127.0.0.1:6333/collections/mirage_base_build"
```

Then verify:

```bash
curl http://127.0.0.1:6333/collections
```

> Deleting the Qdrant collection does **not** reset the SQLite ledger, canonical store, run manifests, or `crop_occurrences.json`.

For a completely fresh build, the Qdrant collection and persisted pipeline state must be reset consistently.

---

# 8. Hugging Face Token

The qualification classifier uses:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

Use this Hugging Face token:

```text
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
```

## Reliable method inside the notebook

Because exporting a variable in a separate terminal does not necessarily modify the environment of an already-running Jupyter kernel, the most reliable method is to run this in a notebook cell before qualification starts:

```python
import os

os.environ["HF_TOKEN"] = "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"
HF_TOKEN = os.environ["HF_TOKEN"]
```

Verify without printing the secret itself:

```python
print("HF token present:", bool(HF_TOKEN))
```

Expected:

```text
HF token present: True
```

The notebook can also prompt for the token when it first loads the classifier if `HF_TOKEN` is empty.

## Terminal method

If the Jupyter process itself is started from a shell where the variable is already present:

```bash
export HF_TOKEN="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"
```

then the notebook may inherit it automatically.

---

# 9. Open the Notebook

Open:

```text
MetaMIRAGE_Cumulative_Qdrant_Preload_CONFIRMED.ipynb
```

from the same working directory containing the inputs and support files.

Run the notebook from the top.

The dependency setup cell installs notebook-local Python packages under:

```text
.python_packages/
```

The notebook adds that directory to `sys.path`.

---

# 10. Configure the Current State

The state is configured explicitly.

For the first Illinois run:

```python
BUILD_ID = "build_2026_08"

STATE_NAME = "Illinois"
STATE_CODE = "IL"
RUN_SEQUENCE = 1

RUN_ID = f"{RUN_SEQUENCE:03d}_{STATE_CODE.upper()}"
```

This gives:

```text
RUN_ID = 001_IL
```

The state configuration controls:

```text
run identity
location metadata
hardiness-zone derivation
crop_occurrences.json state section
manifest state identity
run output directory
```

The state is **not** inferred from the PDF/CSV/URL filename.

---

# 11. Important Run-Switch Behavior

The notebook has:

```python
RUN_PIPELINE = False
```

by default so simply opening/rerunning the notebook cannot launch a large build.

The correct execution order is:

```text
configuration
→ input discovery/setup
→ preflight/status
→ enable RUN_PIPELINE
→ run pipeline
```

## Important

Do **not** rerun a configuration cell that resets:

```python
PDF_ZIP_FILE = None
CSV_ZIP_FILE = None
URL_FILE = None
```

after the input-discovery cell has already populated them.

The safest approach is to enable execution in a tiny separate cell:

```python
RUN_PIPELINE = True
```

Then immediately verify:

```python
print("RUN_PIPELINE:", RUN_PIPELINE)
print("PDF_ZIP_FILE:", PDF_ZIP_FILE)
print("CSV_ZIP_FILE:", CSV_ZIP_FILE)
print("URL_FILE:", URL_FILE)
print("CSV_INPUTS:", CSV_INPUTS)
```

For a PDF-only Illinois run, an expected result is:

```text
RUN_PIPELINE: True
PDF_ZIP_FILE: /.../Database/Illinois-PDFS.zip
CSV_ZIP_FILE: None
URL_FILE: None
CSV_INPUTS: []
```

---

# 12. Verify Input Auto-Discovery Before Running

Before starting the expensive pipeline, verify the discovered inputs.

Example diagnostic:

```python
from pathlib import Path

print("BASE_DIR =", BASE_DIR)
print("Path.cwd() =", Path.cwd())

print("\nFiles:")
for p in BASE_DIR.iterdir():
    print(" ", p.name)

print("\nCurrent inputs:")
print("URL_FILE =", URL_FILE)
print("PDF_ZIP_FILE =", PDF_ZIP_FILE)
print("CSV_ZIP_FILE =", CSV_ZIP_FILE)
print("CSV_INPUTS =", CSV_INPUTS)
```

At least one of the following must be populated:

```text
URL_FILE
PDF_ZIP_FILE / PDF_DIR
CSV_INPUTS
```

Otherwise preflight intentionally fails with:

```text
RuntimeError: No input sources are configured for this state run.
```

---

# 13. Recommended First Smoke Test

Before processing an entire state, use a small source limit.

In the configuration:

```python
DEBUG_SOURCE_LIMIT = 5
```

Then:

```python
RUN_PIPELINE = True
```

Run the pipeline.

This tests the complete path:

```text
source discovery
→ extraction
→ canonical persistence
→ qualification
→ crop enrichment
→ RAG chunking
→ metadata enrichment
→ metadata validation
→ embedding
→ Qdrant
→ validation
→ snapshot
→ manifest
```

For a clean smoke test, use a separate debug build/collection rather than marking the real state run complete.

Example:

```python
BUILD_ID = "debug_build"
QDRANT_COLLECTION = "mirage_debug"
DEBUG_SOURCE_LIMIT = 5
```

After the smoke test, return to:

```python
BUILD_ID = "build_2026_08"
QDRANT_COLLECTION = "mirage_base_build"
DEBUG_SOURCE_LIMIT = None
```

for the real cumulative build.

---

# 14. Start the Real Pipeline

After verifying:

- Qdrant is running.
- `curl` works.
- State configuration is correct.
- Input discovery is correct.
- `crop_occurrences.json` exists.
- hardiness mapping exists.
- HF token is available.

Run:

```python
RUN_PIPELINE = True
```

Then execute the final run cell:

```python
print_run_status()

if RUN_PIPELINE:
    manifest = run_pipeline()
else:
    print(
        "\nRUN_PIPELINE is False. Review the configuration cell, "
        "set RUN_PIPELINE = True, and rerun this cell when ready."
    )
```

---

# 15. What Happens During a State Run

The high-level execution path is:

```text
Source Discovery
      ↓
Extraction + Normalization
      ↓
Canonical Store
      ↓
Global Deduplication
      ↓
Qualification
      ├── Crop Dictionary Enrichment
      └── Accept / Reject
             ↓
        RAG Chunking
             ↓
Metadata Enrichment
             ↓
Metadata Contract Validation
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
Cumulative Qdrant Snapshot
             ↓
Manifest
             ↓
Atomic crop_occurrences.json update
```

---

# 16. Persistent State Created Automatically

During/after the run, the directory will gain:

```text
pipeline_state.db
canonical/
runs/
```

Example:

```text
Database/
├── pipeline_state.db
├── canonical/
│   ├── ab/
│   ├── bc/
│   └── ...
│
└── runs/
    └── build_2026_08/
        └── 001_IL/
            ├── mirage_base_001_IL.snapshot
            └── manifest.json
```

The root-level:

```text
crop_occurrences.json
```

is also updated for the completed state.

---

# 17. Resume After an Interrupted Run

The pipeline is designed to resume.

Do **not** delete:

```text
pipeline_state.db
canonical/
runs/
```

after an interrupted run.

Restart Qdrant, reopen the notebook, reconfigure the same:

```text
BUILD_ID
STATE_NAME
STATE_CODE
RUN_SEQUENCE
```

then rerun setup and execution.

SQLite status, canonical documents, content hashes, deterministic chunk IDs, and Qdrant IDs prevent healthy completed work from being repeated.

Failures are retried within the same run.

---

# 18. Running the Next State

After Illinois completes, prepare the next state's input files.

Remove/move the previous state's source ZIP/file so there is still only one matching file of each source type.

Example next run:

```python
STATE_NAME = "Indiana"
STATE_CODE = "IN"
RUN_SEQUENCE = 2
```

which gives:

```text
RUN_ID = 002_IN
```

If the live cumulative Qdrant collection still exists and is healthy, continue using it.

If using a new allocation or the live collection is absent/inconsistent, set:

```python
PREVIOUS_RUN_DIR = (
    BASE_DIR
    / "runs"
    / "build_2026_08"
    / "001_IL"
)
```

with:

```python
AUTO_RESTORE_PREVIOUS_SNAPSHOT = True
```

The notebook can then restore the prior cumulative snapshot before adding Indiana.

---

# 19. Qdrant Startup on Every New Compute Session

The Qdrant process does not survive the end of a compute allocation.

On a new compute session:

```bash
cd /u/ssingh38/Database

export QDRANT__STORAGE__STORAGE_PATH="$(pwd)/qdrant/storage"

./qdrant/bin/qdrant
```

Keep that terminal running.

Then verify in a second terminal:

```bash
curl http://127.0.0.1:6333/collections
```

Only after this succeeds should the notebook pipeline be started.

---

# 20. Quick Start Checklist

For each state:

```text
[ ] Start a sufficiently long Jupyter/compute session
[ ] Open terminal #1
[ ] cd to the repository
[ ] export QDRANT__STORAGE__STORAGE_PATH
[ ] start ./qdrant/bin/qdrant
[ ] leave terminal #1 running
[ ] open terminal #2
[ ] curl http://127.0.0.1:6333/collections
[ ] verify Qdrant responds
[ ] verify crop_occurrences.json exists
[ ] verify county_state_hardiness_zone.csv exists
[ ] place current state's PDF ZIP / CSV ZIP / URL file in repository root
[ ] ensure at most one matching input of each type exists
[ ] open notebook
[ ] run notebook setup cells
[ ] configure STATE_NAME / STATE_CODE / RUN_SEQUENCE
[ ] verify input auto-discovery
[ ] make HF_TOKEN available
[ ] optionally run a small smoke test
[ ] set RUN_PIPELINE = True in a separate small cell
[ ] verify inputs were not reset
[ ] execute pipeline
[ ] wait for run completion
[ ] verify snapshot + manifest
[ ] verify crop_occurrences.json was updated
[ ] proceed to next state
```

---

# 21. Useful Commands

## Qdrant collections

```bash
curl http://127.0.0.1:6333/collections
```

## Inspect preload collection

```bash
curl http://127.0.0.1:6333/collections/mirage_base_build
```

## Delete preload collection

```bash
curl -X DELETE \
  "http://127.0.0.1:6333/collections/mirage_base_build"
```

## Check Qdrant version

```bash
./qdrant/bin/qdrant --version
```

## Check repository files

```bash
ls -lah
```

## Check the Qdrant process

```bash
ps aux | grep qdrant
```

---

# 22. Important Safety Notes

1. Do not run two state builds concurrently against the same SQLite ledger and Qdrant collection.
2. Do not delete `pipeline_state.db` during a normal resume.
3. Do not manually modify Qdrant payloads during a build.
4. Do not put more than one matching PDF ZIP, CSV ZIP, or URL file in the repository root.
5. Do not fabricate `month_year`; unknown dates remain `""`.
6. Do not remove the canonical store after a successful state.
7. Do not overwrite a completed run directory unless intentionally resetting the build.
8. Keep Qdrant running for the entire notebook run.
9. Keep the Jupyter compute allocation alive long enough for the state workload.
10. Treat the Hugging Face token in this guide as a secret and do not publish this file.
