# Qdrant Setup, Concurrent Jupyter Workers, Snapshots, and Reuse on NCSA Delta

## Purpose

This document captures the working setup verified on NCSA Delta for:

- Running a Qdrant server alongside a JupyterLab session.
- Connecting to Qdrant from a Jupyter notebook.
- Creating collections and storing vectors/payloads.
- Querying stored data.
- Creating a Qdrant snapshot.
- Downloading the snapshot as a portable artifact.
- Restoring/reusing that snapshot later.
- Using the restored collection from the main MetaMIRAGE pipeline.

Current working directory used during testing:

```text
/u/ssingh38/Database
```

The test collection used was:

```text
mirage_test
```

The test snapshot used was:

```text
/u/ssingh38/Database/mirage_test.snapshot
```

---

# 1. Architecture

Jupyter and Qdrant are separate processes running on the same Delta compute node.

```text
Delta Compute Node
│
├── JupyterLab / Notebook
│      │
│      │ Qdrant Python Client
│      ▼
│   http://127.0.0.1:6333
│
└── Qdrant Server
       │
       ▼
   Persistent Storage
```

Because both run on the same allocated node, the notebook can communicate with Qdrant using:

```text
http://127.0.0.1:6333
```

Qdrant is treated as a service. The notebook and later the MetaMIRAGE runtime are simply clients of that service.

---

# 2. Directory Layout

The test setup uses:

```text
/u/ssingh38/Database/
│
├── .python_packages/
│   ├── qdrant_client/
│   ├── portalocker/
│   └── ...
│
├── qdrant/
│   ├── bin/
│   │   └── qdrant
│   │
│   └── storage/
│
├── mirage_test.snapshot
│
└── mirage_pre_ingestion_jupyter.ipynb
```

---

# 3. Installing the Qdrant Server

From the JupyterLab terminal:

```bash
cd /u/ssingh38/Database

mkdir -p qdrant/bin
mkdir -p qdrant/storage

cd qdrant/bin
```

Download Qdrant:

```bash
wget https://github.com/qdrant/qdrant/releases/download/v1.18.0/qdrant-x86_64-unknown-linux-musl.tar.gz
```

Extract it:

```bash
tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz
```

Make it executable:

```bash
chmod +x qdrant
```

Verify:

```bash
./qdrant --version
```

Expected version:

```text
qdrant 1.18.0
```

---

# 4. Starting the Qdrant Server

From the JupyterLab terminal:

```bash
cd /u/ssingh38/Database

export QDRANT__STORAGE__STORAGE_PATH=/u/ssingh38/Database/qdrant/storage

./qdrant/bin/qdrant
```

Leave this terminal running while the notebook interacts with Qdrant.

The default HTTP endpoint is:

```text
http://127.0.0.1:6333
```

To verify the server from another terminal:

```bash
curl http://127.0.0.1:6333/collections
```

A fresh server should return an empty collection list.

---

# 5. Installing the Qdrant Python Client in Jupyter

The Delta-provided Python environment is read-only, so installing directly into its `site-packages` causes a permission error.

Instead, install packages into a user-owned directory:

```python
import sys
from pathlib import Path

pkg_dir = Path("/u/ssingh38/Database/.python_packages")
pkg_dir.mkdir(exist_ok=True)

!{sys.executable} -m pip install \
    --no-cache-dir \
    --target "{pkg_dir}" \
    qdrant-client==1.18.0

if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))
```

Then verify:

```python
from qdrant_client import QdrantClient

print("Qdrant client imported successfully")
```

---

# 6. Connecting from Jupyter to Qdrant

```python
from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"

client = QdrantClient(url=QDRANT_URL)

print(client.get_collections())
```

If this succeeds, the Jupyter kernel is communicating with the Qdrant server.

---

# 7. Creating a Test Collection

For the initial connectivity test, a 4-dimensional vector collection was used.

```python
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "mirage_test"

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=4,
            distance=Distance.COSINE,
        ),
    )

print(client.get_collection(COLLECTION_NAME))
```

---

# 8. Adding Test Data

```python
from qdrant_client.models import PointStruct

points = [
    PointStruct(
        id=1,
        vector=[0.9, 0.1, 0.0, 0.0],
        payload={
            "text": "Soybean rust is a fungal disease affecting soybean crops.",
            "source": "document_1",
            "accepted": True,
            "state": "Illinois",
        },
    ),
    PointStruct(
        id=2,
        vector=[0.1, 0.9, 0.0, 0.0],
        payload={
            "text": "Corn rootworm can cause significant root damage in maize.",
            "source": "document_2",
            "accepted": True,
            "state": "Illinois",
        },
    ),
    PointStruct(
        id=3,
        vector=[0.0, 0.1, 0.9, 0.0],
        payload={
            "text": "Tomato plants require adequate sunlight and irrigation.",
            "source": "document_3",
            "accepted": True,
            "state": "Indiana",
        },
    ),
    PointStruct(
        id=4,
        vector=[0.8, 0.2, 0.0, 0.0],
        payload={
            "text": "Soybean aphids can reduce soybean yield by feeding on plant sap.",
            "source": "document_4",
            "accepted": True,
            "state": "Illinois",
        },
    ),
    PointStruct(
        id=5,
        vector=[0.2, 0.8, 0.0, 0.0],
        payload={
            "text": "Northern corn leaf blight is a fungal disease that affects corn plants.",
            "source": "document_5",
            "accepted": True,
            "state": "Iowa",
        },
    ),
    PointStruct(
        id=6,
        vector=[0.0, 0.2, 0.8, 0.0],
        payload={
            "text": "Tomato late blight can cause dark lesions on leaves, stems, and fruit.",
            "source": "document_6",
            "accepted": True,
            "state": "Indiana",
        },
    ),
    PointStruct(
        id=7,
        vector=[0.0, 0.1, 0.2, 0.9],
        payload={
            "text": "Japanese beetles feed on the foliage of many agricultural and ornamental plants.",
            "source": "document_7",
            "accepted": True,
            "state": "Ohio",
        },
    ),
]

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print("Total points:", client.count(COLLECTION_NAME))
```

Expected count:

```text
7
```

---

# 9. Querying the Collection

```python
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=[0.85, 0.15, 0.0, 0.0],
    limit=3,
).points

for result in results:
    print("Score:", result.score)
    print("Text:", result.payload["text"])
    print()
```

This confirms that stored vectors can be retrieved by similarity.

---

# 10. Intended Real Preload Flow

The current database-preparation flow uses concurrent state workers:

```text
Raw PDF / URL / Document
        ↓
Extract and normalize once
        ↓
Run suitability / qualification logic
        ↓
accepted?
   ┌────┴────┐
   │         │
  No        Yes
   │         │
 Stop        ↓
         RAG chunking
             ↓
        Batch embedding
             ↓
        Qdrant upsert
        ↓
  Shared cumulative Qdrant collection

Several workers may execute this flow concurrently, one state per worker. The
workers use state-local ledgers and canonical stores, while the coordinator
serializes state leases and global content claims. Workers do not create or
restore snapshots. After all expected states in a wave complete, a separate
finalizer validates the wave, merges state-local crop JSON files, and creates
one cumulative snapshot.
```

A conceptual ingestion loop:

```python
for document in documents:

    extracted = extract_document(document)

    classification = classify_document(extracted)

    if not classification["accepted"]:
        continue

    chunks = create_rag_chunks(extracted)

    embeddings = embedding_model.encode(chunks)

    qdrant_points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        qdrant_points.append(
            PointStruct(
                id=...,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source": document["source"],
                    "state": document["state"],
                    "accepted": True,
                },
            )
        )

    client.upsert(
        collection_name="mirage_base_v1",
        points=qdrant_points,
    )
```

The key architectural goal is that accepted documents should not need to be parsed and processed from raw form a second time before ingestion.

---

# 11. Creating a Snapshot

Once a collection is ready:

```python
COLLECTION_NAME = "mirage_test"

snapshot = client.create_snapshot(
    collection_name=COLLECTION_NAME
)

print(snapshot)
```

To list available snapshots:

```python
snapshots = client.list_snapshots(
    collection_name=COLLECTION_NAME
)

for s in snapshots:
    print(s.name, s.size, s.creation_time)
```

The returned snapshot object includes its generated name.

---

# 12. Downloading the Snapshot

The snapshot initially exists inside Qdrant's server-managed snapshot storage.

To copy it into the working directory, use the snapshot download endpoint.

Example:

```bash
cd /u/ssingh38/Database

curl \
  "http://127.0.0.1:6333/collections/mirage_test/snapshots/<SNAPSHOT_NAME>" \
  --output mirage_test.snapshot
```

Afterward:

```text
/u/ssingh38/Database/mirage_test.snapshot
```

is the portable snapshot artifact.

Verify:

```bash
ls -lh /u/ssingh38/Database/mirage_test.snapshot
```

---

# 13. Creating and Downloading a Snapshot Entirely from Jupyter

```python
import requests
from pathlib import Path
from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "mirage_test"
OUTPUT_PATH = Path("/u/ssingh38/Database/mirage_test.snapshot")

client = QdrantClient(url=QDRANT_URL)

snapshot = client.create_snapshot(
    collection_name=COLLECTION_NAME
)

snapshot_name = snapshot.name

print("Created:", snapshot_name)

url = (
    f"{QDRANT_URL}/collections/"
    f"{COLLECTION_NAME}/snapshots/{snapshot_name}"
)

with requests.get(url, stream=True) as response:
    response.raise_for_status()

    with open(OUTPUT_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

print("Saved to:", OUTPUT_PATH)
```

---

# 14. Reusing a Snapshot

A snapshot is reused by restoring it into a running Qdrant server.

Assumptions:

```text
Snapshot:
/u/ssingh38/Database/mirage_test.snapshot

Collection:
mirage_test

Server:
http://127.0.0.1:6333
```

First start Qdrant:

```bash
cd /u/ssingh38/Database

export QDRANT__STORAGE__STORAGE_PATH=/u/ssingh38/Database/qdrant/storage

./qdrant/bin/qdrant
```

Then restore the snapshot from another terminal:

```bash
curl -X POST \
  "http://127.0.0.1:6333/collections/mirage_test/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@/u/ssingh38/Database/mirage_test.snapshot"
```

If the collection does not exist, Qdrant restores it from the snapshot.

---

# 15. Clean Snapshot Restore

If `mirage_test` already exists and a completely clean restore is desired, delete it first:

```bash
curl -X DELETE \
  "http://127.0.0.1:6333/collections/mirage_test"
```

Then restore:

```bash
curl -X POST \
  "http://127.0.0.1:6333/collections/mirage_test/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@/u/ssingh38/Database/mirage_test.snapshot"
```

Full sequence:

```bash
# Check server
curl http://127.0.0.1:6333/collections

# Remove old collection if present
curl -X DELETE \
  "http://127.0.0.1:6333/collections/mirage_test"

# Restore snapshot
curl -X POST \
  "http://127.0.0.1:6333/collections/mirage_test/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@/u/ssingh38/Database/mirage_test.snapshot"

# Verify restored collection
curl http://127.0.0.1:6333/collections/mirage_test
```

---

# 16. Verifying a Restored Snapshot from Jupyter

```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://127.0.0.1:6333"
)

print(client.get_collection("mirage_test"))

print(
    "Points:",
    client.count("mirage_test").count
)
```

For the test snapshot used during verification, the expected count was:

```text
Points: 7
```

This restore flow was verified successfully.

---

# 17. What Happens After Restore

Once the snapshot has been restored, the application does not interact with the `.snapshot` file directly anymore.

The restored collection behaves like any normal Qdrant collection:

```python
client = QdrantClient(
    url="http://127.0.0.1:6333"
)

COLLECTION_NAME = "mirage_test"
```

The MetaMIRAGE pipeline can then query that collection normally.

Conceptually:

```text
mirage_test.snapshot
        ↓
restore once
        ↓
Qdrant Server
        ↓
mirage_test collection
        ↓
MainAgent / RAG retrieval
```

---

# 18. Reusing the Snapshot Across Delta Jobs

The compute node can change between Delta allocations.

Example:

```text
Run 1:
gpua053
```

The collection is built and saved as:

```text
/u/ssingh38/Database/mirage_test.snapshot
```

The interactive job ends.

Later:

```text
Run 2:
gpua087
```

A new Qdrant process is started on the new node:

```bash
cd /u/ssingh38/Database

export QDRANT__STORAGE__STORAGE_PATH=/u/ssingh38/Database/qdrant/storage

./qdrant/bin/qdrant
```

Then restore:

```bash
curl -X POST \
  "http://127.0.0.1:6333/collections/mirage_test/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@/u/ssingh38/Database/mirage_test.snapshot"
```

The snapshot therefore makes the prepared collection portable across Delta compute allocations.

---

# 19. Intended MetaMIRAGE Database Lifecycle

For the final system, the intended pattern is:

```text
OFFLINE PREPARATION

100s GB raw corpus
        ↓
extract / normalize
        ↓
qualify documents
        ↓
accepted documents only
        ↓
RAG chunking
        ↓
embedding
        ↓
Qdrant base collection
        ↓
validation
        ↓
snapshot
        ↓
mirage_base_v1.snapshot
```

Then during inference:

```text
INFERENCE JOB

start Qdrant
        ↓
restore mirage_base_v1.snapshot
        ↓
MainAgent connects to Qdrant
        ↓
retrieve preloaded evidence
        ↓
confidence evaluation
        ↓
if sufficient → answer
if insufficient → web augmentation
```

The base snapshot allows every experimental run to start from the same prepared corpus without rebuilding or re-embedding the database.

---

# 20. Recommended Base vs Runtime Collection Design

For reproducibility, the final architecture should ideally keep the preloaded corpus separate from runtime web-search additions.

```text
Qdrant
│
├── mirage_base_v1
│   └── immutable prepared corpus
│
└── mirage_runtime_aug
    └── mutable web-search additions
```

Retrieval can search both collections and merge/rank the results.

Benefits:

- The base dataset never changes during inference.
- Runtime web ingestion cannot contaminate later experimental runs.
- `mirage_runtime_aug` can be reset between runs.
- Every experiment can restore the same `mirage_base_v1.snapshot`.
- Web-search frequency can be measured against a stable preload baseline.

---

# 21. Important Distinction: Process vs Storage vs Snapshot

These are three separate things:

```text
Qdrant process
    temporary service running on a compute node

Qdrant storage
    live server persistence directory

Qdrant snapshot
    portable versioned collection artifact
```

On Delta:

```text
Interactive allocation ends
        ↓
Qdrant process stops
```

but the snapshot stored at:

```text
/u/ssingh38/Database/mirage_test.snapshot
```

remains available and can later be restored into a fresh Qdrant instance.

---

# 22. Minimal Day-to-Day Commands

## Start Qdrant

```bash
cd /u/ssingh38/Database

export QDRANT__STORAGE__STORAGE_PATH=/u/ssingh38/Database/qdrant/storage

./qdrant/bin/qdrant
```

## Check Qdrant

```bash
curl http://127.0.0.1:6333/collections
```

## Restore the test snapshot

```bash
curl -X POST \
  "http://127.0.0.1:6333/collections/mirage_test/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@/u/ssingh38/Database/mirage_test.snapshot"
```

## Connect from Jupyter

```python
from qdrant_client import QdrantClient

client = QdrantClient(
    url="http://127.0.0.1:6333"
)

print(client.count("mirage_test"))
```

---

# 23. Current Verified Status

The following operations have been tested successfully on NCSA Delta:

- JupyterLab terminal access.
- Starting a standalone Qdrant server.
- Connecting to the server from a Jupyter notebook.
- Installing `qdrant-client` into a user-owned Python package directory.
- Creating a Qdrant collection.
- Upserting vectors and payloads.
- Querying vectors.
- Creating a Qdrant collection snapshot.
- Saving the snapshot as a portable `.snapshot` file.
- Deleting the original collection.
- Restoring the collection from the snapshot.
- Querying/counting the restored collection successfully.

This establishes the core infrastructure required for the redesigned MetaMIRAGE preload pipeline.
