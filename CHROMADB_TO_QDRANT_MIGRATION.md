# Migrating MIRAGE-RAG from ChromaDB to Qdrant

This document is a comprehensive, file-by-file migration guide for replacing ChromaDB with Qdrant as the vector database backend in the MIRAGE-RAG project.  
**No code is changed here** — this is a reference guide only.

---

## Table of Contents

1. [Overview of the Change](#1-overview-of-the-change)
2. [Conceptual Differences](#2-conceptual-differences)
3. [Installation](#3-installation)
4. [File-by-File Changes](#4-file-by-file-changes)
   - [4.1 `rag_agent/utils/Embedding.py`](#41-rag_agentutilsembeddingpy)
   - [4.2 `rag_agent/utils/ContentUtils.py`](#42-rag_agentutilscontentutilspy)
   - [4.3 `rag_agent/tools/web_addition.py`](#43-rag_agenttoolsweb_additionpy)
   - [4.4 `rag_agent/tools/pdf_addition.py`](#44-rag_agenttoolspdf_additionpy)
   - [4.5 `rag_agent/tools/confidence_evaluator.py`](#45-rag_agenttoolsconfidence_evaluatorpy)
   - [4.6 `rag_agent/main.py`](#46-rag_agentmainpy)
   - [4.7 `preload_pipeline/preload/pipeline/chunk_upsert.py`](#47-preload_pipelinepreloadpipelinechunk_upsertpy)
   - [4.8 `preload_pipeline/preload/rag_agent_integration.py`](#48-preload_pipelinepreloadrag_agent_integrationpy)
   - [4.9 `preload_pipeline/bootstrap.py`](#49-preload_pipelinebootstrappy)
5. [Cross-Cutting API Mapping](#5-cross-cutting-api-mapping)
6. [Metadata Filtering Syntax Translation](#6-metadata-filtering-syntax-translation)
7. [Distance Score Semantics](#7-distance-score-semantics)
8. [Persistence & Deployment Modes](#8-persistence--deployment-modes)
9. [Data Migration (Existing Database)](#9-data-migration-existing-database)
10. [Testing Checklist](#10-testing-checklist)

---

## 1. Overview of the Change

ChromaDB is used in **nine places** across the codebase:

| File | Role |
|---|---|
| `rag_agent/utils/Embedding.py` | Defines a Chroma-aware embedding function |
| `rag_agent/utils/ContentUtils.py` | Deduplication check + query execution against Chroma collection |
| `rag_agent/tools/web_addition.py` | Writes chunks to Chroma collection |
| `rag_agent/tools/pdf_addition.py` | Writes chunks to Chroma collection |
| `rag_agent/tools/confidence_evaluator.py` | Reads from Chroma via `ContentUtils` |
| `rag_agent/main.py` | Creates/loads Chroma client + collection; exposes `count()` |
| `preload_pipeline/preload/pipeline/chunk_upsert.py` | Stand-alone Chroma upserter used by preload |
| `preload_pipeline/preload/rag_agent_integration.py` | Mirrors rag_agent Chroma setup for preload pipeline |
| `preload_pipeline/bootstrap.py` | CLI args reference "Chroma" in help text; drives the preload pipeline |

---

## 2. Conceptual Differences

Understanding these differences is essential before writing any code.

### 2.1 Embedding ownership

| Aspect | ChromaDB | Qdrant |
|---|---|---|
| Embedding computed by | Chroma (you pass an `EmbeddingFunction`) | You (compute vectors yourself, pass as `float[]`) |
| At query time | `query_texts=[...]` — Chroma embeds them | `query_vector=[...]` — you embed first |
| At ingest time | `collection.add(documents=[...])` — Chroma embeds them | `client.upsert(points=[PointStruct(vector=...)])` — you embed first |

This is the biggest architectural shift. In every place you call `collection.add(documents=...)` or `collection.query(query_texts=...)`, you must first embed the text yourself using your `SentenceTransformerEmbeddingFunction` (or equivalent), then pass the resulting float list.

### 2.2 Payload vs metadata

| ChromaDB | Qdrant |
|---|---|
| `metadatas=[{...}]` in `add()` | `payload={...}` in `PointStruct` |
| Retrieved via `results["metadatas"]` | Retrieved via `results[i].payload` |

### 2.3 Documents (text) storage

| ChromaDB | Qdrant |
|---|---|
| First-class `documents` field | Stored inside `payload`, e.g. `payload={"text": "...", ...}` |
| Retrieved via `results["documents"]` | Retrieved via `results[i].payload["text"]` |

### 2.4 Point IDs

| ChromaDB | Qdrant |
|---|---|
| String IDs (e.g. `"src_p1_c0"`) | Must be **unsigned integer** or a valid **UUID string** |
| IDs can be arbitrary strings | Arbitrary strings are not supported as IDs |

You will need to convert your string IDs (e.g. SHA-256 hex, `source_p1_c0`) to either UUIDs or deterministic integers. The easiest approach is `uuid.uuid5(uuid.NAMESPACE_DNS, your_string_id)`.

### 2.5 Collection schema at creation time

ChromaDB creates collections without specifying a vector dimension. Qdrant requires you to declare:
- `size`: vector dimension (e.g., 768 for `BAAI/bge-base-en-v1.5`)
- `distance`: `Distance.COSINE` | `Distance.DOT` | `Distance.EUCLID`

### 2.6 Query return type

| ChromaDB | Qdrant |
|---|---|
| Returns a dict with `"documents"`, `"metadatas"`, `"distances"` keys | Returns a list of `ScoredPoint` objects |
| `results["documents"][0]` — list of strings | `results[i].payload["text"]` |
| `results["distances"][0]` — list of floats (lower = more similar for L2; for cosine, Chroma returns `1 - cosine_similarity`) | `results[i].score` — higher is better (for Cosine/Dot) |

---

## 3. Installation

Remove `chromadb` from your environment and install `qdrant-client`:

```bash
pip uninstall chromadb
pip install qdrant-client
```

For a local persistent Qdrant instance (no server required), `qdrant-client >= 1.7` supports an embedded/local mode:

```bash
pip install "qdrant-client[local]>=1.7.0"
```

Update any `requirements.txt` / `environment.yml` accordingly.

---

## 4. File-by-File Changes

---

### 4.1 `rag_agent/utils/Embedding.py`

**Current state:**

```python
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None"):
        if device == "None":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, normalize_embeddings=True, convert_to_numpy=True).tolist()
```

**What needs to change:**

The class inherits from `chromadb.EmbeddingFunction` and uses Chroma-specific types (`Documents`, `Embeddings`). These must be removed. The underlying `SentenceTransformer` logic stays exactly the same — only the base class and imports change.

**New version:**

```python
import torch
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None"):
        if device == "None":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input, normalize_embeddings=True, convert_to_numpy=True).tolist()

    def embed_one(self, text: str) -> List[float]:
        """Convenience method: embed a single string and return a flat float list."""
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True).tolist()[0]
```

You should also expose the vector dimension so collection creation can reference it:

```python
    @property
    def vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

---

### 4.2 `rag_agent/utils/ContentUtils.py`

This file has two ChromaDB-dependent methods:

1. `content_hash_exists(collection, content_hash)` — uses `collection.get(where={...})`
2. `retrieve_with_priority_filters(...)` — uses `collection.query(query_texts=[...], where=..., include=[...])`

#### 4.2.1 `content_hash_exists`

**Current code (line 49–52):**

```python
@staticmethod
def content_hash_exists(collection, content_hash: str) -> bool:
    result = collection.get(where={"content_hash": content_hash})
    return len(result.get("ids", [])) > 0
```

**New version:**

Qdrant uses `scroll` to filter by payload fields. The `collection` parameter is now replaced with a `(qdrant_client, collection_name)` pair, or you can wrap them in a thin adapter object (see section 4.6 for how to pass both).

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

@staticmethod
def content_hash_exists(client: QdrantClient, collection_name: str, content_hash: str) -> bool:
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0
```

> **Signature change note:** Every call site that passes `collection` to `content_hash_exists` must now pass `client, collection_name` instead. This affects `web_addition.py`, `pdf_addition.py`, and indirectly `chunk_upsert.py`.

#### 4.2.2 `retrieve_with_priority_filters`

**Current code (lines 232–246):**

```python
query_args = {
    "query_texts": [query],
    "n_results": k,
    "include": ["documents", "metadatas", "distances"],
}
if where_filter is not None:
    query_args["where"] = where_filter

results = collection.query(**query_args)

docs = results.get("documents", [[]])[0]
metadatas = results.get("metadatas", [[]])[0]
distances = results.get("distances", [[]])[0]
```

**New version:**

You need an `embedding_function` accessible here to embed the query string. Add it as a constructor parameter to `ContentUtils`, or pass it explicitly.

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchRequest

# Inside retrieve_with_priority_filters:

# 1. Embed the query text first
query_vector = self.embedding_fn.embed_one(query)  # self.embedding_fn must be set in __init__

# 2. Build Qdrant filter from the existing where_filter dict (see helper below)
qdrant_filter = _chroma_where_to_qdrant_filter(where_filter) if where_filter else None

# 3. Run the search
hits = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    query_filter=qdrant_filter,
    limit=k,
    with_payload=True,
    with_vectors=False,
)

# 4. Unpack results
docs = [hit.payload.get("text", "") for hit in hits]
metadatas = [{k: v for k, v in hit.payload.items() if k != "text"} for hit in hits]
# Qdrant cosine score is 1.0 = identical; convert to "distance" (lower = better) to keep existing logic intact:
distances = [1.0 - hit.score for hit in hits]
```

**Filter translation helper** (see full details in [Section 6](#6-metadata-filtering-syntax-translation)):

```python
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def _chroma_where_to_qdrant_filter(where: dict) -> Filter:
    """Convert a Chroma where-clause dict to a Qdrant Filter."""
    if "$and" in where:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v["$eq"]))
            for clause in where["$and"]
            for k, v in clause.items()
        ]
        return Filter(must=conditions)
    else:
        # Single equality clause: {field: {"$eq": value}}
        for field, cond in where.items():
            return Filter(must=[FieldCondition(key=field, match=MatchValue(value=cond["$eq"]))])
```

**Distance score conversion (lines 249–251):**

The existing code converts Chroma distances to a higher-is-better score:

```python
# Current (Chroma):
similarity_scores.append(1.0 / (1.0 + max(float(distance), 0.0)))
```

With Qdrant cosine similarity, the score is already higher-is-better (range 0–1 for normalized vectors). You can either:

- Keep the existing formula unchanged (it still works since `distance = 1 - score`, so `1/(1 + (1-score))`)
- Or simplify to `similarity_scores.append(hit.score)` if you adapt the caller

The safest change is to keep the formula and just make sure `distance = 1.0 - hit.score`.

**`ContentUtils.__init__` changes:**

You must add `embedding_fn` and remove the Chroma-specific `EmbeddingFunction` dependency. Pass the embedding function in from `main.py` / `rag_agent_integration.py`:

```python
def __init__(
    self,
    embed_model: str = "BAAI/bge-base-en-v1.5",
    chunk_config: Dict | None = None,
    embedding_fn=None,        # NEW: required for query embedding in Qdrant
):
    self.embed_model = embed_model
    self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
    self.embedding_fn = embedding_fn  # NEW
    # ... rest unchanged
```

---

### 4.3 `rag_agent/tools/web_addition.py`

This file calls:
- `self.content_utils.content_hash_exists(self.collection, content_hash)` — must update call signature
- `self.collection.add(documents=docs, metadatas=metas, ids=ids)` — must replace with Qdrant upsert

#### Constructor

**Current:**
```python
def __init__(self, collection, content_utils, null_str="", null_int=-1):
    self.collection = collection
    ...
```

**New:**
```python
from qdrant_client import QdrantClient

def __init__(self, client: QdrantClient, collection_name: str, content_utils, null_str="", null_int=-1):
    self.client = client
    self.collection_name = collection_name
    self.content_utils = content_utils
    ...
```

#### Deduplication check (line 301–306 of `web_addition.py`)

**Current:**
```python
if self.content_utils.content_hash_exists(self.collection, content_hash):
    skipped += 1
    continue
```

**New:**
```python
if self.content_utils.content_hash_exists(self.client, self.collection_name, content_hash):
    skipped += 1
    continue
```

#### Document insertion (line 336–340 of `web_addition.py`)

**Current:**
```python
self.collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)
```

**New:**
```python
import uuid
from qdrant_client.http.models import PointStruct

points = []
for doc_id, doc_text, meta in zip(ids, documents, metadatas):
    # Convert string ID to UUID
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
    # Embed the document text
    vector = self.content_utils.embedding_fn.embed_one(doc_text)
    payload = {"text": doc_text, **meta}
    points.append(PointStruct(id=point_id, vector=vector, payload=payload))

self.client.upsert(collection_name=self.collection_name, points=points)
```

> **Note on document format:** The text is currently stored as `"Title: {title}\n\n{chunk}"`. This stays the same — it just moves into `payload["text"]` instead of a Chroma `documents` field. During retrieval in `ContentUtils`, you unpack it as `hit.payload.get("text", "")`.

---

### 4.4 `rag_agent/tools/pdf_addition.py`

Identical pattern to `web_addition.py`. The changes are:

1. Constructor: replace `collection` with `client: QdrantClient, collection_name: str`
2. Deduplication call: `content_hash_exists(self.client, self.collection_name, content_hash)`
3. Insertion: replace `self.collection.add(...)` with `self.client.upsert(...)` using `PointStruct` (embed each document first)

**Insertion block (line 243–247 of `pdf_addition.py`):**

```python
# Current:
self.collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)

# New:
import uuid
from qdrant_client.http.models import PointStruct

points = []
for doc_id, doc_text, meta in zip(ids, documents, metadatas):
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
    vector = self.content_utils.embedding_fn.embed_one(doc_text)
    payload = {"text": doc_text, **meta}
    points.append(PointStruct(id=point_id, vector=vector, payload=payload))

self.client.upsert(collection_name=self.collection_name, points=points)
```

---

### 4.5 `rag_agent/tools/confidence_evaluator.py`

This file does not call Chroma directly — it delegates everything to `ContentUtils.retrieve_with_priority_filters`. However, the constructor must change to match the new signature:

**Current:**
```python
def __init__(self, collection, content_utils):
    self.collection = collection
    self.content_utils = content_utils
```

**New:**
```python
def __init__(self, client: QdrantClient, collection_name: str, content_utils):
    self.client = client
    self.collection_name = collection_name
    self.content_utils = content_utils
```

Every call to `self.content_utils.retrieve_with_priority_filters(collection=self.collection, ...)` must change to:

```python
self.content_utils.retrieve_with_priority_filters(
    client=self.client,
    collection_name=self.collection_name,
    query=query,
    ...
)
```

The `retrieve_with_priority_filters` method signature in `ContentUtils` must correspondingly change from accepting `collection` to accepting `client` and `collection_name`.

**Distance/score conversion (lines 91–92 of `confidence_evaluator.py`):**

```python
# Current — distances from Chroma are (1 - cosine_similarity):
similarities = [1 - r["distance"] for r in results]

# New — after the change in ContentUtils where distance = 1.0 - hit.score:
# This line stays the same because distance is still 1 - score.
similarities = [1 - r["distance"] for r in results]
```

No change needed here if you keep `distance = 1.0 - hit.score` in `ContentUtils`.

---

### 4.6 `rag_agent/main.py`

This is the largest file to change. Every ChromaDB-specific call lives here.

#### Imports (line 1)

**Current:**
```python
import chromadb
```

**New:**
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
```

#### `__init__` — client and collection creation (lines 22–24)

**Current:**
```python
persist_path = "/work/nvme/bfox/ssingh38/chroma_database/chroma_db"
self.client = chromadb.PersistentClient(path=persist_path)
self.collection = self.client.get_or_create_collection(
    name="meta-mirage_collection",
    embedding_function=self.embedding_function
)
```

**New:**
```python
persist_path = "/work/nvme/bfox/ssingh38/qdrant_database"
self.collection_name = "meta-mirage_collection"
self.client = QdrantClient(path=persist_path)  # local persistent mode

# Create collection only if it doesn't exist yet
existing = [c.name for c in self.client.get_collections().collections]
if self.collection_name not in existing:
    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(
            size=self.embedding_function.vector_size,  # e.g. 768 for bge-base-en-v1.5
            distance=Distance.COSINE,
        ),
    )
```

> **Note on `embedding_function`:** The embedding function no longer needs to be passed to the Qdrant client — it is used directly in each tool. Keep `self.embedding_function` on the `MainAgent` instance and pass it to `ContentUtils` as well.

Also pass `embedding_fn` to `ContentUtils`:
```python
self.content_utils = ContentUtils(embed_model=embed_model_name, embedding_fn=self.embedding_function)
```

#### Tool instantiation (lines 36–40)

**Current:**
```python
self.pdf_addition = PDFAddition(self.collection, self.content_utils, self.null_str)
self.web_addition = WebAddition(self.collection, self.content_utils, self.null_str, self.null_int)
self.confidence_evaluator = ConfidenceEvaluator(self.collection, self.content_utils)
```

**New:**
```python
self.pdf_addition = PDFAddition(self.client, self.collection_name, self.content_utils, self.null_str)
self.web_addition = WebAddition(self.client, self.collection_name, self.content_utils, self.null_str, self.null_int)
self.confidence_evaluator = ConfidenceEvaluator(self.client, self.collection_name, self.content_utils)
```

#### `collection.count()` (lines 178, 261–265, 336)

**Current:**
```python
self.collection.count()
```

**New:**
```python
self.client.count(collection_name=self.collection_name).count
```

#### `list_collections()` (lines 27–31)

**Current:**
```python
collections = self.client.list_collections()
collection_names = [c.name if hasattr(c, "name") else str(c) for c in collections]
```

**New:**
```python
collections_response = self.client.get_collections()
collection_names = [c.name for c in collections_response.collections]
```

#### `reset_collection()` (lines 128–163)

**Current:**
```python
self.client.delete_collection(name=name)
self.collection = self.client.get_or_create_collection(
    name=name, embedding_function=self.embedding_function
)
```

**New:**
```python
# Delete if exists
try:
    self.client.delete_collection(collection_name=self.collection_name)
    print(f"[RAG reset_collection] Deleted collection: {self.collection_name}")
except Exception as e:
    print(f"[RAG reset_collection] Delete failed (continuing): {e}")

# Recreate
self.client.create_collection(
    collection_name=self.collection_name,
    vectors_config=VectorParams(
        size=self.embedding_function.vector_size,
        distance=Distance.COSINE,
    ),
)
print(f"[RAG reset_collection] Created collection: {self.collection_name}")

# Rebind tools
self.pdf_addition = PDFAddition(self.client, self.collection_name, self.content_utils, self.null_str)
self.web_addition = WebAddition(self.client, self.collection_name, self.content_utils, self.null_str, self.null_int)
self.confidence_evaluator = ConfidenceEvaluator(self.client, self.collection_name, self.content_utils)
```

#### `reload_existing_collection()` (lines 165–185)

**Current:**
```python
self.collection = self.client.get_collection(
    name=name, embedding_function=self.embedding_function
)
print(f"[RAG reload] Collection count: {self.collection.count()}", flush=True)
```

**New:**
```python
# Verify collection exists
existing = [c.name for c in self.client.get_collections().collections]
if self.collection_name not in existing:
    raise ValueError(f"Collection '{self.collection_name}' does not exist")

count = self.client.count(collection_name=self.collection_name).count
print(f"[RAG reload] Collection count: {count}", flush=True)

# Rebind tools
self.pdf_addition = PDFAddition(self.client, self.collection_name, self.content_utils, self.null_str)
self.web_addition = WebAddition(self.client, self.collection_name, self.content_utils, self.null_str, self.null_int)
self.confidence_evaluator = ConfidenceEvaluator(self.client, self.collection_name, self.content_utils)
```

#### `_tracked_retrieve_content` stale handle self-heal (lines 76–100)

**Current:**
```python
if "does not exist" in msg or "not exist" in msg:
    self.collection = self.client.get_or_create_collection(
        name="meta-mirage_collection",
        embedding_function=self.embedding_function,
    )
    self.pdf_addition = PDFAddition(self.collection, ...)
    self.web_addition = WebAddition(self.collection, ...)
    self.confidence_evaluator = ConfidenceEvaluator(self.collection, ...)
```

**New:**
```python
if "does not exist" in msg or "not exist" in msg:
    # Recreate if missing
    existing = [c.name for c in self.client.get_collections().collections]
    if self.collection_name not in existing:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_function.vector_size,
                distance=Distance.COSINE,
            ),
        )
    self.pdf_addition = PDFAddition(self.client, self.collection_name, ...)
    self.web_addition = WebAddition(self.client, self.collection_name, ...)
    self.confidence_evaluator = ConfidenceEvaluator(self.client, self.collection_name, ...)
```

#### `_tracked_add_web_content` — count before/after (lines 261–265)

**Current:**
```python
before = self.collection.count()
result = self.web_addition.add_web_content(...)
after = self.collection.count()
```

**New:**
```python
before = self.client.count(collection_name=self.collection_name).count
result = self.web_addition.add_web_content(...)
after = self.client.count(collection_name=self.collection_name).count
```

---

### 4.7 `preload_pipeline/preload/pipeline/chunk_upsert.py`

**Current state:**

The `ChromaUpserter` class wraps `chromadb.PersistentClient` and `collection.upsert(...)`.

**New version — rename to `QdrantUpserter`:**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from preload.utils.hashing import sha1_hex
from preload.transforms.normalize import split_into_chunks


@dataclass
class UpsertStats:
    chunks_created: int = 0
    chunks_upserted: int = 0
    chunks_failed: int = 0


class QdrantUpserter:
    """
    Standalone upserter that writes text chunks into a Qdrant persistent collection.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model_label: str,
        vector_size: int,           # NEW: required — e.g. 768 for bge-base-en-v1.5
        embedding_fn,               # NEW: callable that embeds List[str] -> List[List[float]]
        dry_run: bool = False,
        logger=None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_label = embedding_model_label
        self.vector_size = vector_size
        self.embedding_fn = embedding_fn
        self.dry_run = dry_run
        self.logger = logger

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.persist_dir))

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def chunk_and_upsert(self, text: str, metadata: Dict[str, Any], stable_id: Optional[str]) -> Dict[str, int]:
        chunks = split_into_chunks(text)
        stats = UpsertStats(chunks_created=len(chunks))

        points: List[PointStruct] = []

        for idx, chunk in enumerate(chunks):
            chunk_hash = sha1_hex(chunk)
            base = stable_id or metadata.get("url") or metadata.get("path") or metadata.get("source_name") or "unknown"
            chunk_id_str = sha1_hex(f"{base}::chunk{idx}::${chunk_hash}")
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id_str))

            m = dict(metadata)
            m["chunk_index"] = idx
            m["content_hash"] = chunk_hash
            m["embedding_model_label"] = self.embedding_model_label
            m["text"] = chunk   # Store text in payload

            if self.dry_run:
                stats.chunks_upserted += 1
                continue

            vector = self.embedding_fn([chunk])[0]
            points.append(PointStruct(id=point_id, vector=vector, payload=m))

        if self.dry_run:
            return {
                "chunks_created": stats.chunks_created,
                "chunks_upserted": stats.chunks_upserted,
                "chunks_failed": 0,
            }

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            stats.chunks_upserted = len(points)
        except Exception:
            stats.chunks_failed = len(points)
            if self.logger:
                self.logger.exception("Qdrant upsert failed.")

        return {
            "chunks_created": stats.chunks_created,
            "chunks_upserted": stats.chunks_upserted,
            "chunks_failed": stats.chunks_failed,
        }

    def close(self):
        self.client.close()
```

**Key differences from the Chroma version:**
- Class renamed from `ChromaUpserter` → `QdrantUpserter`
- `vector_size` and `embedding_fn` are new required constructor params
- Text is stored in `payload["text"]` instead of Chroma's documents field
- String IDs converted to UUIDs via `uuid.uuid5`
- `collection.upsert(documents=..., metadatas=..., ids=...)` → `client.upsert(points=[PointStruct(...)])`

---

### 4.8 `preload_pipeline/preload/rag_agent_integration.py`

**Current state:**

```python
import chromadb

def create_rag_agent_collection_and_utils(...):
    embedding_fn = SentenceTransformerEmbeddingFunction(embed_model, device)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)
    content_utils = ContentUtils(embed_model=embed_model)
    tool_collection = _DryRunCollection(collection, logger=logger) if dry_run else collection
    web_adder = WebAddition(collection=tool_collection, content_utils=content_utils)
    pdf_adder = PDFAddition(collection=tool_collection, content_utils=content_utils)
    return collection, content_utils, web_adder, pdf_adder
```

**New version:**

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def create_rag_agent_collection_and_utils(
    *,
    persist_dir: Path,
    collection_name: str,
    embed_model: str,
    device: str,
    dry_run: bool,
    logger=None,
) -> Tuple[QdrantClient, str, ContentUtils, WebAddition, PDFAddition]:
    embedding_fn = SentenceTransformerEmbeddingFunction(embed_model, device)
    client = QdrantClient(path=str(persist_dir))

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_fn.vector_size,
                distance=Distance.COSINE,
            ),
        )

    content_utils = ContentUtils(embed_model=embed_model, embedding_fn=embedding_fn)

    if dry_run:
        web_adder = _DryRunWebAddition(client, collection_name, content_utils, logger=logger)
        pdf_adder = _DryRunPDFAddition(client, collection_name, content_utils, logger=logger)
    else:
        web_adder = WebAddition(client=client, collection_name=collection_name, content_utils=content_utils)
        pdf_adder = PDFAddition(client=client, collection_name=collection_name, content_utils=content_utils)

    return client, collection_name, content_utils, web_adder, pdf_adder
```

**Note on the `_DryRunCollection` shim:**

The current `_DryRunCollection` wraps a Chroma collection to intercept writes. With Qdrant, you have two options:
1. Subclass `WebAddition`/`PDFAddition` and override the `client.upsert(...)` call to be a no-op — this is the cleanest approach.
2. Replace the shim with a mock `QdrantClient` that suppresses upsert calls.

The `_DryRunCollection.get(...)` for deduplication reads should still pass through to the real client.

**Return signature change:** The function now returns `(client, collection_name, ...)` instead of `(collection, ...)`. All callers in `bootstrap.py` must be updated (see below).

---

### 4.9 `preload_pipeline/bootstrap.py`

**Changes required:**

1. **`--persist-dir` help text** (line 44): Change `"Chroma persistence directory"` → `"Qdrant persistence directory"`.

2. **`--collection` help text** (line 45): Change `"Chroma collection name"` → `"Qdrant collection name"`.

3. **`--dry-run` help text** (line 51): Change `"Do everything except writing to Chroma"` → `"Do everything except writing to Qdrant"`.

4. **Unpacking the return value of `create_rag_agent_collection_and_utils` (line 100):**

```python
# Current:
collection, content_utils, web_adder, pdf_adder = create_rag_agent_collection_and_utils(...)

# New (returns client + collection_name instead of collection):
client, collection_name, content_utils, web_adder, pdf_adder = create_rag_agent_collection_and_utils(...)
```

5. **`CSVAdapter` instantiation (line 114):** The `CSVAdapter` receives `collection=collection`. Change to `client=client, collection_name=collection_name`. The `CSVAdapter` class itself must be updated in `csv_adapter.py` to pass these through to `ingest_csv_row_record` (and that function must also be updated to use `client.upsert(...)`).

---

## 5. Cross-Cutting API Mapping

Quick reference for every ChromaDB call in the codebase and its Qdrant equivalent.

| Operation | ChromaDB | Qdrant |
|---|---|---|
| Create local persistent client | `chromadb.PersistentClient(path=p)` | `QdrantClient(path=p)` |
| Connect to remote server | `chromadb.HttpClient(host=h, port=p)` | `QdrantClient(url="http://h:6333")` |
| Create or get a collection | `client.get_or_create_collection(name, embedding_function=fn)` | Create only if missing: `client.create_collection(collection_name, vectors_config=VectorParams(size=N, distance=Distance.COSINE))` |
| Get existing collection | `client.get_collection(name, embedding_function=fn)` | Check via `client.get_collections()` |
| Delete a collection | `client.delete_collection(name)` | `client.delete_collection(collection_name)` |
| List collections | `client.list_collections()` → list of Collection objects | `client.get_collections().collections` → list of CollectionDescription |
| Count documents | `collection.count()` | `client.count(collection_name).count` |
| Insert documents | `collection.add(documents=[], metadatas=[], ids=[])` | `client.upsert(collection_name, points=[PointStruct(id=uuid, vector=[], payload={})])` |
| Upsert documents | `collection.upsert(documents=[], metadatas=[], ids=[])` | `client.upsert(collection_name, points=[...])` (always upserts by ID) |
| Semantic search | `collection.query(query_texts=[q], n_results=k, where=filter, include=[...])` | First embed: `v = fn([q])[0]`; then `client.search(collection_name, query_vector=v, query_filter=f, limit=k, with_payload=True)` |
| Scroll / filter without vector | `collection.get(where={...})` | `client.scroll(collection_name, scroll_filter=Filter(...), limit=n)` |
| Delete points by ID | `collection.delete(ids=[...])` | `client.delete(collection_name, points_selector=PointIdsList(points=[...]))` |
| Access returned text | `results["documents"][0][i]` | `results[i].payload["text"]` |
| Access returned metadata | `results["metadatas"][0][i]` | `results[i].payload` (minus the "text" key) |
| Access distance/score | `results["distances"][0][i]` (lower = more similar for cosine) | `results[i].score` (higher = more similar, range 0–1 for cosine on normalized vectors) |

---

## 6. Metadata Filtering Syntax Translation

The entire metadata filtering system in `ContentUtils.retrieve_with_priority_filters` uses Chroma's `$eq` / `$and` syntax. Here is a complete translation reference.

### Single equality filter

```python
# Chroma:
{"field": {"$eq": "value"}}

# Qdrant:
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
Filter(must=[FieldCondition(key="field", match=MatchValue(value="value"))])
```

### AND of multiple equality filters

```python
# Chroma:
{"$and": [
    {"hardiness_zone": {"$eq": "5b"}},
    {"month_year": {"$eq": "2024-06"}},
]}

# Qdrant:
Filter(must=[
    FieldCondition(key="hardiness_zone", match=MatchValue(value="5b")),
    FieldCondition(key="month_year", match=MatchValue(value="2024-06")),
])
```

### All seven filter strategies in `retrieve_with_priority_filters`

| Strategy | Chroma `where` | Qdrant `Filter` |
|---|---|---|
| `hardiness_zone+month_year+title` | `{"$and": [{hz: {$eq}}, {my: {$eq}}, {title: {$eq}}]}` | `Filter(must=[hz_cond, my_cond, title_cond])` |
| `hardiness_zone+title` | `{"$and": [{hz: {$eq}}, {title: {$eq}}]}` | `Filter(must=[hz_cond, title_cond])` |
| `title` | `{"title": {"$eq": val}}` | `Filter(must=[title_cond])` |
| `month_year` | `{"month_year": {"$eq": val}}` | `Filter(must=[my_cond])` |
| `hardiness_zone+month_year` | `{"$and": [{hz: {$eq}}, {my: {$eq}}]}` | `Filter(must=[hz_cond, my_cond])` |
| `hardiness_zone` | `{"hardiness_zone": {"$eq": val}}` | `Filter(must=[hz_cond])` |
| `semantic_only` | `None` | `None` |

### Required Qdrant payload index (IMPORTANT)

For filtered searches to perform well in Qdrant, you must create payload indexes for the fields you filter on. Do this once when creating the collection:

```python
for field in ["hardiness_zone", "month_year", "title", "content_hash"]:
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field,
        field_schema="keyword",   # all these fields are string equality matches
    )
```

Without these indexes, Qdrant will full-scan the payload on every filtered query.

---

## 7. Distance Score Semantics

### ChromaDB

With `BAAI/bge-base-en-v1.5` and normalized vectors (which is what `normalize_embeddings=True` gives you), Chroma returns distances as `1 - cosine_similarity`. So:

- Distance `0.0` = identical vectors
- Distance `1.0` = orthogonal vectors
- Distance `2.0` = opposite vectors

The existing conversion in `ContentUtils` and `ConfidenceEvaluator`:

```python
# ContentUtils (line 251):
similarity_scores.append(1.0 / (1.0 + max(float(distance), 0.0)))

# ConfidenceEvaluator (line 91):
similarities = [1 - r["distance"] for r in results]
```

### Qdrant with `Distance.COSINE`

Qdrant returns a `score` that equals `cosine_similarity` directly:

- Score `1.0` = identical vectors
- Score `0.0` = orthogonal
- Score `-1.0` = opposite

**To keep all existing logic working without changes**, normalize Qdrant scores to Chroma's "distance" format in `ContentUtils`:

```python
# In retrieve_with_priority_filters, after searching:
distances = [1.0 - hit.score for hit in hits]
```

This makes `distance = 1 - cosine_similarity`, which is exactly what Chroma returned. All downstream code (`ConfidenceEvaluator`, `ContentUtils` similarity conversion) requires zero changes.

---

## 8. Persistence & Deployment Modes

### Local persistent mode (equivalent to `chromadb.PersistentClient`)

```python
from qdrant_client import QdrantClient

client = QdrantClient(path="/path/to/qdrant_database")
```

The data is stored on disk at the given path. This is the direct replacement for `chromadb.PersistentClient(path=...)`. Uses the `qdrant-client[local]` extra.

### In-memory mode (for testing, equivalent to `chromadb.Client()`)

```python
client = QdrantClient(":memory:")
```

### Remote server mode

```python
client = QdrantClient(url="http://localhost:6333")
# With API key:
client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-api-key")
```

### Backup strategy

The preload pipeline's backup stage copies the entire Chroma persistence directory before each run. With Qdrant in local persistent mode, the same strategy applies — copy the Qdrant storage directory. The `backup_persist_dir` utility in `preload_pipeline/preload/pipeline/backup.py` requires no changes since it just copies a directory.

---

## 9. Data Migration (Existing Database)

If you have an existing ChromaDB database that you need to migrate to Qdrant:

### Step 1: Export from ChromaDB

```python
import chromadb

client = chromadb.PersistentClient(path="/path/to/chroma_db")
collection = client.get_collection("meta-mirage_collection")

# Export all data (Chroma stores up to 41,666 items per batch; page if needed)
total = collection.count()
batch_size = 5000
all_ids, all_docs, all_metas, all_embeddings = [], [], [], []

for offset in range(0, total, batch_size):
    result = collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents", "metadatas", "embeddings"],
    )
    all_ids.extend(result["ids"])
    all_docs.extend(result["documents"])
    all_metas.extend(result["metadatas"])
    all_embeddings.extend(result["embeddings"])
```

### Step 2: Import into Qdrant

```python
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

qclient = QdrantClient(path="/path/to/qdrant_database")
qclient.create_collection(
    collection_name="meta-mirage_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

points = []
for chroma_id, doc, meta, embedding in zip(all_ids, all_docs, all_metas, all_embeddings):
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chroma_id))
    payload = {"text": doc, **meta}
    points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

# Upsert in batches
batch_size = 256
for i in range(0, len(points), batch_size):
    qclient.upsert(collection_name="meta-mirage_collection", points=points[i:i+batch_size])
    print(f"Migrated {min(i+batch_size, len(points))}/{len(points)} points")

# Create payload indexes for fast filtering
for field in ["hardiness_zone", "month_year", "title", "content_hash"]:
    qclient.create_payload_index(
        collection_name="meta-mirage_collection",
        field_name=field,
        field_schema="keyword",
    )
```

> **Important:** The `embedding` values exported from ChromaDB are the raw vectors stored internally by Chroma. They are the same normalized `BAAI/bge-base-en-v1.5` vectors and can be reused directly in Qdrant — no re-embedding needed.

---

## 10. Testing Checklist

After completing the migration, verify the following:

### Unit-level
- [ ] `SentenceTransformerEmbeddingFunction.__call__` still returns a list of float lists
- [ ] `SentenceTransformerEmbeddingFunction.embed_one` returns a single float list
- [ ] `SentenceTransformerEmbeddingFunction.vector_size` returns 768 for `bge-base-en-v1.5`
- [ ] `ContentUtils.content_hash_exists` correctly returns `True` for an existing hash and `False` for a new one
- [ ] `ContentUtils.retrieve_with_priority_filters` returns results in the same shape `[{"text": ..., "metadata": ..., "distance": ...}]`
- [ ] All seven filter strategies in `retrieve_with_priority_filters` produce results (or empty lists for no-match)

### Integration-level
- [ ] `WebAddition.add_web_content` successfully adds a web page and returns `{"status": "success", "chunks_added": N, ...}`
- [ ] `PDFAddition.add_pdf_content` successfully adds a PDF and returns `{"status": "success", ...}`
- [ ] `ConfidenceEvaluator.evaluate_retrieval_confidence` returns `confidence_level` in `{"high", "medium", "low"}`
- [ ] `MainAgent.retrieve_content` returns results after inserting content
- [ ] `MainAgent.reset_collection` drops and recreates the collection cleanly
- [ ] `MainAgent.reload_existing_collection` loads the collection and logs the correct count
- [ ] Duplicate content is correctly skipped (deduplication via `content_hash_exists`)

### Preload pipeline
- [ ] `bootstrap.py` runs end-to-end with `--dry-run` and reports `sources_succeeded > 0`
- [ ] `bootstrap.py` runs without `--dry-run` and data appears in Qdrant collection
- [ ] Backup stage correctly copies the Qdrant storage directory

### test_standalone.py
- [ ] `python test_standalone.py --query "..." --reset-collection` completes without errors
- [ ] `python test_standalone.py --query "..."` retrieves results from the populated collection
- [ ] `--db-path` argument works correctly with the new Qdrant client (update the `MainAgent.__init__` signature to accept `db_path` since the current code notes it as a TODO)

---

*End of migration guide.*
