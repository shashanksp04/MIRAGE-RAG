#!/usr/bin/env python3
"""Smoke tests for ChromaDB -> Qdrant migration (rag_agent)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from rag_agent.utils.ContentUtils import ContentUtils
from rag_agent.utils.qdrant_store import (
    QdrantStore,
    chroma_where_to_qdrant_filter,
    string_id_to_point_id,
)


class _MockEmbeddingFunction:
    vector_size = 8

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / 8.0] * self.vector_size for i, _ in enumerate(texts)]

    def embed_one(self, text: str) -> list[float]:
        return self([text])[0]


def _get_embedding_fn():
    try:
        from rag_agent.utils.Embedding import SentenceTransformerEmbeddingFunction
        return SentenceTransformerEmbeddingFunction(device="cpu")
    except Exception as exc:
        print(f"[smoke] SentenceTransformer unavailable ({exc}); using mock embeddings")
        return _MockEmbeddingFunction()


def _make_client() -> tuple[QdrantClient, bool]:
    """Return (client, is_server_mode)."""
    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    try:
        client = QdrantClient(url=url, timeout=3, check_compatibility=False)
        client.get_collections()
        print(f"[smoke] Using Qdrant server at {url}")
        return client, True
    except Exception as exc:
        tmp = tempfile.mkdtemp(prefix="qdrant_smoke_")
        print(f"[smoke] Server unavailable ({exc}); using local path {tmp}")
        return QdrantClient(path=tmp, check_compatibility=False), False


def test_embedding_helpers() -> None:
    fn = _get_embedding_fn()
    batch = fn(["hello world", "second doc"])
    one = fn.embed_one("hello world")
    assert isinstance(batch, list) and isinstance(batch[0], list)
    assert isinstance(one, list) and len(one) == fn.vector_size
    print("[smoke] embedding helpers OK")


def test_filter_translation() -> None:
    where = {
        "$and": [
            {"hardiness_zone": {"$eq": "5b"}},
            {"month_year": {"$eq": "2024-06"}},
        ]
    }
    filt = chroma_where_to_qdrant_filter(where)
    assert isinstance(filt, Filter)
    assert len(filt.must) == 2
    print("[smoke] filter translation OK")


def test_store_lifecycle() -> None:
    client, is_server = _make_client()
    collection_name = "smoke_test_collection"
    embed_fn = _get_embedding_fn()
    store = QdrantStore(client, collection_name, embed_fn)

    try:
        store.delete_collection()
    except Exception:
        pass

    store.ensure_collection()
    assert store.count() == 0

    chunk_text = "Corn planting guidance for Minnesota."
    text = f"Title: Demo\n\n{chunk_text}"
    meta = {
        "source_type": "web",
        "source_id": "abc123",
        "title": "Demo",
        "url": "https://example.edu/page",
        "page": -1,
        "chunk_index": 0,
        "location": "MINNESOTA",
        "month_year": "2024-06",
        "content_hash": ContentUtils.compute_content_hash(chunk_text),
        "language": "en",
        "hardiness_zone": "4a",
    }
    doc_id = "abc123_c0"
    store.upsert_chunks(texts=[text], metadatas=[meta], string_ids=[doc_id])
    assert store.count() == 1
    assert store.content_hash_exists(meta["content_hash"])
    assert string_id_to_point_id(doc_id) == string_id_to_point_id(doc_id)

    hits = store.search(
        query_vector=embed_fn.embed_one("corn planting"),
        limit=5,
        qdrant_filter=chroma_where_to_qdrant_filter({"title": {"$eq": "Demo"}}),
    )
    assert hits, "expected search hits"
    assert hits[0]["text"] == text
    assert hits[0]["metadata"]["title"] == "Demo"

    store.delete_collection()
    store.ensure_collection()
    if is_server:
        assert store.count() == 0
    else:
        print("[smoke] Skipping post-delete count check in local embedded mode")
    print("[smoke] store lifecycle OK")


def main() -> None:
    test_embedding_helpers()
    test_filter_translation()
    test_store_lifecycle()
    print("[smoke] All tests passed")


if __name__ == "__main__":
    main()
