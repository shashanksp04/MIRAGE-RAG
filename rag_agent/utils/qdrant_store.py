from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

PAYLOAD_INDEX_FIELDS = ("hardiness_zone", "month_year", "title", "content_hash")


def string_id_to_point_id(string_id: str) -> str:
    """Convert a string chunk ID to a deterministic Qdrant-compatible UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))


def chroma_where_to_qdrant_filter(where: dict) -> Filter:
    """Convert a Chroma where-clause dict to a Qdrant Filter."""
    if "$and" in where:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v["$eq"]))
            for clause in where["$and"]
            for k, v in clause.items()
        ]
        return Filter(must=conditions)

    for field, cond in where.items():
        return Filter(must=[FieldCondition(key=field, match=MatchValue(value=cond["$eq"]))])

    raise ValueError(f"Unsupported where filter: {where}")


class QdrantStore:
    """Thin adapter over QdrantClient for rag_agent ingest and retrieval."""

    def __init__(self, client: QdrantClient, collection_name: str, embedding_fn, *, require_existing: bool = False):
        self.client = client
        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.require_existing = require_existing

    def _collection_names(self) -> List[str]:
        return [c.name for c in self.client.get_collections().collections]

    def _create_payload_indexes(self) -> None:
        for field in PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "already exist" in msg:
                    continue
                raise

    def ensure_collection(self) -> None:
        """Create the collection and payload indexes if they do not exist."""
        if self.collection_name not in self._collection_names():
            if self.require_existing:
                raise RuntimeError(f"Required Qdrant collection does not exist: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_fn.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        self._create_payload_indexes()

    def delete_collection(self) -> None:
        """Delete the collection if it exists."""
        if self.collection_name in self._collection_names():
            self.client.delete_collection(collection_name=self.collection_name)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def content_hash_exists(self, content_hash: str) -> bool:
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="content_hash",
                        match=MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

    def upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        string_ids: List[str],
    ) -> None:
        if not texts:
            return

        vectors = self.embedding_fn(texts)
        points = []
        for doc_id, doc_text, meta, vector in zip(string_ids, texts, metadatas, vectors):
            point_id = string_id_to_point_id(doc_id)
            payload = {"text": doc_text, "chunk_id": doc_id, **meta}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: List[float],
        limit: int,
        qdrant_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        hits = response.points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(
                {
                    "text": payload.get("text", ""),
                    "metadata": metadata,
                    # Qdrant COSINE scores are higher-is-better similarities.
                    "similarity": float(hit.score),
                }
            )
        return results
