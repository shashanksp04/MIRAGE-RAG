from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient

from preload.utils.hashing import sha1_hex
from preload.transforms.normalize import split_into_chunks
from rag_agent.utils.Embedding import SentenceTransformerEmbeddingFunction
from rag_agent.utils.qdrant_store import QdrantStore


@dataclass
class UpsertStats:
    chunks_created: int = 0
    chunks_upserted: int = 0
    chunks_failed: int = 0


class QdrantUpserter:
    """
    Standalone upserter that writes text chunks into a Qdrant collection.

    This does NOT depend on your RAG agent internals, so it works even if you can't import them.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model_label: str,
        device: str = "None",
        qdrant_api_key: Optional[str] = None,
        dry_run: bool = False,
        logger=None,
    ):
        self.collection_name = collection_name
        self.embedding_model_label = embedding_model_label
        self.dry_run = dry_run
        self.logger = logger

        embedding_fn = SentenceTransformerEmbeddingFunction(embedding_model_label, device)
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)
        self.store = QdrantStore(client, collection_name, embedding_fn)
        self.store.ensure_collection()

    def chunk_and_upsert(self, text: str, metadata: Dict[str, Any], stable_id: Optional[str]) -> Dict[str, int]:
        chunks = split_into_chunks(text)
        stats = UpsertStats(chunks_created=len(chunks))

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_hash = sha1_hex(chunk)
            base = stable_id or metadata.get("url") or metadata.get("path") or metadata.get("source_name") or "unknown"
            chunk_id = sha1_hex(f"{base}::chunk{idx}::${chunk_hash}")

            m = dict(metadata)
            m["chunk_index"] = idx
            m["content_hash"] = chunk_hash
            m["embedding_model_label"] = self.embedding_model_label

            ids.append(chunk_id)
            docs.append(chunk)
            metas.append(m)

        if self.dry_run:
            stats.chunks_upserted = len(ids)
            return {
                "chunks_created": stats.chunks_created,
                "chunks_upserted": stats.chunks_upserted,
                "chunks_failed": stats.chunks_failed,
            }

        try:
            self.store.upsert_chunks(texts=docs, metadatas=metas, string_ids=ids)
            stats.chunks_upserted = len(ids)
        except Exception:
            stats.chunks_failed = len(ids)
            if self.logger:
                self.logger.exception("Qdrant upsert failed.")
        return {
            "chunks_created": stats.chunks_created,
            "chunks_upserted": stats.chunks_upserted,
            "chunks_failed": stats.chunks_failed,
        }

    def close(self):
        return