from __future__ import annotations

import os
from typing import Any, Tuple

from qdrant_client import QdrantClient

from rag_agent.utils.Embedding import SentenceTransformerEmbeddingFunction
from rag_agent.utils.ContentUtils import ContentUtils
from rag_agent.utils.qdrant_store import QdrantStore
from rag_agent.tools.web_addition import WebAddition
from rag_agent.tools.pdf_addition import PDFAddition


class _DryRunStore:
    """
    A tiny shim to prevent Qdrant writes during --dry-run while still exercising ingestion logic.
    """
    def __init__(self, real_store, logger=None):
        self.real = real_store
        self.logger = logger

    def content_hash_exists(self, content_hash):
        return self.real.content_hash_exists(content_hash)

    def upsert_chunks(self, *, texts, metadatas, string_ids):
        if self.logger:
            self.logger.info(f"[DRY-RUN] Would upsert {len(texts)} chunks")


def create_rag_agent_collection_and_utils(
    *,
    qdrant_url: str,
    qdrant_api_key: str | None,
    collection_name: str,
    embed_model: str,
    device: str,
    dry_run: bool,
    logger=None,
) -> Tuple[Any, ContentUtils, WebAddition, PDFAddition]:
    """
    Creates the same Qdrant store + ContentUtils + tools that rag_agent uses,
    so chunking, dedupe, embeddings, and document formatting match exactly.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(embed_model, device)
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY") or None,
        check_compatibility=False,
    )
    store = QdrantStore(client, collection_name, embedding_fn)
    store.ensure_collection()

    content_utils = ContentUtils(embed_model=embed_model, embedding_fn=embedding_fn)

    tool_store = _DryRunStore(store, logger=logger) if dry_run else store

    web_adder = WebAddition(store=tool_store, content_utils=content_utils)
    pdf_adder = PDFAddition(store=tool_store, content_utils=content_utils)

    return tool_store, content_utils, web_adder, pdf_adder