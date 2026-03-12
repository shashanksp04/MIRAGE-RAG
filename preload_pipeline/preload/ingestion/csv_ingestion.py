from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_agent.utils.metadata import build_canonical_chunk_metadata
from preload.transforms.record_to_text import record_to_text
from preload.transforms.normalize import normalize_text


def ingest_csv_row_record(
    *,
    collection,
    content_utils,
    source_name: str,
    csv_path: str,
    record: Dict[str, Any],
    record_id: str,
    entity_type: Optional[str],
    source_org: Optional[str],
    tags: List[str],
    dry_run: bool,
) -> Dict[str, int]:
    """
    Row -> text -> rag_agent token chunking -> rag_agent dedupe -> collection.add()

    Reuses:
      - content_utils.tokenizer
      - content_utils.chunk_by_tokens
      - content_utils.compute_content_hash
      - content_utils.content_hash_exists
    """
    text = record_to_text(record, entity_type=entity_type)
    text = normalize_text(text)
    if not text:
        return {"chunks_added": 0, "chunks_skipped": 0}

    # Chunk using rag_agent token chunker
    chunks = content_utils.chunk_by_tokens(
        text,
        max_tokens=content_utils.chunk_config["web"]["max_tokens"],  # reusing "web" chunking defaults
        overlap=content_utils.chunk_config["web"]["overlap"],
    )

    documents = []
    metadatas = []
    ids = []

    added = 0
    skipped = 0
    seen_hashes = set()

    for chunk_index, chunk in enumerate(chunks):
        content_hash = content_utils.compute_content_hash(chunk)

        if content_hash in seen_hashes:
            skipped += 1
            continue

        # dedupe against DB
        if content_utils.content_hash_exists(collection, content_hash):
            skipped += 1
            continue

        seen_hashes.add(content_hash)

        source_id = f"{source_name}:{record_id}"
        doc_id = f"{source_id}_c{chunk_index}"
        title = source_name

        documents.append(chunk)
        metadatas.append(
            build_canonical_chunk_metadata(
                source_type="csv",
                source_id=source_id,
                title=title,
                url="",
                page=-1,
                chunk_index=chunk_index,
                location="",
                month_year="",
                content_hash=content_hash,
                language="en",
                extra_metadata={
                    "source_name": source_name,
                    "path": csv_path,
                    "record_id": record_id,
                    "entity_type": entity_type or "",
                    "source_org": source_org or "",
                    "tags": tags,
                },
            )
        )
        ids.append(doc_id)
        added += 1

    if dry_run or not documents:
        return {"chunks_added": added, "chunks_skipped": skipped}

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return {"chunks_added": added, "chunks_skipped": skipped}