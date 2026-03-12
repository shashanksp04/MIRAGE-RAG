from __future__ import annotations

from typing import Any, Dict, Optional


def build_canonical_chunk_metadata(
    *,
    source_type: str,
    source_id: str,
    title: str,
    url: str,
    page: int,
    chunk_index: int,
    location: str,
    month_year: str,
    content_hash: str,
    language: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the canonical chunk metadata payload used across ingestion paths.

    Canonical keys:
      source_type, source_id, title, url, page, chunk_index,
      location, month_year, content_hash, language
    """
    metadata: Dict[str, Any] = {
        "source_type": source_type,
        "source_id": source_id,
        "title": title,
        "url": url,
        "page": page,
        "chunk_index": chunk_index,
        "location": location,
        "month_year": month_year,
        "content_hash": content_hash,
        "language": language,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata
