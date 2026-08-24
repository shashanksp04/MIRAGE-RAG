from __future__ import annotations

from typing import Any, Dict, Optional


def record_to_text(record: Dict[str, Any], entity_type: Optional[str] = None) -> str:
    lines = []
    if entity_type:
        lines.append(f"ENTITY_TYPE: {entity_type}")

    # A few common "name" fields first
    preferred_keys = [
        "Scientific Name", "scientific_name", "ScientificName",
        "Common Name", "common_name", "CommonName",
        "Name", "name",
        "Symbol", "symbol",
        "ID", "id",
    ]

    used = set()
    for k in preferred_keys:
        if k in record and record[k] not in (None, ""):
            lines.append(f"{k}: {str(record[k]).strip()}")
            used.add(k)

    for k, v in record.items():
        if k in used or v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        lines.append(f"{k}: {s}")

    return "\n".join(lines).strip()