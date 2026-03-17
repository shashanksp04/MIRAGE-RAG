from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class PreloadConfig:
    manifest_path: Path
    persist_dir: Path
    collection_name: str
    backups_root: Path
    keep_last: int
    sources: List[Dict[str, Any]]
    embed_model: str
    device: str
    dry_run: bool

    @staticmethod
    def from_manifest(
        *,
        manifest_path: Path,
        persist_dir: Path,
        collection_name: str,
        backups_root: Path,
        keep_last: int,
        embed_model: str,
        device: str,
        dry_run: bool,
    ) -> "PreloadConfig":
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        sources = data.get("sources", [])
        if not isinstance(sources, list) or not sources:
            raise ValueError("manifest.yaml must contain a non-empty 'sources' list")

        for s in sources:
            if "name" not in s or "type" not in s:
                raise ValueError(f"Each source must have name and type. Bad source: {s}")
            st = str(s["type"]).strip().lower()
            if st == "csv":
                if not s.get("location") and not s.get("location_field"):
                    raise ValueError(
                        f"{s['name']}: csv sources require 'location' or 'location_field' "
                        f"to derive hardiness_zone metadata."
                    )
            elif st in {"web_page_list", "pdf_dir"}:
                if not s.get("location"):
                    raise ValueError(
                        f"{s['name']}: {st} sources require 'location' "
                        f"to derive hardiness_zone metadata."
                    )

        return PreloadConfig(
            manifest_path=manifest_path,
            persist_dir=persist_dir,
            collection_name=collection_name,
            backups_root=backups_root,
            keep_last=keep_last,
            sources=sources,
            embed_model=embed_model,
            device=device,
            dry_run=dry_run,
        )