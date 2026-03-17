from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from preload.adapters.base import BaseAdapter
from preload.ingestion.csv_ingestion import ingest_csv_row_record


class CSVAdapter(BaseAdapter):
    def __init__(self, source_cfg: Dict[str, Any], *, collection, content_utils, dry_run: bool):
        super().__init__(source_cfg, dry_run=dry_run)
        self.collection = collection
        self.content_utils = content_utils

    def run(self, logger=None) -> Dict[str, int]:
        path = Path(self.source_cfg["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{self.source_name}: CSV not found: {path}")

        id_field = self.source_cfg.get("id_field")
        entity_type = self.source_cfg.get("entity_type")
        source_org = self.source_cfg.get("source_org")
        source_location = self.source_cfg.get("location")
        location_field = self.source_cfg.get("location_field")
        tags = self.source_cfg.get("tags", [])

        processed = added = skipped = failed = 0

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                processed += 1
                try:
                    record_id = (row.get(id_field) if id_field else None) or str(idx)
                    row_location = (row.get(location_field) if location_field else None) or source_location
                    if not row_location or not str(row_location).strip():
                        raise ValueError(
                            f"{self.source_name}: missing location for row {idx}. "
                            f"Provide source-level 'location' or populate '{location_field}'."
                        )
                    stats = ingest_csv_row_record(
                        collection=self.collection,
                        content_utils=self.content_utils,
                        source_name=self.source_name,
                        csv_path=str(path),
                        record=row,
                        record_id=str(record_id),
                        entity_type=entity_type,
                        source_org=source_org,
                        location=row_location,
                        tags=tags,
                        dry_run=self.dry_run,
                    )
                    added += stats["chunks_added"]
                    skipped += stats["chunks_skipped"]
                except Exception as e:
                    failed += 1
                    if logger:
                        logger.exception(f"CSV row failed ({self.source_name} row {idx}): {e}")

        return {
            "items_processed": processed,
            "items_added": added,
            "items_skipped": skipped,
            "items_failed": failed,
        }