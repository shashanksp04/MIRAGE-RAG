from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunReport:
    manifest_path: str
    qdrant_url: str
    collection: str

    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    finished_at: Optional[str] = None
    sources_started: int = 0
    sources_succeeded: int = 0
    sources_failed: int = 0

    items_processed: int = 0
    items_added: int = 0
    items_skipped: int = 0
    items_failed: int = 0

    errors: List[Dict[str, Any]] = field(default_factory=list)

    def finalize(self):
        self.finished_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        self.finalize()
        return {
            "manifest_path": self.manifest_path,
            "qdrant_url": self.qdrant_url,
            "collection": self.collection,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "sources_started": self.sources_started,
            "sources_succeeded": self.sources_succeeded,
            "sources_failed": self.sources_failed,
            "items_processed": self.items_processed,
            "items_added": self.items_added,
            "items_skipped": self.items_skipped,
            "items_failed": self.items_failed,
            "errors": self.errors,
        }

    def write_json(self, out_dir: Path, logger=None) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"preload_run_report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        if logger:
            logger.info(f"Report written: {path}")
        return path

    def summary_str(self) -> str:
        return (
            f"Sources: {self.sources_succeeded}/{self.sources_started} ok, {self.sources_failed} failed | "
            f"Items: processed={self.items_processed} added={self.items_added} skipped={self.items_skipped} failed={self.items_failed}"
        )