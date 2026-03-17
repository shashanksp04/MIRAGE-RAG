from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from preload.adapters.base import BaseAdapter


class PDFDirAdapter(BaseAdapter):
    """
    Uses rag_agent.tools.pdf_addition.PDFAddition.add_pdf_content directly.
    """

    def __init__(self, source_cfg: Dict[str, Any], *, pdf_adder, dry_run: bool):
        super().__init__(source_cfg, dry_run=dry_run)
        self.pdf_adder = pdf_adder

    def run(self, logger=None) -> Dict[str, int]:
        pdf_dir = Path(self.source_cfg["path"]).resolve()
        if not pdf_dir.exists():
            raise FileNotFoundError(f"{self.source_name}: pdf_dir not found: {pdf_dir}")

        location = self.source_cfg.get("location")
        month_year = self.source_cfg.get("month_year")
        processed = added = skipped = failed = 0

        for pdf_path in pdf_dir.rglob("*.pdf"):
            processed += 1
            try:
                source_id = f"{self.source_name}:{pdf_path.name}"
                res = self.pdf_adder.add_pdf_content(
                    pdf_path=str(pdf_path),
                    source_id=source_id,
                    title=pdf_path.name,
                    location=location,
                    month_year=month_year,
                )
                if res.get("status") == "success":
                    added += int(res.get("chunks_added", 0))
                    skipped += int(res.get("chunks_skipped_as_duplicates", 0))
                else:
                    failed += 1
                    if logger:
                        logger.warning(f"PDF add failed for {pdf_path}: {res}")
            except Exception as e:
                failed += 1
                if logger:
                    logger.exception(f"PDF add exception for {pdf_path}: {e}")

        return {
            "items_processed": processed,
            "items_added": added,
            "items_skipped": skipped,
            "items_failed": failed,
        }