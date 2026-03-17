from __future__ import annotations

from typing import Any, Dict, List

from preload.adapters.base import BaseAdapter


class WebPageListAdapter(BaseAdapter):
    """
    Uses rag_agent.tools.web_addition.WebAddition.add_web_content directly.
    That means:
      - extraction = rag_agent (trafilatura)
      - chunking = rag_agent ContentUtils tokenizer chunking
      - dedupe = rag_agent content_hash_exists
      - formatting = rag_agent (Title: ... + content)
    """

    def __init__(self, source_cfg: Dict[str, Any], *, web_adder, dry_run: bool):
        super().__init__(source_cfg, dry_run=dry_run)
        self.web_adder = web_adder

    def run(self, logger=None) -> Dict[str, int]:
        urls: List[str] = self.source_cfg.get("urls", [])
        if not urls:
            raise ValueError(f"{self.source_name}: web_page_list requires 'urls'")

        location = self.source_cfg.get("location")
        month_year = self.source_cfg.get("month_year")

        processed = added = skipped = failed = 0

        for url in urls:
            processed += 1
            try:
                # rag_agent tool accepts location/month_year; other source-level fields remain for provenance only.
                res = self.web_adder.add_web_content(url=url, location=location, month_year=month_year)
                if res.get("status") == "success":
                    # rag_agent reports chunks_added + skipped
                    added += int(res.get("chunks_added", 0))
                    skipped += int(res.get("chunks_skipped_as_duplicates", 0))
                else:
                    failed += 1
                    if logger:
                        logger.warning(f"Web add failed for {url}: {res}")
            except Exception as e:
                failed += 1
                if logger:
                    logger.exception(f"Web add exception for {url}: {e}")

        return {
            "items_processed": processed,
            "items_added": added,
            "items_skipped": skipped,
            "items_failed": failed,
        }


