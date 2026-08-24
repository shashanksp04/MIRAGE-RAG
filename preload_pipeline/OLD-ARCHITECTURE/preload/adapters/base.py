from __future__ import annotations

from typing import Any, Dict


class BaseAdapter:
    def __init__(self, source_cfg: Dict[str, Any], *, dry_run: bool):
        self.source_cfg = source_cfg
        self.source_name = source_cfg["name"]
        self.source_type = source_cfg["type"]
        self.dry_run = dry_run

    def run(self, logger=None) -> Dict[str, int]:
        raise NotImplementedError