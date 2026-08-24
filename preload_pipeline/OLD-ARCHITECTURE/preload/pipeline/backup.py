from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


def backup_persist_dir(
    *,
    persist_dir: Path,
    backups_root: Path,
    label: str = "before_preload",
    keep_last: Optional[int] = 20,
    logger=None,
) -> Path:
    if not persist_dir.exists() or not persist_dir.is_dir():
        raise FileNotFoundError(f"Persist dir not found: {persist_dir}")

    backups_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_dir = backups_root / f"{ts}_{label}"

    if logger:
        logger.info(f"Backup: {persist_dir} -> {backup_dir}")

    shutil.copytree(persist_dir, backup_dir)

    if keep_last is not None:
        backups = sorted([p for p in backups_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
        for old in backups[keep_last:]:
            if logger:
                logger.info(f"Prune old backup: {old}")
            shutil.rmtree(old, ignore_errors=True)

    return backup_dir