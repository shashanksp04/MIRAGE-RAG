from __future__ import annotations

import logging
import sys


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("preload")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(h)
    return logger