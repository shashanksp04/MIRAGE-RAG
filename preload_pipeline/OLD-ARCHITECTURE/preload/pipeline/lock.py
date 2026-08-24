from __future__ import annotations

import os
import time
from pathlib import Path


class FileLock:
    def __init__(self, lock_path: Path, timeout_s: int = 10, poll_s: float = 0.25, logger=None):
        self.lock_path = Path(lock_path)
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.logger = logger
        self._fd = None

    def __enter__(self):
        start = time.time()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                if self.logger:
                    self.logger.info(f"Lock acquired: {self.lock_path}")
                return self
            except FileExistsError:
                if time.time() - start > self.timeout_s:
                    raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                os.close(self._fd)
            if self.lock_path.exists():
                self.lock_path.unlink()
            if self.logger:
                self.logger.info(f"Lock released: {self.lock_path}")
        except Exception:
            pass
        return False