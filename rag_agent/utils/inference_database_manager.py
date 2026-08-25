from __future__ import annotations

import re
import atexit
from datetime import datetime
from typing import Optional

from qdrant_client.http.models import Distance, VectorParams

BASE_COLLECTION = "mirage_base"
VECTOR_SIZE = 768
RUNTIME_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RUNTIME_PREFIX = "mirage_runtime_"
RUNTIME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def sanitize_ablation_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "default"


class InferenceDatabaseManager:
    """Owns inference collection lifecycle; never mutates the curated base."""

    def __init__(self, qdrant_client, *, base_collection=BASE_COLLECTION,
                 use_base_collection=True, ablation_id="default",
                 runtime_mode="resume", runtime_collection_override=None,
                 snapshot_runtime=False, vector_size=VECTOR_SIZE,
                 distance=Distance.COSINE):
        if not isinstance(use_base_collection, bool):
            raise ValueError("use_base_collection must be boolean")
        if not (ablation_id or "").strip():
            raise ValueError("ablation_id must be non-empty")
        if runtime_mode not in {"resume", "fresh"}:
            raise ValueError("runtime_mode must be 'resume' or 'fresh'")
        if runtime_collection_override and runtime_mode != "resume":
            raise ValueError("runtime_collection_override is only valid in resume mode")
        if runtime_collection_override and not runtime_collection_override.startswith(RUNTIME_PREFIX):
            raise ValueError("runtime_collection_override must name a runtime collection")
        if use_base_collection and not (base_collection or "").strip():
            raise ValueError("base_collection must be non-empty when enabled")
        self.client = qdrant_client
        self.base_collection = base_collection
        self.use_base_collection = use_base_collection
        self.ablation_id = ablation_id.strip()
        self.safe_ablation_id = sanitize_ablation_id(self.ablation_id)
        self.runtime_mode = runtime_mode
        self.runtime_collection_override = runtime_collection_override
        self.snapshot_runtime = bool(snapshot_runtime)
        self.vector_size = vector_size
        self.distance = distance
        self.active_runtime_collection: Optional[str] = None
        self._finalized = False
        atexit.register(self._preserve_at_exit)

    def _preserve_at_exit(self):
        if self.active_runtime_collection and not self._finalized:
            self.preserve_failure()

    def collection_names(self):
        return [c.name for c in self.client.get_collections().collections]

    def validate_base(self):
        if self.use_base_collection and self.base_collection not in self.collection_names():
            raise RuntimeError(
                f"Required base Qdrant collection '{self.base_collection}' was not found. "
                "Start Qdrant using the curated base collection, or run with "
                "USE_BASE_COLLECTION=False for runtime-only development/testing."
            )

    def list_matching_runtime_collections(self):
        prefix = f"{RUNTIME_PREFIX}{self.safe_ablation_id}_"
        candidates = []
        for name in self.collection_names():
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            try:
                stamp = datetime.strptime(suffix, RUNTIME_TIMESTAMP_FORMAT)
            except ValueError:
                continue
            candidates.append((stamp, name))
        return [name for _, name in sorted(candidates, reverse=True)]

    def _create_runtime(self, name):
        if name in self.collection_names():
            raise RuntimeError(f"Runtime collection already exists: {name}")
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )
        for field in ("hardiness_zone", "month_year", "title", "content_hash"):
            self.client.create_payload_index(
                collection_name=name, field_name=field, field_schema="keyword"
            )
        return name

    def _ensure_runtime_indexes(self, name):
        for field in ("hardiness_zone", "month_year", "title", "content_hash"):
            try:
                self.client.create_payload_index(
                    collection_name=name, field_name=field, field_schema="keyword"
                )
            except Exception as exc:
                if "already exist" not in str(exc).lower():
                    raise

    def delete_matching_runtime_collections(self):
        deleted = []
        for name in self.list_matching_runtime_collections():
            self.client.delete_collection(collection_name=name)
            deleted.append(name)
        return deleted

    def resolve_runtime_collection(self):
        self.validate_base()
        if self.runtime_collection_override:
            if self.runtime_collection_override not in self.collection_names():
                raise RuntimeError(f"Runtime collection does not exist: {self.runtime_collection_override}")
            self.active_runtime_collection = self.runtime_collection_override
            self._ensure_runtime_indexes(self.active_runtime_collection)
            return self.active_runtime_collection

        candidates = self.list_matching_runtime_collections()
        if self.runtime_mode == "resume" and candidates:
            if len(candidates) > 1:
                print(f"[DB] Runtime candidates for {self.ablation_id}: {candidates}", flush=True)
            self.active_runtime_collection = candidates[0]
            self._ensure_runtime_indexes(self.active_runtime_collection)
            print(f"[DB] Resuming runtime collection: {self.active_runtime_collection}", flush=True)
            return self.active_runtime_collection

        if self.runtime_mode == "fresh":
            for name in self.delete_matching_runtime_collections():
                print(f"[DB] Deleted interrupted runtime collection: {name}", flush=True)
        stamp = datetime.now().strftime(RUNTIME_TIMESTAMP_FORMAT)
        name = f"{RUNTIME_PREFIX}{self.safe_ablation_id}_{stamp}"
        self.active_runtime_collection = self._create_runtime(name)
        print(f"[DB] Created runtime collection: {name}", flush=True)
        return name

    def snapshot_runtime_if_enabled(self):
        if not self.snapshot_runtime:
            return None
        if not self.active_runtime_collection:
            raise RuntimeError("No active runtime collection to snapshot")
        result = self.client.create_snapshot(collection_name=self.active_runtime_collection)
        print(f"[DB] Runtime snapshot created: {result}", flush=True)
        return result

    def finalize_success(self):
        self.snapshot_runtime_if_enabled()
        if self.active_runtime_collection:
            self.client.delete_collection(collection_name=self.active_runtime_collection)
            self._finalized = True
            print("[DB] Runtime collection deleted after successful completion.", flush=True)

    def preserve_failure(self):
        if self.active_runtime_collection:
            print(f"Inference interrupted. Runtime collection preserved: {self.active_runtime_collection}", flush=True)
