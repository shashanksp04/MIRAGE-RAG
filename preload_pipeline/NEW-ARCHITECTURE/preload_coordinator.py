#!/usr/bin/env python3
"""
MetaMIRAGE Concurrent Preload Coordinator
=========================================

Small control-plane service for the concurrent MetaMIRAGE preload architecture.

Runs on the shared service node next to Qdrant.

Responsibilities
----------------
- One active worker per (build_id, state_code).
- Lease/heartbeat based state ownership.
- Atomic global deduplication claims for:
    * document
    * rag_chunk
- Lease/heartbeat based content claims.
- Idempotent state/content completion.
- State/wave status inspection for finalize_wave.ipynb.
- Persistent coordinator state in one SQLite database.

Non-responsibilities
--------------------
This service is NOT in the data path. It does not:
- receive PDFs/CSVs/pages,
- run extraction or qualification,
- chunk documents,
- compute embeddings,
- forward vectors to Qdrant,
- create/restore Qdrant snapshots,
- merge crop JSON files.

Workers process data locally and send vectors directly to Qdrant.

Worker compatibility
--------------------
This API is designed for:
    MetaMIRAGE_Concurrent_Preload_Worker.ipynb

Expected worker endpoints:
    GET  /health

    POST /state/acquire
    POST /state/heartbeat
    POST /state/release
    POST /state/complete
    GET  /state/status

    POST /content/claim
    POST /content/heartbeat
    POST /content/release
    POST /content/complete
    GET  /content/status

    GET  /wave/status

Run
---
Install dependencies if needed:

    pip install fastapi uvicorn

Choose a durable database path on shared persistent storage:

    export METAMIRAGE_COORDINATOR_DB=/work/nvme/<project>/<user>/preload_coordinator.db

Start exactly one coordinator service:

    uvicorn preload_coordinator:app \
        --host 0.0.0.0 \
        --port 8001

Workers then use:

    METAMIRAGE_COORDINATOR_URL=http://<service-node>:8001

Qdrant remains separate, for example:

    QDRANT_URL=http://<service-node>:6333

Notes
-----
- SQLite is accessed only through this service.
- Acquisition/claim operations use BEGIN IMMEDIATE transactions.
- State and content leases are recoverable after crashes.
- A newly resumed worker that successfully acquires a state's state lease can
  immediately take over stale content claims owned by an older worker for the
  SAME state. This avoids waiting for every content lease to expire after a
  notebook/kernel crash.
- Cross-state claims are never stolen while their lease is active.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

API_VERSION = "1.0"
SERVICE_NAME = "metamirage-preload-coordinator"

DB_PATH = Path(
    os.environ.get(
        "METAMIRAGE_COORDINATOR_DB",
        "./preload_coordinator.db",
    )
).expanduser().resolve()

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_STATE_LEASE_SECONDS = int(
    os.environ.get("METAMIRAGE_STATE_LEASE_SECONDS", "1800")
)
DEFAULT_CONTENT_LEASE_SECONDS = int(
    os.environ.get("METAMIRAGE_CONTENT_LEASE_SECONDS", "1800")
)

MIN_LEASE_SECONDS = 30
MAX_LEASE_SECONDS = 24 * 60 * 60

VALID_CONTENT_SCOPES = {"document", "rag_chunk"}


# ============================================================================
# Time helpers
# ============================================================================

def now_ts() -> float:
    return time.time()


def iso_from_ts(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_lease_seconds(value: Optional[int], default: int) -> int:
    if value is None:
        value = default
    return max(MIN_LEASE_SECONDS, min(int(value), MAX_LEASE_SECONDS))


def normalize_nonempty(value: str, field_name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def normalize_state_code(value: str) -> str:
    return normalize_nonempty(value, "state_code").upper()


def normalize_scope(value: str) -> str:
    value = normalize_nonempty(value, "scope").lower()
    if value not in VALID_CONTENT_SCOPES:
        raise ValueError(
            f"scope must be one of {sorted(VALID_CONTENT_SCOPES)}; got {value!r}"
        )
    return value


# ============================================================================
# SQLite
# ============================================================================

SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS state_leases (
    build_id TEXT NOT NULL,
    state_code TEXT NOT NULL,
    wave_id TEXT NOT NULL,
    state_name TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    status TEXT NOT NULL,

    acquired_at REAL,
    last_heartbeat REAL,
    lease_until REAL,

    completed_at REAL,
    released_at REAL,
    release_reason TEXT,

    manifest_path TEXT,
    manifest_sha256 TEXT,

    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,

    PRIMARY KEY (build_id, state_code)
);

CREATE INDEX IF NOT EXISTS idx_state_leases_wave
ON state_leases(build_id, wave_id, status);

CREATE INDEX IF NOT EXISTS idx_state_leases_worker
ON state_leases(build_id, worker_id, status);


CREATE TABLE IF NOT EXISTS content_claims (
    build_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    content_hash TEXT NOT NULL,

    wave_id TEXT NOT NULL,
    owner_state TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    status TEXT NOT NULL,

    resource_id TEXT,

    claimed_at REAL,
    last_heartbeat REAL,
    lease_until REAL,

    completed_at REAL,
    released_at REAL,
    release_reason TEXT,

    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,

    PRIMARY KEY (build_id, scope, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_content_claims_owner
ON content_claims(build_id, owner_state, status);

CREATE INDEX IF NOT EXISTS idx_content_claims_wave
ON content_claims(build_id, wave_id, status);

CREATE INDEX IF NOT EXISTS idx_content_claims_status
ON content_claims(build_id, scope, status);


CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    event_type TEXT NOT NULL,
    build_id TEXT,
    wave_id TEXT,
    state_code TEXT,
    worker_id TEXT,
    scope TEXT,
    content_hash TEXT,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_events_build
ON audit_events(build_id, created_at);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH,
        timeout=60,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_SQL)


@contextmanager
def immediate_transaction():
    """
    Serialize state/claim acquisition decisions.

    BEGIN IMMEDIATE makes the read-modify-write operation atomic across concurrent
    HTTP requests and across accidental multiple server threads/processes.
    """
    conn = get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def audit(
    conn: sqlite3.Connection,
    event_type: str,
    *,
    build_id: Optional[str] = None,
    wave_id: Optional[str] = None,
    state_code: Optional[str] = None,
    worker_id: Optional[str] = None,
    scope: Optional[str] = None,
    content_hash: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit_events(
            created_at, event_type, build_id, wave_id, state_code,
            worker_id, scope, content_hash, detail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_ts(),
            event_type,
            build_id,
            wave_id,
            state_code,
            worker_id,
            scope,
            content_hash,
            detail,
        ),
    )


init_db()


# ============================================================================
# Request models
# ============================================================================

class StateAcquireRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    state_name: str
    worker_id: str
    lease_seconds: Optional[int] = Field(default=None, ge=1)


class StateHeartbeatRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    worker_id: str
    lease_seconds: Optional[int] = Field(default=None, ge=1)


class StateReleaseRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    worker_id: str
    reason: str = "worker_release"


class StateCompleteRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    state_name: str
    worker_id: str
    manifest_path: str
    manifest_sha256: str


class ContentClaimRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    worker_id: str
    scope: Literal["document", "rag_chunk"]
    content_hash: str
    resource_id: Optional[str] = None
    lease_seconds: Optional[int] = Field(default=None, ge=1)


class ContentHeartbeatRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    worker_id: str
    scope: Literal["document", "rag_chunk"]
    content_hash: str
    lease_seconds: Optional[int] = Field(default=None, ge=1)


class ContentCompleteRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    worker_id: str
    scope: Literal["document", "rag_chunk"]
    content_hash: str
    resource_id: str


class ContentReleaseRequest(BaseModel):
    build_id: str
    wave_id: str
    state_code: str
    worker_id: str
    scope: Literal["document", "rag_chunk"]
    content_hash: str
    reason: str = "worker_release"


# ============================================================================
# Response helpers
# ============================================================================

def state_row_payload(row: sqlite3.Row, *, status: Optional[str] = None) -> Dict[str, Any]:
    now = now_ts()
    lease_until = row["lease_until"]
    return {
        "status": status or row["status"],
        "build_id": row["build_id"],
        "wave_id": row["wave_id"],
        "state_code": row["state_code"],
        "state_name": row["state_name"],
        "worker_id": row["worker_id"],
        "lease_until": iso_from_ts(lease_until),
        "lease_until_epoch": lease_until,
        "lease_expired": bool(
            row["status"] == "active"
            and lease_until is not None
            and float(lease_until) <= now
        ),
        "manifest_path": row["manifest_path"],
        "manifest_sha256": row["manifest_sha256"],
        "completed_at": iso_from_ts(row["completed_at"]),
        "released_at": iso_from_ts(row["released_at"]),
        "release_reason": row["release_reason"],
    }


def content_row_payload(
    row: sqlite3.Row,
    *,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    now = now_ts()
    lease_until = row["lease_until"]
    return {
        "status": status or row["status"],
        "build_id": row["build_id"],
        "wave_id": row["wave_id"],
        "scope": row["scope"],
        "content_hash": row["content_hash"],
        "owner_state": row["owner_state"],
        "worker_id": row["worker_id"],
        "resource_id": row["resource_id"],
        "lease_until": iso_from_ts(lease_until),
        "lease_until_epoch": lease_until,
        "lease_expired": bool(
            row["status"] == "claimed"
            and lease_until is not None
            and float(lease_until) <= now
        ),
        "completed_at": iso_from_ts(row["completed_at"]),
        "released_at": iso_from_ts(row["released_at"]),
        "release_reason": row["release_reason"],
    }


def fetch_state(
    conn: sqlite3.Connection,
    build_id: str,
    state_code: str,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM state_leases
        WHERE build_id=? AND state_code=?
        """,
        (build_id, state_code),
    ).fetchone()


def fetch_content_claim(
    conn: sqlite3.Connection,
    build_id: str,
    scope: str,
    content_hash: str,
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM content_claims
        WHERE build_id=? AND scope=? AND content_hash=?
        """,
        (build_id, scope, content_hash),
    ).fetchone()


def validate_wave_binding(
    row: sqlite3.Row,
    requested_wave_id: str,
) -> Optional[Dict[str, Any]]:
    """
    A state is permanently bound to the wave where it first started within a build.

    This prevents accidentally resuming a partially processed state under another
    wave while its state-local RUN_ID remains BUILD_ID + STATE_CODE.
    """
    if row["wave_id"] == requested_wave_id:
        return None

    payload = state_row_payload(row, status="wave_mismatch")
    payload["requested_wave_id"] = requested_wave_id
    payload["detail"] = (
        "This state is already bound to another wave in this build. "
        "Resume it with its original WAVE_ID or use a new BUILD_ID."
    )
    return payload


def active_state_owner(
    conn: sqlite3.Connection,
    *,
    build_id: str,
    wave_id: str,
    state_code: str,
    worker_id: str,
) -> tuple[bool, Dict[str, Any]]:
    row = fetch_state(conn, build_id, state_code)

    if row is None:
        return False, {
            "status": "state_not_acquired",
            "build_id": build_id,
            "wave_id": wave_id,
            "state_code": state_code,
        }

    mismatch = validate_wave_binding(row, wave_id)
    if mismatch:
        return False, mismatch

    if row["status"] == "complete":
        return False, state_row_payload(row, status="already_complete")

    if row["status"] != "active":
        return False, state_row_payload(row, status="state_not_active")

    if row["worker_id"] != worker_id:
        return False, state_row_payload(row, status="held_by_other")

    lease_until = float(row["lease_until"] or 0)
    if lease_until <= now_ts():
        return False, state_row_payload(row, status="state_lease_expired")

    return True, state_row_payload(row, status="active")


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="MetaMIRAGE Preload Coordinator",
    version=API_VERSION,
    description=(
        "Control-plane coordinator for concurrent MetaMIRAGE preload workers. "
        "Workers send vector data directly to Qdrant; this service only manages "
        "state leases, global dedupe claims, and completion status."
    ),
)


@app.get("/health")
def health() -> Dict[str, Any]:
    with get_db() as conn:
        conn.execute("SELECT 1").fetchone()
        state_rows = conn.execute(
            "SELECT COUNT(*) FROM state_leases"
        ).fetchone()[0]
        content_rows = conn.execute(
            "SELECT COUNT(*) FROM content_claims"
        ).fetchone()[0]

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "time": utc_now(),
        "database": str(DB_PATH),
        "state_records": int(state_rows),
        "content_claim_records": int(content_rows),
    }


# ============================================================================
# State lease endpoints
# ============================================================================

@app.post("/state/acquire")
def state_acquire(req: StateAcquireRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    state_name = normalize_nonempty(req.state_name, "state_name")
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    lease_seconds = normalized_lease_seconds(
        req.lease_seconds,
        DEFAULT_STATE_LEASE_SECONDS,
    )

    now = now_ts()
    lease_until = now + lease_seconds

    with immediate_transaction() as conn:
        row = fetch_state(conn, build_id, state_code)

        if row is None:
            conn.execute(
                """
                INSERT INTO state_leases(
                    build_id, state_code, wave_id, state_name, worker_id, status,
                    acquired_at, last_heartbeat, lease_until,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    state_code,
                    wave_id,
                    state_name,
                    worker_id,
                    now,
                    now,
                    lease_until,
                    now,
                    now,
                ),
            )
            audit(
                conn,
                "state_acquired",
                build_id=build_id,
                wave_id=wave_id,
                state_code=state_code,
                worker_id=worker_id,
            )
            row = fetch_state(conn, build_id, state_code)
            payload = state_row_payload(row, status="acquired")
            payload["takeover"] = False
            return payload

        mismatch = validate_wave_binding(row, wave_id)
        if mismatch:
            return mismatch

        if row["status"] == "complete":
            return state_row_payload(row, status="already_complete")

        if row["status"] == "active":
            current_lease_until = float(row["lease_until"] or 0)

            if row["worker_id"] == worker_id:
                # Same live session asking again: idempotently renew.
                conn.execute(
                    """
                    UPDATE state_leases
                    SET state_name=?,
                        last_heartbeat=?,
                        lease_until=?,
                        updated_at=?
                    WHERE build_id=? AND state_code=?
                    """,
                    (
                        state_name,
                        now,
                        lease_until,
                        now,
                        build_id,
                        state_code,
                    ),
                )
                audit(
                    conn,
                    "state_reacquire_same_worker",
                    build_id=build_id,
                    wave_id=wave_id,
                    state_code=state_code,
                    worker_id=worker_id,
                )
                row = fetch_state(conn, build_id, state_code)
                return state_row_payload(row, status="already_owned")

            if current_lease_until > now:
                payload = state_row_payload(row, status="held_by_other")
                payload["detail"] = "Another worker currently owns the state lease."
                return payload

            # Expired state lease: safe takeover by the new worker.
            previous_worker = row["worker_id"]
            conn.execute(
                """
                UPDATE state_leases
                SET state_name=?,
                    worker_id=?,
                    status='active',
                    acquired_at=?,
                    last_heartbeat=?,
                    lease_until=?,
                    completed_at=NULL,
                    released_at=NULL,
                    release_reason=NULL,
                    manifest_path=NULL,
                    manifest_sha256=NULL,
                    updated_at=?
                WHERE build_id=? AND state_code=?
                """,
                (
                    state_name,
                    worker_id,
                    now,
                    now,
                    lease_until,
                    now,
                    build_id,
                    state_code,
                ),
            )
            audit(
                conn,
                "state_takeover_expired",
                build_id=build_id,
                wave_id=wave_id,
                state_code=state_code,
                worker_id=worker_id,
                detail=f"previous_worker={previous_worker}",
            )
            row = fetch_state(conn, build_id, state_code)
            payload = state_row_payload(row, status="acquired")
            payload["takeover"] = True
            payload["previous_worker_id"] = previous_worker
            return payload

        # released / failed-like state: reacquire immediately.
        previous_worker = row["worker_id"]
        conn.execute(
            """
            UPDATE state_leases
            SET state_name=?,
                worker_id=?,
                status='active',
                acquired_at=?,
                last_heartbeat=?,
                lease_until=?,
                completed_at=NULL,
                released_at=NULL,
                release_reason=NULL,
                manifest_path=NULL,
                manifest_sha256=NULL,
                updated_at=?
            WHERE build_id=? AND state_code=?
            """,
            (
                state_name,
                worker_id,
                now,
                now,
                lease_until,
                now,
                build_id,
                state_code,
            ),
        )
        audit(
            conn,
            "state_reacquired",
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
            detail=f"previous_status={row['status']}; previous_worker={previous_worker}",
        )
        row = fetch_state(conn, build_id, state_code)
        payload = state_row_payload(row, status="acquired")
        payload["takeover"] = True
        payload["previous_worker_id"] = previous_worker
        return payload


@app.post("/state/heartbeat")
def state_heartbeat(req: StateHeartbeatRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    lease_seconds = normalized_lease_seconds(
        req.lease_seconds,
        DEFAULT_STATE_LEASE_SECONDS,
    )

    now = now_ts()
    lease_until = now + lease_seconds

    with immediate_transaction() as conn:
        row = fetch_state(conn, build_id, state_code)

        if row is None:
            return {
                "status": "not_found",
                "build_id": build_id,
                "wave_id": wave_id,
                "state_code": state_code,
            }

        mismatch = validate_wave_binding(row, wave_id)
        if mismatch:
            return mismatch

        if row["status"] == "complete":
            return state_row_payload(row, status="already_complete")

        if row["status"] != "active":
            return state_row_payload(row, status="state_not_active")

        if row["worker_id"] != worker_id:
            return state_row_payload(row, status="held_by_other")

        # Do not resurrect an expired state lease. Another worker may legitimately
        # acquire it immediately after expiry.
        if float(row["lease_until"] or 0) <= now:
            return state_row_payload(row, status="lease_expired")

        conn.execute(
            """
            UPDATE state_leases
            SET last_heartbeat=?, lease_until=?, updated_at=?
            WHERE build_id=? AND state_code=?
            """,
            (now, lease_until, now, build_id, state_code),
        )
        row = fetch_state(conn, build_id, state_code)
        return state_row_payload(row, status="renewed")


@app.post("/state/release")
def state_release(req: StateReleaseRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    reason = str(req.reason or "worker_release").strip()

    now = now_ts()

    with immediate_transaction() as conn:
        row = fetch_state(conn, build_id, state_code)

        if row is None:
            return {
                "status": "not_found",
                "build_id": build_id,
                "wave_id": wave_id,
                "state_code": state_code,
            }

        mismatch = validate_wave_binding(row, wave_id)
        if mismatch:
            return mismatch

        if row["status"] == "complete":
            return state_row_payload(row, status="already_complete")

        if row["worker_id"] != worker_id:
            return state_row_payload(row, status="not_owner")

        if row["status"] == "released":
            return state_row_payload(row, status="released")

        conn.execute(
            """
            UPDATE state_leases
            SET status='released',
                lease_until=?,
                released_at=?,
                release_reason=?,
                updated_at=?
            WHERE build_id=? AND state_code=?
            """,
            (now, now, reason, now, build_id, state_code),
        )
        audit(
            conn,
            "state_released",
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
            detail=reason,
        )
        row = fetch_state(conn, build_id, state_code)
        return state_row_payload(row, status="released")


@app.post("/state/complete")
def state_complete(req: StateCompleteRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    state_name = normalize_nonempty(req.state_name, "state_name")
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    manifest_path = normalize_nonempty(req.manifest_path, "manifest_path")
    manifest_sha256 = normalize_nonempty(req.manifest_sha256, "manifest_sha256")

    now = now_ts()

    with immediate_transaction() as conn:
        row = fetch_state(conn, build_id, state_code)

        if row is None:
            return {
                "status": "not_found",
                "build_id": build_id,
                "wave_id": wave_id,
                "state_code": state_code,
            }

        mismatch = validate_wave_binding(row, wave_id)
        if mismatch:
            return mismatch

        if row["status"] == "complete":
            if (
                row["manifest_sha256"]
                and row["manifest_sha256"] != manifest_sha256
            ):
                payload = state_row_payload(row, status="completion_conflict")
                payload["requested_manifest_sha256"] = manifest_sha256
                payload["detail"] = (
                    "State is already complete with a different manifest hash."
                )
                return payload
            return state_row_payload(row, status="already_complete")

        if row["status"] != "active":
            return state_row_payload(row, status="state_not_active")

        if row["worker_id"] != worker_id:
            return state_row_payload(row, status="held_by_other")

        if float(row["lease_until"] or 0) <= now:
            return state_row_payload(row, status="lease_expired")

        conn.execute(
            """
            UPDATE state_leases
            SET state_name=?,
                status='complete',
                completed_at=?,
                lease_until=?,
                manifest_path=?,
                manifest_sha256=?,
                updated_at=?
            WHERE build_id=? AND state_code=?
            """,
            (
                state_name,
                now,
                now,
                manifest_path,
                manifest_sha256,
                now,
                build_id,
                state_code,
            ),
        )
        audit(
            conn,
            "state_completed",
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
            detail=f"manifest_sha256={manifest_sha256}",
        )
        row = fetch_state(conn, build_id, state_code)
        return state_row_payload(row, status="complete")


@app.get("/state/status")
def state_status(
    build_id: str = Query(...),
    state_code: str = Query(...),
) -> Dict[str, Any]:
    build_id = normalize_nonempty(build_id, "build_id")
    state_code = normalize_state_code(state_code)

    with get_db() as conn:
        row = fetch_state(conn, build_id, state_code)

    if row is None:
        return {
            "status": "not_found",
            "build_id": build_id,
            "state_code": state_code,
        }

    return state_row_payload(row)


# ============================================================================
# Content claim endpoints
# ============================================================================

@app.post("/content/claim")
def content_claim(req: ContentClaimRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    scope = normalize_scope(req.scope)
    content_hash = normalize_nonempty(req.content_hash, "content_hash")
    resource_id = str(req.resource_id).strip() if req.resource_id is not None else None
    lease_seconds = normalized_lease_seconds(
        req.lease_seconds,
        DEFAULT_CONTENT_LEASE_SECONDS,
    )

    now = now_ts()
    lease_until = now + lease_seconds

    with immediate_transaction() as conn:
        owns_state, state_info = active_state_owner(
            conn,
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
        )
        if not owns_state:
            payload = {
                "status": "state_not_owned",
                "build_id": build_id,
                "wave_id": wave_id,
                "state_code": state_code,
                "scope": scope,
                "content_hash": content_hash,
                "state": state_info,
            }
            return payload

        row = fetch_content_claim(conn, build_id, scope, content_hash)

        if row is None:
            conn.execute(
                """
                INSERT INTO content_claims(
                    build_id, scope, content_hash,
                    wave_id, owner_state, worker_id, status, resource_id,
                    claimed_at, last_heartbeat, lease_until,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    scope,
                    content_hash,
                    wave_id,
                    state_code,
                    worker_id,
                    resource_id,
                    now,
                    now,
                    lease_until,
                    now,
                    now,
                ),
            )
            audit(
                conn,
                "content_claimed",
                build_id=build_id,
                wave_id=wave_id,
                state_code=state_code,
                worker_id=worker_id,
                scope=scope,
                content_hash=content_hash,
            )
            row = fetch_content_claim(conn, build_id, scope, content_hash)
            payload = content_row_payload(row, status="claimed")
            payload["takeover"] = False
            return payload

        if row["status"] == "complete":
            return content_row_payload(row, status="already_complete")

        if row["status"] == "claimed":
            current_lease_until = float(row["lease_until"] or 0)

            if (
                row["owner_state"] == state_code
                and row["worker_id"] == worker_id
            ):
                # Same worker retry: renew claim.
                stored_resource = row["resource_id"] or resource_id
                conn.execute(
                    """
                    UPDATE content_claims
                    SET last_heartbeat=?,
                        lease_until=?,
                        resource_id=?,
                        updated_at=?
                    WHERE build_id=? AND scope=? AND content_hash=?
                    """,
                    (
                        now,
                        lease_until,
                        stored_resource,
                        now,
                        build_id,
                        scope,
                        content_hash,
                    ),
                )
                row = fetch_content_claim(conn, build_id, scope, content_hash)
                return content_row_payload(row, status="already_owned")

            if row["owner_state"] == state_code:
                # A new notebook may have acquired the STATE lease after the old
                # notebook died/released/expired. The state lease is authoritative:
                # transfer this state's still-open content claims immediately.
                previous_worker = row["worker_id"]
                stored_resource = row["resource_id"] or resource_id
                conn.execute(
                    """
                    UPDATE content_claims
                    SET wave_id=?,
                        worker_id=?,
                        status='claimed',
                        resource_id=?,
                        claimed_at=?,
                        last_heartbeat=?,
                        lease_until=?,
                        released_at=NULL,
                        release_reason=NULL,
                        updated_at=?
                    WHERE build_id=? AND scope=? AND content_hash=?
                    """,
                    (
                        wave_id,
                        worker_id,
                        stored_resource,
                        now,
                        now,
                        lease_until,
                        now,
                        build_id,
                        scope,
                        content_hash,
                    ),
                )
                audit(
                    conn,
                    "content_same_state_takeover",
                    build_id=build_id,
                    wave_id=wave_id,
                    state_code=state_code,
                    worker_id=worker_id,
                    scope=scope,
                    content_hash=content_hash,
                    detail=f"previous_worker={previous_worker}",
                )
                row = fetch_content_claim(conn, build_id, scope, content_hash)
                payload = content_row_payload(row, status="claimed")
                payload["takeover"] = True
                payload["previous_worker_id"] = previous_worker
                return payload

            if current_lease_until <= now:
                # Cross-state claim may be taken only after its lease expires.
                previous_state = row["owner_state"]
                previous_worker = row["worker_id"]
                conn.execute(
                    """
                    UPDATE content_claims
                    SET wave_id=?,
                        owner_state=?,
                        worker_id=?,
                        status='claimed',
                        resource_id=?,
                        claimed_at=?,
                        last_heartbeat=?,
                        lease_until=?,
                        completed_at=NULL,
                        released_at=NULL,
                        release_reason=NULL,
                        updated_at=?
                    WHERE build_id=? AND scope=? AND content_hash=?
                    """,
                    (
                        wave_id,
                        state_code,
                        worker_id,
                        resource_id,
                        now,
                        now,
                        lease_until,
                        now,
                        build_id,
                        scope,
                        content_hash,
                    ),
                )
                audit(
                    conn,
                    "content_takeover_expired",
                    build_id=build_id,
                    wave_id=wave_id,
                    state_code=state_code,
                    worker_id=worker_id,
                    scope=scope,
                    content_hash=content_hash,
                    detail=(
                        f"previous_state={previous_state}; "
                        f"previous_worker={previous_worker}"
                    ),
                )
                row = fetch_content_claim(conn, build_id, scope, content_hash)
                payload = content_row_payload(row, status="claimed")
                payload["takeover"] = True
                payload["previous_owner_state"] = previous_state
                payload["previous_worker_id"] = previous_worker
                return payload

            return content_row_payload(row, status="claimed_by_other")

        # released (or any future non-complete, non-claimed state) is claimable.
        previous_state = row["owner_state"]
        previous_worker = row["worker_id"]
        conn.execute(
            """
            UPDATE content_claims
            SET wave_id=?,
                owner_state=?,
                worker_id=?,
                status='claimed',
                resource_id=?,
                claimed_at=?,
                last_heartbeat=?,
                lease_until=?,
                completed_at=NULL,
                released_at=NULL,
                release_reason=NULL,
                updated_at=?
            WHERE build_id=? AND scope=? AND content_hash=?
            """,
            (
                wave_id,
                state_code,
                worker_id,
                resource_id,
                now,
                now,
                lease_until,
                now,
                build_id,
                scope,
                content_hash,
            ),
        )
        audit(
            conn,
            "content_reclaimed_released",
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
            scope=scope,
            content_hash=content_hash,
            detail=(
                f"previous_state={previous_state}; "
                f"previous_worker={previous_worker}"
            ),
        )
        row = fetch_content_claim(conn, build_id, scope, content_hash)
        payload = content_row_payload(row, status="claimed")
        payload["takeover"] = True
        return payload


@app.post("/content/heartbeat")
def content_heartbeat(req: ContentHeartbeatRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    scope = normalize_scope(req.scope)
    content_hash = normalize_nonempty(req.content_hash, "content_hash")
    lease_seconds = normalized_lease_seconds(
        req.lease_seconds,
        DEFAULT_CONTENT_LEASE_SECONDS,
    )

    now = now_ts()
    lease_until = now + lease_seconds

    with immediate_transaction() as conn:
        owns_state, state_info = active_state_owner(
            conn,
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
        )
        if not owns_state:
            return {
                "status": "state_not_owned",
                "build_id": build_id,
                "wave_id": wave_id,
                "state_code": state_code,
                "scope": scope,
                "content_hash": content_hash,
                "state": state_info,
            }

        row = fetch_content_claim(conn, build_id, scope, content_hash)

        if row is None:
            return {
                "status": "not_found",
                "build_id": build_id,
                "scope": scope,
                "content_hash": content_hash,
            }

        if row["status"] == "complete":
            return content_row_payload(row, status="already_complete")

        if row["status"] != "claimed":
            return content_row_payload(row, status="not_active")

        if row["owner_state"] != state_code or row["worker_id"] != worker_id:
            return content_row_payload(row, status="claimed_by_other")

        # Unlike the state lease, a content lease may be renewed after its nominal
        # expiry IF this worker still owns the active state lease and nobody has
        # taken over the content row. The state lease prevents zombie workers from
        # resurrecting abandoned claims.
        conn.execute(
            """
            UPDATE content_claims
            SET last_heartbeat=?,
                lease_until=?,
                updated_at=?
            WHERE build_id=? AND scope=? AND content_hash=?
            """,
            (
                now,
                lease_until,
                now,
                build_id,
                scope,
                content_hash,
            ),
        )
        row = fetch_content_claim(conn, build_id, scope, content_hash)
        return content_row_payload(row, status="renewed")


@app.post("/content/release")
def content_release(req: ContentReleaseRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    scope = normalize_scope(req.scope)
    content_hash = normalize_nonempty(req.content_hash, "content_hash")
    reason = str(req.reason or "worker_release").strip()

    now = now_ts()

    with immediate_transaction() as conn:
        row = fetch_content_claim(conn, build_id, scope, content_hash)

        if row is None:
            return {
                "status": "not_found",
                "build_id": build_id,
                "scope": scope,
                "content_hash": content_hash,
            }

        if row["status"] == "complete":
            return content_row_payload(row, status="already_complete")

        if (
            row["owner_state"] != state_code
            or row["worker_id"] != worker_id
        ):
            return content_row_payload(row, status="not_owner")

        if row["status"] == "released":
            return content_row_payload(row, status="released")

        conn.execute(
            """
            UPDATE content_claims
            SET status='released',
                lease_until=?,
                released_at=?,
                release_reason=?,
                updated_at=?
            WHERE build_id=? AND scope=? AND content_hash=?
            """,
            (
                now,
                now,
                reason,
                now,
                build_id,
                scope,
                content_hash,
            ),
        )
        audit(
            conn,
            "content_released",
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
            scope=scope,
            content_hash=content_hash,
            detail=reason,
        )
        row = fetch_content_claim(conn, build_id, scope, content_hash)
        return content_row_payload(row, status="released")


@app.post("/content/complete")
def content_complete(req: ContentCompleteRequest) -> Dict[str, Any]:
    build_id = normalize_nonempty(req.build_id, "build_id")
    wave_id = normalize_nonempty(req.wave_id, "wave_id")
    state_code = normalize_state_code(req.state_code)
    worker_id = normalize_nonempty(req.worker_id, "worker_id")
    scope = normalize_scope(req.scope)
    content_hash = normalize_nonempty(req.content_hash, "content_hash")
    resource_id = normalize_nonempty(req.resource_id, "resource_id")

    now = now_ts()

    with immediate_transaction() as conn:
        row = fetch_content_claim(conn, build_id, scope, content_hash)

        if row is None:
            return {
                "status": "not_found",
                "build_id": build_id,
                "scope": scope,
                "content_hash": content_hash,
            }

        if row["status"] == "complete":
            # Completion is idempotent. The authoritative canonical resource ID is
            # returned so workers can distinguish crash recovery from a true duplicate.
            return content_row_payload(row, status="already_complete")

        owns_state, state_info = active_state_owner(
            conn,
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
        )
        if not owns_state:
            return {
                "status": "state_not_owned",
                "build_id": build_id,
                "wave_id": wave_id,
                "state_code": state_code,
                "scope": scope,
                "content_hash": content_hash,
                "state": state_info,
            }

        if row["status"] != "claimed":
            return content_row_payload(row, status="not_active")

        if (
            row["owner_state"] != state_code
            or row["worker_id"] != worker_id
        ):
            return content_row_payload(row, status="claimed_by_other")

        if row["resource_id"] and row["resource_id"] != resource_id:
            payload = content_row_payload(row, status="resource_conflict")
            payload["requested_resource_id"] = resource_id
            payload["detail"] = (
                "The active claim was created for a different resource_id."
            )
            return payload

        conn.execute(
            """
            UPDATE content_claims
            SET status='complete',
                resource_id=?,
                completed_at=?,
                lease_until=?,
                updated_at=?
            WHERE build_id=? AND scope=? AND content_hash=?
            """,
            (
                resource_id,
                now,
                now,
                now,
                build_id,
                scope,
                content_hash,
            ),
        )
        audit(
            conn,
            "content_completed",
            build_id=build_id,
            wave_id=wave_id,
            state_code=state_code,
            worker_id=worker_id,
            scope=scope,
            content_hash=content_hash,
            detail=f"resource_id={resource_id}",
        )
        row = fetch_content_claim(conn, build_id, scope, content_hash)
        return content_row_payload(row, status="complete")


@app.get("/content/status")
def content_status(
    build_id: str = Query(...),
    scope: Literal["document", "rag_chunk"] = Query(...),
    content_hash: str = Query(...),
) -> Dict[str, Any]:
    build_id = normalize_nonempty(build_id, "build_id")
    scope = normalize_scope(scope)
    content_hash = normalize_nonempty(content_hash, "content_hash")

    with get_db() as conn:
        row = fetch_content_claim(conn, build_id, scope, content_hash)

    if row is None:
        return {
            "status": "not_found",
            "build_id": build_id,
            "scope": scope,
            "content_hash": content_hash,
        }

    return content_row_payload(row)


# ============================================================================
# Wave status
# ============================================================================

@app.get("/wave/status")
def wave_status(
    build_id: str = Query(...),
    wave_id: str = Query(...),
) -> Dict[str, Any]:
    build_id = normalize_nonempty(build_id, "build_id")
    wave_id = normalize_nonempty(wave_id, "wave_id")

    now = now_ts()

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM state_leases
            WHERE build_id=? AND wave_id=?
            ORDER BY state_code
            """,
            (build_id, wave_id),
        ).fetchall()

        content_counts_rows = conn.execute(
            """
            SELECT scope, status, COUNT(*) AS n
            FROM content_claims
            WHERE build_id=? AND wave_id=?
            GROUP BY scope, status
            ORDER BY scope, status
            """,
            (build_id, wave_id),
        ).fetchall()

    states = []
    counts: Dict[str, int] = {}

    for row in rows:
        item = state_row_payload(row)
        if (
            row["status"] == "active"
            and row["lease_until"] is not None
            and float(row["lease_until"]) <= now
        ):
            effective = "expired"
        else:
            effective = row["status"]

        item["effective_status"] = effective
        states.append(item)
        counts[effective] = counts.get(effective, 0) + 1

    content_counts: Dict[str, Dict[str, int]] = {}
    for row in content_counts_rows:
        content_counts.setdefault(row["scope"], {})[row["status"]] = int(row["n"])

    return {
        "status": "ok",
        "build_id": build_id,
        "wave_id": wave_id,
        "state_count": len(states),
        "state_status_counts": counts,
        "states": states,
        "content_status_counts": content_counts,
        "all_registered_states_complete": bool(states)
        and all(s["effective_status"] == "complete" for s in states),
        "time": utc_now(),
    }


# ============================================================================
# Optional command-line launch
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("METAMIRAGE_COORDINATOR_HOST", "0.0.0.0")
    port = int(os.environ.get("METAMIRAGE_COORDINATOR_PORT", "8001"))

    print(f"{SERVICE_NAME} {API_VERSION}")
    print(f"Database: {DB_PATH}")
    print(f"Listening on http://{host}:{port}")
    print("Run one coordinator process for the shared preload build.")

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )
