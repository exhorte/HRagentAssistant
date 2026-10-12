"""SQLite persistence for business data (users, documents registry, leave requests, escalations).

One small file-based database is enough for the prototype. The vector index
(Qdrant) only stores searchable passages; the *source of truth* for documents
and workflows is here.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    username      TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('employee', 'hr_manager')),
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    name            TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    language        TEXT,
    version         TEXT,
    family          TEXT,
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'obsolete', 'draft')),
    access_level    TEXT NOT NULL DEFAULT 'all' CHECK (access_level IN ('all', 'hr_only')),
    is_demo         INTEGER NOT NULL DEFAULT 0,
    imported_at     TEXT NOT NULL,
    imported_by     TEXT,
    stored_path     TEXT,
    indexing_status TEXT NOT NULL DEFAULT 'pending' CHECK (indexing_status IN ('pending', 'indexed', 'failed')),
    indexing_error  TEXT,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    page_count      INTEGER NOT NULL DEFAULT 0,
    collection      TEXT,
    UNIQUE (tenant_id, sha256)
);

CREATE TABLE IF NOT EXISTS leave_requests (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL REFERENCES users(id),
    request_type    TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    reason          TEXT,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    submitted_at    TEXT,
    decided_by      TEXT,
    decided_at      TEXT,
    decision_comment TEXT
);
CREATE INDEX IF NOT EXISTS ix_leave_user ON leave_requests(tenant_id, user_id);

CREATE TABLE IF NOT EXISTS leave_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL REFERENCES leave_requests(id),
    at          TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    comment     TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    user_id       TEXT NOT NULL REFERENCES users(id),
    question      TEXT NOT NULL,
    summary       TEXT NOT NULL,
    reason        TEXT NOT NULL,
    context_json  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved')),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    handled_by    TEXT,
    resolution_note TEXT
);
CREATE INDEX IF NOT EXISTS ix_escalations_user ON escalations(tenant_id, user_id);
"""


class Database:
    """Thin wrapper around a single SQLite connection (thread-safe via a lock).

    ``path=None`` or ``":memory:"`` gives an ephemeral database (used by tests).
    """

    def __init__(self, path: str | Path | None = None):
        self._path = str(path) if path else ":memory:"
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialised transaction: commit on success, roll back on any exception."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple | dict = ()) -> None:
        with self.transaction() as conn:
            conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
