"""Tiny SQLite-backed cache with TTL.

Used by MCP servers that wrap external data sources so repeated dev/demo runs
do not hammer Yahoo Finance or news APIs. Values are JSON blobs keyed by
(tool_name, args).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def _key(namespace: str, payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


class TTLCache:
    def __init__(self, db_path: Path, ttl_seconds: int = 900) -> None:
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: multiple server subprocesses share this DB; wait on locks.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, namespace: str, payload: Any) -> Any | None:
        k = _key(namespace, payload)
        cur = self._conn.execute("SELECT v, expires_at FROM cache WHERE k = ?", (k,))
        row = cur.fetchone()
        if row is None:
            return None
        value, expires_at = row
        if time.time() > expires_at:
            self._conn.execute("DELETE FROM cache WHERE k = ?", (k,))
            self._conn.commit()
            return None
        return json.loads(value)

    def set(self, namespace: str, payload: Any, value: Any, ttl_seconds: int | None = None) -> None:
        k = _key(namespace, payload)
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (k, v, expires_at) VALUES (?, ?, ?)",
            (k, json.dumps(value, default=str), time.time() + ttl),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
