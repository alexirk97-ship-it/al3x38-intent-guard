from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ShardedLockPool:
    """Pool of sharded async locks keyed by session_id (Python 3.9+ loop-safe)."""

    def __init__(self, shards: int = 256):
        self.shards = max(1, shards)
        self._guard = threading.Lock()
        self._loops: dict[int, list[asyncio.Lock]] = {}

    def get_lock(self, session_id: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=4).digest()
        idx = int.from_bytes(digest, "big") % self.shards

        with self._guard:
            pool = self._loops.get(loop_id)
            if pool is None:
                pool = [asyncio.Lock() for _ in range(self.shards)]
                self._loops[loop_id] = pool
                if len(self._loops) > 16:
                    oldest_key = next(iter(self._loops))
                    if oldest_key != loop_id:
                        self._loops.pop(oldest_key, None)
            return pool[idx]


@runtime_checkable
class StateStore(Protocol):
    async def load_session(self, session_id: str) -> dict[str, Any] | None: ...
    async def save_session(self, session_id: str, payload: dict[str, Any]) -> None: ...
    async def delete_session(self, session_id: str) -> None: ...
    async def count_sessions(self) -> int: ...


class MemoryStateStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        raw = self._data.get(session_id)
        return dict(raw) if raw is not None else None

    async def save_session(self, session_id: str, payload: dict[str, Any]) -> None:
        self._data[session_id] = dict(payload)

    async def delete_session(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    async def count_sessions(self) -> int:
        return len(self._data)

class JsonFileStateStore:
    """Atomic JSON file store with configurable debounce flush."""

    def __init__(self, file_path: str | Path, flush_interval_sec: float = 0.0):
        self.path = Path(file_path)
        self.flush_interval_sec = max(0.0, float(flush_interval_sec))
        self._io_lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._last_flush = 0.0
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "sessions" in raw and isinstance(raw["sessions"], dict):
                    self._sessions = {str(k): dict(v) for k, v in raw["sessions"].items() if isinstance(v, dict)}
                else:
                    for k, v in raw.items():
                        if isinstance(v, (int, float)):
                            self._sessions[str(k)] = {"trust": float(v), "updated_at": time.time()}
        except (OSError, json.JSONDecodeError):
            self._sessions = {}

    def _write_atomic(self) -> None:
        with self._io_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": "1.2.0", "sessions": self._sessions}
            data = json.dumps(payload, ensure_ascii=False, indent=2)
            fd, tmp_name = tempfile.mkstemp(prefix=".ig_state_", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_name, self.path)
                self._dirty = False
                self._last_flush = time.monotonic()
            except Exception:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
                raise

    async def flush(self) -> None:
        if self._dirty:
            await asyncio.to_thread(self._write_atomic)

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        item = self._sessions.get(session_id)
        return dict(item) if item is not None else None

    async def save_session(self, session_id: str, payload: dict[str, Any]) -> None:
        self._sessions[session_id] = dict(payload)
        self._dirty = True
        now = time.monotonic()
        if self.flush_interval_sec <= 0.0 or (now - self._last_flush) >= self.flush_interval_sec:
            await asyncio.to_thread(self._write_atomic)

    async def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._dirty = True
            await asyncio.to_thread(self._write_atomic)

    async def count_sessions(self) -> int:
        return len(self._sessions)


class SQLStateStore:
    """Lightweight DB-API 2.0 store supporting SQLite, MySQL, and PostgreSQL."""

    def __init__(self, connect_fn: Callable[[], Any], table_name: str = "intent_guard_sessions", placeholder: str = "?"):
        if not table_name.replace("_", "").isalnum():
            raise ValueError("Invalid SQL table name")
        self.connect_fn = connect_fn
        self.table_name = table_name
        self.placeholder = placeholder
        self._db_lock = threading.Lock()
        self._init_table()

    @classmethod
    def from_sqlite(cls, db_path: str | Path, table_name: str = "intent_guard_sessions") -> SQLStateStore:
        target = str(db_path)
        def _conn():
            c = sqlite3.connect(target, timeout=30.0, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL;")
            return c
        return cls(connect_fn=_conn, table_name=table_name, placeholder="?")

    def _init_table(self) -> None:
        sql = (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            "session_id VARCHAR(191) PRIMARY KEY, payload_json TEXT NOT NULL, updated_at DOUBLE PRECISION NOT NULL)"
        )
        with self._db_lock:
            conn = self.connect_fn()
            try:
                cur = conn.cursor()
                cur.execute(sql)
                conn.commit()
            finally:
                conn.close()

    def _sync_load(self, session_id: str) -> dict[str, Any] | None:
        p = self.placeholder
        with self._db_lock:
            conn = self.connect_fn()
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT payload_json FROM {self.table_name} WHERE session_id = {p}", (session_id,))
                row = cur.fetchone()
                if not row or not row[0]:
                    return None
                data = json.loads(row[0])
                return dict(data) if isinstance(data, dict) else None
            finally:
                conn.close()

    def _sync_save(self, session_id: str, payload: dict[str, Any]) -> None:
        p = self.placeholder
        raw_json = json.dumps(payload, ensure_ascii=False)
        now = time.time()
        with self._db_lock:
            conn = self.connect_fn()
            try:
                cur = conn.cursor()
                cur.execute(f"DELETE FROM {self.table_name} WHERE session_id = {p}", (session_id,))
                cur.execute(
                    f"INSERT INTO {self.table_name} (session_id, payload_json, updated_at) VALUES ({p}, {p}, {p})",
                    (session_id, raw_json, now),
                )
                conn.commit()
            finally:
                conn.close()

    def _sync_delete(self, session_id: str) -> None:
        p = self.placeholder
        with self._db_lock:
            conn = self.connect_fn()
            try:
                cur = conn.cursor()
                cur.execute(f"DELETE FROM {self.table_name} WHERE session_id = {p}", (session_id,))
                conn.commit()
            finally:
                conn.close()

    def _sync_count(self) -> int:
        with self._db_lock:
            conn = self.connect_fn()
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row = cur.fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()

    async def load_session(self, session_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._sync_load, session_id)

    async def save_session(self, session_id: str, payload: dict[str, Any]) -> None:
        await asyncio.to_thread(self._sync_save, session_id, payload)

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._sync_delete, session_id)

    async def count_sessions(self) -> int:
        return await asyncio.to_thread(self._sync_count)