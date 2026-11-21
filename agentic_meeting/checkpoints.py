"""Checkpoints: where a running meeting is saved between steps.

A meeting paused for a human answer is resumed from its checkpoint, also from
another process or after a restart when SqliteCheckpoints is used. This is
execution state; completed meetings and project knowledge live in
MeetingStore, which is separate.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class CheckpointStore(Protocol):
    def save(self, thread_id: str, state: dict, next_step: str) -> None: ...

    def load(self, thread_id: str) -> tuple[dict, str] | None:
        """Return (state, next_step), or None for an unknown thread."""


class MemoryCheckpoints:
    """Checkpoints kept in memory, for one process."""

    def __init__(self):
        self._data: dict[str, str] = {}

    def save(self, thread_id: str, state: dict, next_step: str) -> None:
        # Stored as JSON so later changes to the state cannot leak in.
        self._data[thread_id] = json.dumps(
            {"state": state, "next_step": next_step}, ensure_ascii=False
        )

    def load(self, thread_id: str) -> tuple[dict, str] | None:
        if thread_id not in self._data:
            return None
        data = json.loads(self._data[thread_id])
        return data["state"], data["next_step"]


class SqliteCheckpoints:
    """Checkpoints in a SQLite file, surviving restarts."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    next_step TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, thread_id: str, state: dict, next_step: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (thread_id, next_step, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    next_step = excluded.next_step,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    next_step,
                    json.dumps(state, ensure_ascii=False, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load(self, thread_id: str) -> tuple[dict, str] | None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state_json, next_step FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

        if row is None:
            return None

        return json.loads(row[0]), row[1]
