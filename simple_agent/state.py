"""SQLite state persistence for autonomous agent loops.

Tracks three things: items the agent has seen (deduplication), drafts the
agent has generated (lifecycle management), and runs (observability).
Uses SQLite because it's stdlib, zero-config, and sufficient for
single-machine agents.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone


class StateStore:
    """Persistent state backed by SQLite."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new',
                score REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                persona TEXT NOT NULL,
                draft_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                FOREIGN KEY (item_id) REFERENCES items(id)
            );

            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                items_processed INTEGER NOT NULL DEFAULT 0,
                drafts_created INTEGER NOT NULL DEFAULT 0,
                errors TEXT NOT NULL DEFAULT '[]'
            );
        """)
        self._conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- Items ---

    def save_item(self, item_id: str, data: dict, score: float = 0.0) -> None:
        """Save or update an item. Skips silently if item_id already exists."""
        self._conn.execute(
            "INSERT OR IGNORE INTO items (id, data, score, created_at) VALUES (?, ?, ?, ?)",
            (item_id, json.dumps(data), score, self._now()),
        )
        self._conn.commit()

    def has_item(self, item_id: str) -> bool:
        """Check if an item has been seen before."""
        row = self._conn.execute(
            "SELECT 1 FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        return row is not None

    def get_pending_items(self) -> list[dict]:
        """Get items with status 'new' that haven't been drafted yet."""
        rows = self._conn.execute(
            "SELECT id, data, score FROM items WHERE status = 'new' ORDER BY score DESC"
        ).fetchall()
        return [
            {"id": r["id"], "data": json.loads(r["data"]), "score": r["score"]}
            for r in rows
        ]

    def update_item_status(self, item_id: str, status: str) -> None:
        """Update an item's status (e.g., 'new' -> 'drafted')."""
        self._conn.execute(
            "UPDATE items SET status = ? WHERE id = ?", (status, item_id)
        )
        self._conn.commit()

    # --- Drafts ---

    def save_draft(self, item_id: str, persona: str, text: str) -> int:
        """Save a draft and return its ID."""
        cur = self._conn.execute(
            "INSERT INTO drafts (item_id, persona, draft_text, created_at) VALUES (?, ?, ?, ?)",
            (item_id, persona, text, self._now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_pending_drafts(self) -> list[dict]:
        """Get all drafts with 'pending' status."""
        rows = self._conn.execute(
            "SELECT id, item_id, persona, draft_text, created_at FROM drafts WHERE status = 'pending'"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_draft_status(self, draft_id: int, status: str) -> None:
        """Update a draft's status (e.g., 'pending' -> 'approved')."""
        self._conn.execute(
            "UPDATE drafts SET status = ? WHERE id = ?", (status, draft_id)
        )
        self._conn.commit()

    def expire_stale_drafts(self, hours: int = 48) -> int:
        """Expire pending drafts older than the given threshold. Returns count."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = self._conn.execute(
            "UPDATE drafts SET status = 'expired' WHERE status = 'pending' AND created_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    # --- Runs ---

    def start_run(self) -> int:
        """Log the start of a pipeline run. Returns the run ID."""
        cur = self._conn.execute(
            "INSERT INTO runs (started_at) VALUES (?)", (self._now(),)
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def finish_run(
        self,
        run_id: int,
        items_processed: int,
        drafts_created: int,
        errors: list[str] | None = None,
    ) -> None:
        """Log the completion of a pipeline run."""
        self._conn.execute(
            "UPDATE runs SET completed_at = ?, items_processed = ?, drafts_created = ?, errors = ? WHERE id = ?",
            (self._now(), items_processed, drafts_created, json.dumps(errors or []), run_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
