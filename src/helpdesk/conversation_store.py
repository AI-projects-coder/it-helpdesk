"""Conversations: an ordered list of turns, where each turn is one helpdesk session.

Stored in SQLite so the list survives restarts. The full per-turn record
(tool I/O, decisions) stays in SessionStore; this only indexes and orders it.
Swap this class for a Postgres/Firestore implementation in production.
"""
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    customer_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    seq INTEGER NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    query TEXT NOT NULL,
    customer_id TEXT,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT,
    PRIMARY KEY (conversation_id, seq)
);
"""

TITLE_LEN = 60


@dataclass
class Turn:
    session_id: str
    seq: int
    query: str
    customer_id: str | None
    role: str
    status: str
    outcome: str | None


@dataclass
class Conversation:
    id: str
    title: str
    customer_id: str | None
    created_at: str
    updated_at: str
    turns: list[Turn] = field(default_factory=list)
    turn_count: int = 0
    pending: bool = False  # a turn is waiting for human approval


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def create(self, first_query: str, customer_id: str | None) -> str:
        cid = uuid.uuid4().hex[:12]
        title = first_query.strip().splitlines()[0][:TITLE_LEN] if first_query.strip() else "New conversation"
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO conversations VALUES (?, ?, ?, ?, ?)", (cid, title, customer_id, _now(), _now())
            )
        return cid

    def add_turn(self, conversation_id: str, session_id: str, query: str, customer_id: str | None, role: str) -> None:
        with self._lock, self._conn:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM turns WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()[0]
            self._conn.execute(
                "INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, 'running', NULL)",
                (conversation_id, seq, session_id, query, customer_id, role),
            )
            self._touch(conversation_id)

    def update_turn(self, session_id: str, status: str, outcome: str | None) -> None:
        with self._lock, self._conn:
            row = self._conn.execute("SELECT conversation_id FROM turns WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return
            self._conn.execute("UPDATE turns SET status = ?, outcome = ? WHERE session_id = ?",
                               (status, outcome, session_id))
            self._touch(row["conversation_id"])

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row is None:
                return None
            turns = [Turn(**{k: t[k] for k in Turn.__dataclass_fields__}) for t in self._conn.execute(
                "SELECT * FROM turns WHERE conversation_id = ? ORDER BY seq", (conversation_id,))]
        return Conversation(**dict(row), turns=turns, turn_count=len(turns),
                            pending=any(t.status == "awaiting_approval" for t in turns))

    def recent(self, limit: int = 20) -> list[Conversation]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT c.*, COUNT(t.seq) AS turn_count,
                          SUM(t.status = 'awaiting_approval') > 0 AS pending
                   FROM conversations c LEFT JOIN turns t ON t.conversation_id = c.id
                   GROUP BY c.id ORDER BY c.updated_at DESC, c.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [Conversation(**{k: r[k] for k in ("id", "title", "customer_id", "created_at", "updated_at")},
                             turn_count=r["turn_count"], pending=bool(r["pending"])) for r in rows]

    def _touch(self, conversation_id: str) -> None:
        self._conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id))
