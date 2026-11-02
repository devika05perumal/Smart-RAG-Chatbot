"""SQLite persistence for conversations and messages."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


class ChatHistoryError(Exception):
    """Raised when the chat database cannot be used."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ChatHistoryDB:
    DEFAULT_TITLE = "New chat"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except (sqlite3.Error, OSError) as exc:
            raise ChatHistoryError(f"Could not open the chat history database at {self.path}: {exc}") from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise ChatHistoryError(f"Chat history database error: {exc}") from exc
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
                """
            )

    # ----------------------------------------------------- conversations
    def create_conversation(self, title: str = DEFAULT_TITLE) -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)", (title, now, now))
            return int(cur.lastrowid)

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_title(self, conversation_id: int) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return row["title"] if row else None

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    # ----------------------------------------------------------- messages
    def add_message(self, conversation_id: int, role: str, content: str,
                    sources: Optional[list[dict]] = None) -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(sources) if sources else None, now))
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            return int(cur.lastrowid)

    def get_messages(self, conversation_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, sources, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)).fetchall()
        messages = []
        for r in rows:
            item = dict(r)
            item["sources"] = json.loads(item["sources"]) if item["sources"] else []
            messages.append(item)
        return messages

    def delete_last_assistant_message(self, conversation_id: int) -> bool:
        """Remove the final message if it is an assistant reply (used by Regenerate)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, role FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
                (conversation_id,)).fetchone()
            if row and row["role"] == "assistant":
                conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
                return True
        return False

    def clear_conversation(self, conversation_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                         (self.DEFAULT_TITLE, _now(), conversation_id))
