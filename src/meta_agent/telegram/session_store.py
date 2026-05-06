"""SQLite-backed session store for Telegram with multiple sessions per user."""

import sqlite3
from pathlib import Path
from typing import Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from src.meta_agent.utils.thread_ids import generate_thread_id

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a user's meta-agent session."""

    thread_id: str  # Primary identifier - the actual meta-agent thread
    user_key: str
    name: str
    is_active: bool
    has_messages: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: tuple) -> "Session":
        """Create Session from database row."""
        return cls(
            thread_id=row[0],
            user_key=row[1],
            name=row[2],
            is_active=bool(row[3]),
            has_messages=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
        )


class TelegramSessionStore:
    """Store multiple meta-agent sessions per user with switching capability."""

    def __init__(self, db_path: Path | str):
        """Initialize session store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn_cache: Optional[sqlite3.Connection] = None
        self._init_db()

    def __enter__(self) -> "TelegramSessionStore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        self.close()
        return None

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    user_key TEXT NOT NULL,
                    name TEXT NOT NULL,
                    is_active INTEGER DEFAULT 0,
                    has_messages BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(user_key, name)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_active
                ON sessions(user_key, is_active)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _get_user_key(self, chat_id: int, user_id: int) -> str:
        """Get unique user key for storage.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.

        Returns:
            Unique user key.
        """
        return f"chat_{chat_id}_user_{user_id}"

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.isolation_level = None  # Autocommit mode
        return conn

    def close(self) -> None:
        """Close any cached database connection."""
        if self._conn_cache is not None:
            self._conn_cache.close()
            self._conn_cache = None

    def __del__(self) -> None:
        """Ensure database connection is closed when object is destroyed."""
        try:
            self.close()
        except Exception:
            pass

    def get_active_thread_id(self, chat_id: int, user_id: int) -> Optional[str]:
        """Get active thread ID for a user.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.

        Returns:
            Thread ID or None if no active session.
        """
        user_key = self._get_user_key(chat_id, user_id)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT thread_id FROM sessions WHERE user_key = ? AND is_active = 1",
                (user_key,),
            )
            row = cursor.fetchone()
            return row["thread_id"] if row else None
        finally:
            conn.close()

    def get_active_session(self, chat_id: int, user_id: int) -> Optional[Session]:
        """Get active session for a user.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.

        Returns:
            Session or None if no active session.
        """
        user_key = self._get_user_key(chat_id, user_id)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE user_key = ? AND is_active = 1",
                (user_key,),
            )
            row = cursor.fetchone()
            return Session.from_row(tuple(row)) if row else None
        finally:
            conn.close()

    def create_session(
        self, chat_id: int, user_id: int, thread_id: str, name: str = "default", activate: bool = True
    ) -> Session:
        """Create a new session.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.
            thread_id: Meta-agent thread ID. If not provided, generates a new UUID.
            name: Session name.
            activate: Whether to make this the active session.

        Returns:
            Created Session.
        """
        user_key = self._get_user_key(chat_id, user_id)

        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE user_key = ? AND name = ?",
                (user_key, name),
            )
            existing_row = cursor.fetchone()
            if existing_row:
                if activate:
                    conn.execute("UPDATE sessions SET is_active = 0 WHERE user_key = ?", (user_key,))
                    conn.execute(
                        """
                        UPDATE sessions
                        SET is_active = 1, updated_at = datetime('now')
                        WHERE user_key = ? AND name = ?
                        """,
                        (user_key, name),
                    )
                    conn.commit()
                    cursor = conn.execute(
                        "SELECT * FROM sessions WHERE user_key = ? AND name = ?",
                        (user_key, name),
                    )
                    existing_row = cursor.fetchone()
                return Session.from_row(tuple(existing_row))

            # If no thread_id provided, generate a new one
            if not thread_id:
                thread_id = generate_thread_id()

            if activate:
                # Deactivate all other sessions for this user
                conn.execute(
                    "UPDATE sessions SET is_active = 0 WHERE user_key = ?",
                    (user_key,),
                )

            conn.execute(
                """
                INSERT INTO sessions (thread_id, user_key, name, is_active, has_messages)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, user_key, name, 1 if activate else 0, 0),
            )
            conn.commit()

            # Fetch the created session
            cursor = conn.execute("SELECT * FROM sessions WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()
            return Session.from_row(tuple(row))
        finally:
            conn.close()

    def switch_session(self, chat_id: int, user_id: int, name: str) -> Optional[Session]:
        """Switch to a different session by name.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.
            name: Session name to switch to.

        Returns:
            Session or None if not found.
        """
        user_key = self._get_user_key(chat_id, user_id)

        conn = self._get_conn()
        try:
            # Check if session exists
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE user_key = ? AND name = ?",
                (user_key, name),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Deactivate all sessions
            conn.execute("UPDATE sessions SET is_active = 0 WHERE user_key = ?", (user_key,))

            # Activate target session
            conn.execute(
                """
                UPDATE sessions
                SET is_active = 1, updated_at = datetime('now')
                WHERE user_key = ? AND name = ?
                """,
                (user_key, name),
            )
            conn.commit()

            # Re-fetch the updated row to get correct is_active state
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE user_key = ? AND name = ?",
                (user_key, name),
            )
            updated_row = cursor.fetchone()
            return Session.from_row(tuple(updated_row)) if updated_row else None
        finally:
            conn.close()

    def list_sessions(self, chat_id: int, user_id: int) -> list[Session]:
        """List all sessions for a user.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.

        Returns:
            List of Sessions, ordered by updated_at descending.
        """
        user_key = self._get_user_key(chat_id, user_id)

        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE user_key = ? ORDER BY updated_at DESC",
                (user_key,),
            )
            return [Session.from_row(tuple(row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_session(self, chat_id: int, user_id: int, name: str) -> bool:
        """Delete a session by name.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.
            name: Session name to delete.

        Returns:
            True if deleted, False if not found.
        """
        user_key = self._get_user_key(chat_id, user_id)

        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE user_key = ? AND name = ?",
                (user_key, name),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def mark_session_has_messages(self, chat_id: int, user_id: int, thread_id: str) -> None:
        """Mark a session as having received messages.

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.
            thread_id: Session thread ID.
        """
        user_key = self._get_user_key(chat_id, user_id)

        conn = self._get_conn()
        try:
            conn.execute(
                """
                UPDATE sessions
                SET has_messages = 1, updated_at = datetime('now')
                WHERE thread_id = ? AND user_key = ?
                """,
                (thread_id, user_key),
            )
            conn.commit()
        finally:
            conn.close()

    def replace_thread_id(self, chat_id: int, user_id: int, old_thread_id: str, new_thread_id: str) -> None:
        """Replace session thread_id (e.g., replace placeholder with real).

        Args:
            chat_id: Telegram chat ID.
            user_id: Telegram user ID.
            old_thread_id: Current thread ID to replace.
            new_thread_id: New thread ID.
        """
        user_key = self._get_user_key(chat_id, user_id)

        conn = self._get_conn()
        try:
            # Get the old session first to preserve its name and state
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE thread_id = ? AND user_key = ?",
                (old_thread_id, user_key),
            )
            old_session = cursor.fetchone()
            if not old_session:
                return

            old_session_row = Session.from_row(tuple(old_session))

            # Delete old and insert new with same name and active status
            conn.execute("DELETE FROM sessions WHERE thread_id = ?", (old_thread_id,))
            conn.execute(
                """
                INSERT INTO sessions (thread_id, user_key, name, is_active, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_thread_id, user_key, old_session_row.name, old_session_row.is_active, old_session_row.created_at.isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
