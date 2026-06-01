"""Configuration for Telegram bot."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TelegramBotConfig:
    """Configuration for Telegram bot runtime."""

    token: str
    meta_agent_api_url: str
    poll_timeout: int = 300
    request_timeout: float = 300.0
    session_db_path: Path | str = "data/telegram_sessions.sqlite3"
    thread_scope: str = "chat"  # "chat" or "user"

    @classmethod
    def from_env(cls) -> "TelegramBotConfig":
        """Load configuration from environment variables."""

        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        api_url = os.getenv("META_AGENT_API_URL", "http://localhost:8000")
        poll_timeout = int(os.getenv("TELEGRAM_POLL_TIMEOUT", "300"))
        request_timeout = float(os.getenv("TELEGRAM_REQUEST_TIMEOUT", "300.0"))
        session_db = os.getenv(
            "TELEGRAM_SESSION_DB_PATH", "data/telegram_sessions.sqlite3"
        )
        thread_scope = os.getenv("TELEGRAM_THREAD_SCOPE", "chat")

        return cls(
            token=token,
            meta_agent_api_url=api_url,
            poll_timeout=poll_timeout,
            request_timeout=request_timeout,
            session_db_path=session_db,
            thread_scope=thread_scope,
        )
