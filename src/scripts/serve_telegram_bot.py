"""Telegram bot long-polling server.

Runs as a separate process from the meta-agent API.

Usage:
    python -m src.scripts.serve_telegram_bot

Requires environment variables:
    TELEGRAM_BOT_TOKEN
    META_AGENT_API_URL (default: http://localhost:8000)
"""

import asyncio
import logging
import mimetypes
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path so imports work when running as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta_agent.configs import CHARTS_DIR, TelegramBotConfig  # noqa: E402
from src.meta_agent.telegram.bot_client import TelegramBotClient  # noqa: E402
from src.meta_agent.telegram.meta_agent_client import MetaAgentClient  # noqa: E402
from src.meta_agent.telegram.session_store import TelegramSessionStore  # noqa: E402
from src.meta_agent.telegram.message_handler import MessageHandler  # noqa: E402
from src.meta_agent.telegram.update_parser import parse_update, parse_callback_query  # noqa: E402

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("telegram_bot")

ALLOWED_UPDATES = ["message", "callback_query"]


class LocalMetaAgentClient:
    """Dev-only meta-agent client that bypasses the HTTP API."""

    async def ask(self, question: str, thread_id: str | None = None):
        from src.meta_agent import MetaAgentApiResponse, meta_graph_manager

        result = await meta_graph_manager.invoke_graph_session(question, thread_id)
        return MetaAgentApiResponse(thread_id=result.thread_id, outputs=result.outputs)

    async def fetch_artifact_bytes(self, artifact_url: str) -> tuple[bytes, str, str]:
        if not artifact_url.startswith("/artifacts/"):
            raise ValueError(f"Only relative artifact URLs are allowed: {artifact_url}")

        filename = artifact_url.removeprefix("/artifacts/")
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError(f"Invalid artifact filename: {filename}")

        artifact_path = (CHARTS_DIR / filename).resolve()
        artifact_path.relative_to(CHARTS_DIR.resolve())

        mime_type = mimetypes.guess_type(artifact_path.name)[0] or "application/octet-stream"
        return artifact_path.read_bytes(), mime_type, artifact_path.name

    async def close(self) -> None:
        from src.meta_agent import meta_graph_manager

        await meta_graph_manager.aclose()


def is_local_meta_agent_enabled() -> bool:
    """Return whether Telegram should call the meta-agent in-process."""
    return os.getenv("TELEGRAM_LOCAL_META_AGENT") == "1"


def build_meta_agent_client(config: TelegramBotConfig) -> MetaAgentClient | LocalMetaAgentClient:
    """Create the meta-agent client, optionally bypassing HTTP for local artifact checks."""
    if is_local_meta_agent_enabled():
        logger.warning("TELEGRAM_LOCAL_META_AGENT=1: bypassing meta-agent HTTP API")
        return LocalMetaAgentClient()
    return MetaAgentClient(config.meta_agent_api_url, config.request_timeout)


async def discard_pending_updates(telegram: TelegramBotClient) -> int:
    """Skip pending Telegram updates and return the next offset to poll from."""
    updates = await telegram.get_updates(
        offset=-1,
        timeout=0,
        allowed_updates=ALLOWED_UPDATES,
    )
    if not updates:
        logger.info("No pending Telegram updates to discard")
        return 0

    next_offset = max(update["update_id"] for update in updates if "update_id" in update) + 1
    logger.warning("Discarded pending Telegram updates; starting from offset %d", next_offset)
    return next_offset


async def initial_update_offset(telegram: TelegramBotClient) -> int:
    """Return startup polling offset, discarding backlog in local meta-agent mode."""
    if is_local_meta_agent_enabled():
        return await discard_pending_updates(telegram)
    return 0


async def run_bot() -> None:
    """Run the Telegram bot long-polling loop."""
    config = TelegramBotConfig.from_env()
    logger.info("Starting Telegram bot with config: %s", config)

    telegram = TelegramBotClient(config.token, config.request_timeout)
    meta_agent = build_meta_agent_client(config)
    session_store = TelegramSessionStore(config.session_db_path)
    handler = MessageHandler(telegram, meta_agent, session_store)

    try:
        update_offset = await initial_update_offset(telegram)

        while True:
            try:
                updates = await telegram.get_updates(
                    offset=update_offset,
                    timeout=config.poll_timeout,
                    allowed_updates=ALLOWED_UPDATES,
                )

                for update in updates:
                    update_id = update.get("update_id")
                    if update_id:
                        update_offset = update_id + 1

                    msg = parse_update(update)
                    if msg:
                        logger.debug(
                            "Received message from chat %d (user %d): %s",
                            msg.chat_id,
                            msg.user_id,
                            msg.text[:50],
                        )
                        try:
                            await handler.handle_message(msg)
                        except Exception:
                            logger.exception(
                            "Error handling message from chat %d", msg.chat_id
                        )

                    callback = parse_callback_query(update)
                    if callback:
                        logger.debug(
                            "Received callback from chat %d (user %d): %s",
                            callback.chat_id,
                            callback.user_id,
                            callback.data[:50],
                        )
                        try:
                            await handler.handle_callback_query(
                                callback.callback_id,
                                callback.chat_id,
                                callback.user_id,
                                callback.data,
                            )
                        except Exception:
                            logger.exception(
                                "Error handling callback from chat %d", callback.chat_id
                            )

            except Exception as e:
                logger.error("Error in polling loop: %s. Retrying...", e)
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("Bot interrupted")
    finally:
        await telegram.close()
        await meta_agent.close()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(run_bot())
