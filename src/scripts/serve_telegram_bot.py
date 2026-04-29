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
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path so imports work when running as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta_agent.configs import TelegramBotConfig  # noqa: E402
from meta_agent.telegram.bot_client import TelegramBotClient  # noqa: E402
from meta_agent.telegram.meta_agent_client import MetaAgentClient  # noqa: E402
from meta_agent.telegram.session_store import TelegramSessionStore  # noqa: E402
from meta_agent.telegram.message_handler import MessageHandler  # noqa: E402
from meta_agent.telegram.update_parser import parse_update  # noqa: E402

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("telegram_bot")


async def run_bot() -> None:
    """Run the Telegram bot long-polling loop."""
    config = TelegramBotConfig.from_env()
    logger.info("Starting Telegram bot with config: %s", config)

    telegram = TelegramBotClient(config.token, config.request_timeout)
    meta_agent = MetaAgentClient(config.meta_agent_api_url, config.request_timeout)
    session_store = TelegramSessionStore(config.session_db_path)
    handler = MessageHandler(telegram, meta_agent, session_store)

    try:
        update_offset = 0
        while True:
            try:
                updates = await telegram.get_updates(
                    offset=update_offset,
                    timeout=config.poll_timeout,
                    allowed_updates=["message"],
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
