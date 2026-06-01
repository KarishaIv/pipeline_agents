"""Telegram update parsing and command/message handling."""

from typing import Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TelegramMessage:
    """Parsed Telegram message."""

    chat_id: int
    user_id: int
    message_id: int
    text: str
    is_command: bool = False
    command: Optional[str] = None


@dataclass
class TelegramCallback:
    """Parsed Telegram callback query."""

    callback_id: str
    chat_id: int
    user_id: int
    data: str


def parse_update(update: dict[str, Any]) -> Optional[TelegramMessage]:
    """Parse a Telegram update into a message.

    Args:
        update: Raw Telegram update object.

    Returns:
        TelegramMessage or None if update is not a message type.
    """
    if "message" not in update:
        return None

    message_data = update["message"]
    text = message_data.get("text", "").strip()

    if not text:
        return None

    chat_id = message_data.get("chat", {}).get("id")
    user_id = message_data.get("from", {}).get("id")
    message_id = message_data.get("message_id")

    if not chat_id or not user_id:
        logger.warning("Missing chat_id or user_id in update")
        return None

    is_command = text.startswith("/")
    command = None
    if is_command:
        parts = text.split()
        command = parts[0][1:]  # Remove leading /
        if command:
            # If text is just the command, keep it as-is
            # Otherwise, use the full text for the question
            if len(parts) > 1:
                text = " ".join(parts[1:])
            else:
                text = ""

    return TelegramMessage(
        chat_id=chat_id,
        user_id=user_id,
        message_id=message_id,
        text=text,
        is_command=is_command,
        command=command,
    )


def parse_callback_query(update: dict[str, Any]) -> Optional[TelegramCallback]:
    """Parse a Telegram callback_query update.

    Args:
        update: Raw Telegram update object.

    Returns:
        TelegramCallback or None if update is not a callback_query type.
    """
    if "callback_query" not in update:
        return None

    callback_data = update["callback_query"]
    callback_id = callback_data.get("id")
    data = callback_data.get("data", "").strip()

    if not callback_id or not data:
        logger.warning("Missing callback_id or data in callback_query")
        return None

    from_data = callback_data.get("from", {})
    chat_id = callback_data.get("message", {}).get("chat", {}).get("id")
    user_id = from_data.get("id")

    if not chat_id or not user_id:
        logger.warning("Missing chat_id or user_id in callback_query")
        return None

    return TelegramCallback(
        callback_id=callback_id,
        chat_id=chat_id,
        user_id=user_id,
        data=data,
    )


def parse_message_text(text: str, max_length: int = 4096) -> list[str]:
    """Split long message text into Telegram-compatible chunks.

    Telegram has a 4096 character limit per message.

    Args:
        text: Message text.
        max_length: Maximum characters per chunk.

    Returns:
        List of message chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    lines = text.split("\n")
    for line in lines:
        # If adding this line would exceed max length, save current chunk and start new
        if current_chunk and len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

        # If a single line is longer than max_length, split it directly
        if len(current_chunk) > max_length:
            # Split the long line by max_length
            while len(current_chunk) > max_length:
                chunks.append(current_chunk[:max_length])
                current_chunk = current_chunk[max_length:]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
