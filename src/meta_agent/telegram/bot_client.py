"""Telegram Bot API client for long-polling and message sending."""

from typing import Any, Optional
import httpx


class TelegramBotClient:
    """Async HTTP client for Telegram Bot API."""

    def __init__(self, token: str, request_timeout: float = 60.0):
        """Initialize Telegram client.

        Args:
            token: Telegram bot token.
            request_timeout: HTTP request timeout in seconds.
        """
        self.token = token
        self.request_timeout = request_timeout
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.client = httpx.AsyncClient(timeout=request_timeout)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_updates(
        self, offset: int = 0, timeout: int = 30, allowed_updates: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Get updates from Telegram (long polling).

        Args:
            offset: ID of the first update to be returned.
            timeout: Long polling timeout in seconds.
            allowed_updates: List of allowed update types.

        Returns:
            List of update objects.
        """
        params = {
            "offset": offset,
            "timeout": timeout,
        }
        if allowed_updates:
            params["allowed_updates"] = allowed_updates

        response = await self.client.post(
            f"{self.base_url}/getUpdates",
            json=params,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Send a text message.

        Args:
            chat_id: Telegram chat ID.
            text: Message text.
            parse_mode: Parse mode (HTML, Markdown, etc.).
            reply_to_message_id: Optional message ID to reply to.

        Returns:
            Telegram message object.
        """
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        response = await self.client.post(
            f"{self.base_url}/sendMessage",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})

    async def send_chat_action(self, chat_id: int | str, action: str) -> bool:
        """Send a chat action (typing, upload_photo, etc.).

        Args:
            chat_id: Telegram chat ID.
            action: Action type (typing, upload_document, etc.).

        Returns:
            True if action was sent successfully.
        """
        response = await self.client.post(
            f"{self.base_url}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
        )
        response.raise_for_status()
        return response.json().get("ok", False)

    async def send_document(
        self,
        chat_id: int | str,
        url: str,
        caption: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Send a document from URL.

        Args:
            chat_id: Telegram chat ID.
            url: URL of the document.
            caption: Optional caption.
            reply_to_message_id: Optional message ID to reply to.

        Returns:
            Telegram message object.
        """
        payload = {
            "chat_id": chat_id,
            "document": url,
        }
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        response = await self.client.post(
            f"{self.base_url}/sendDocument",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})
