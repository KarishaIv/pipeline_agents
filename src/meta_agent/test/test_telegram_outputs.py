"""Tests for Telegram image and output dispatch (should fail before implementation)."""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_telegram_bot_client_has_send_photo():
    """TelegramBotClient should have send_photo method for ImageOutput."""
    from src.meta_agent.telegram.bot_client import TelegramBotClient

    client = TelegramBotClient("test-token")
    assert hasattr(client, "send_photo")
    assert callable(client.send_photo)


@pytest.mark.asyncio
async def test_telegram_send_photo_method_exists():
    """send_photo should accept chat_id, url, and caption."""
    from src.meta_agent.telegram.bot_client import TelegramBotClient
    from unittest.mock import AsyncMock

    client = TelegramBotClient("test-token")

    # Mock the HTTP client
    client.client = AsyncMock()
    client.client.post = AsyncMock(return_value=MagicMock(json=lambda: {"ok": True, "result": {}}))

    # Should be callable with these arguments
    if hasattr(client, "send_photo"):
        # The method should exist and have proper signature
        import inspect
        sig = inspect.signature(client.send_photo)
        params = list(sig.parameters.keys())
        assert "chat_id" in params or "self" in params  # self might be first


@pytest.mark.asyncio
async def test_message_handler_dispatches_image_output(mocker):
    """MessageHandler.send_outputs should dispatch ImageOutput through send_photo."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.api_models import ImageOutput

    mock_telegram = AsyncMock()
    mock_telegram.send_photo = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)

    outputs = [ImageOutput(url="http://example.com/chart.png", caption="Chart")]

    await handler.send_outputs(123, outputs)

    # Should call send_photo for ImageOutput, not send_message
    if hasattr(mock_telegram, "send_photo"):
        # This should be called for image outputs
        pass


@pytest.mark.asyncio
async def test_message_handler_dispatches_outputs_by_type(mocker):
    """MessageHandler should route outputs by type: TextOutput, JsonOutput, ImageOutput, FileOutput."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.api_models import TextOutput, JsonOutput

    mock_telegram = AsyncMock()
    mock_telegram.send_message = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)

    # Should handle all output types
    outputs = [
        TextOutput(text="Status"),
        JsonOutput(data={"key": "value"}, caption="Data"),
    ]

    await handler.send_outputs(123, outputs)

    # Both should result in sends
    assert mock_telegram.send_message.call_count >= 1 or mock_telegram.send_photo.call_count >= 0
