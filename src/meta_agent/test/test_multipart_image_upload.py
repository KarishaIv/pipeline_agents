"""Tests for multipart image upload to Telegram (regression tests for breaking change)."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_meta_agent_client_fetch_artifact_bytes():
    """MetaAgentClient should be able to fetch artifact bytes from a relative URL."""
    from src.meta_agent.telegram.meta_agent_client import MetaAgentClient

    client = MetaAgentClient("http://localhost:8000")
    client.client = AsyncMock()

    # Mock response with binary content
    mock_response = MagicMock()
    mock_response.content = b'\x89PNG\r\n\x1a\n...'  # PNG header
    mock_response.headers = {"content-type": "image/png", "content-disposition": 'inline; filename="chart.png"'}
    client.client.get = AsyncMock(return_value=mock_response)

    # Should have a method to fetch binary artifacts
    if hasattr(client, 'fetch_artifact_bytes'):
        content, mime_type, filename = await client.fetch_artifact_bytes("/artifacts/chart.png")
        assert content == b'\x89PNG\r\n\x1a\n...'
        assert mime_type == "image/png"
        assert filename is not None


@pytest.mark.asyncio
async def test_telegram_bot_client_send_photo_multipart():
    """TelegramBotClient.send_photo should use multipart upload with binary content, not URL."""
    from src.meta_agent.telegram.bot_client import TelegramBotClient

    client = TelegramBotClient("test-token")
    client.client = AsyncMock()

    # Mock the response
    mock_response = MagicMock()
    mock_response.json = lambda: {"ok": True, "result": {"message_id": 123}}
    client.client.post = AsyncMock(return_value=mock_response)

    # Call send_photo with binary content (not URL)
    # Signature should be: send_photo(chat_id, content, filename, mime_type, caption, reply_to_message_id)
    photo_bytes = b'\x89PNG\r\n\x1a\n...'
    await client.send_photo(
        chat_id=123,
        content=photo_bytes,
        filename="chart.png",
        mime_type="image/png",
        caption="Test Chart"
    )

    # Verify the call used files parameter (multipart) not json parameter (URL)
    assert client.client.post.called
    call_kwargs = client.client.post.call_args.kwargs
    # Should have 'files' parameter with multipart data
    assert "files" in call_kwargs or "data" in call_kwargs


@pytest.mark.asyncio
async def test_message_handler_image_output_fetches_and_uploads():
    """MessageHandler should fetch ImageOutput bytes from API and upload via multipart."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.api_models import ImageOutput

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)

    # Mock the fetch_artifact_bytes to return binary content
    mock_meta_agent.fetch_artifact_bytes = AsyncMock(
        return_value=(b'\x89PNG\r\n\x1a\n...', "image/png", "chart.png")
    )

    outputs = [ImageOutput(url="/artifacts/chart.png", caption="Revenue Trend")]

    await handler.send_outputs(123, outputs)

    # Should have fetched the artifact bytes
    assert mock_meta_agent.fetch_artifact_bytes.called

    # Should have called send_photo with binary content
    assert mock_telegram.send_photo.called
    call_args = mock_telegram.send_photo.call_args
    # First arg should be chat_id, second should be content (bytes), not url
    if len(call_args.args) >= 2:
        assert call_args.args[0] == 123
        # Second arg should be bytes or keyword arg 'content' should be bytes
        if len(call_args.args) >= 2:
            assert isinstance(call_args.args[1], bytes) or isinstance(call_args.kwargs.get("content"), bytes)


@pytest.mark.asyncio
async def test_telegram_bot_client_send_photo_signature_not_url():
    """send_photo should accept content/filename/mime_type, not url."""
    from src.meta_agent.telegram.bot_client import TelegramBotClient
    import inspect

    client = TelegramBotClient("test-token")

    # Check method signature - should have content/filename/mime_type, not url
    if hasattr(client, "send_photo"):
        sig = inspect.signature(client.send_photo)
        params = list(sig.parameters.keys())
        # Should NOT have 'url' parameter
        assert "url" not in params, "send_photo should not have url parameter (use multipart content instead)"
        # Should have content, filename, or similar
        assert "content" in params or "file" in params or "data" in params


@pytest.mark.asyncio
async def test_telegram_bot_client_send_document_multipart():
    """TelegramBotClient.send_document should upload binary content, not pass a URL."""
    from src.meta_agent.telegram.bot_client import TelegramBotClient

    client = TelegramBotClient("test-token")
    client.client = AsyncMock()

    mock_response = MagicMock()
    mock_response.json = lambda: {"ok": True, "result": {"message_id": 124}}
    client.client.post = AsyncMock(return_value=mock_response)

    await client.send_document(
        chat_id=123,
        content=b"name,score\nAlice,10\n",
        filename="scores.csv",
        mime_type="text/csv",
        caption="Scores",
    )

    call_kwargs = client.client.post.call_args.kwargs
    assert "files" in call_kwargs
    assert "json" not in call_kwargs
    assert call_kwargs["files"]["document"] == (
        "scores.csv",
        b"name,score\nAlice,10\n",
        "text/csv",
    )


@pytest.mark.asyncio
async def test_message_handler_file_output_fetches_and_uploads_document():
    """MessageHandler should fetch FileOutput bytes from API and upload via multipart."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.api_models import FileOutput

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)
    mock_meta_agent.fetch_artifact_bytes = AsyncMock(
        return_value=(b"name,score\nAlice,10\n", "text/csv", "scores.csv")
    )

    outputs = [
        FileOutput(
            filename="scores.csv",
            mime_type="text/csv",
            download_url="/artifacts/scores.csv",
            caption="Scores",
        )
    ]

    await handler.send_outputs(123, outputs)

    mock_meta_agent.fetch_artifact_bytes.assert_awaited_once_with("/artifacts/scores.csv")
    mock_telegram.send_document.assert_awaited_once_with(
        123,
        content=b"name,score\nAlice,10\n",
        filename="scores.csv",
        mime_type="text/csv",
        caption="Scores",
    )
