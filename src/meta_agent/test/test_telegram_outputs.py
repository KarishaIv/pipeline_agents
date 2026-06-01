from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_message_handler_dispatches_text_and_json_outputs():
    """TextOutput and JsonOutput should be sent as Telegram messages."""
    from src.meta_agent.api_models import JsonOutput, TextOutput
    from src.meta_agent.telegram.message_handler import MessageHandler

    mock_telegram = AsyncMock()
    mock_telegram.send_message = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)

    outputs = [
        TextOutput(text="Status"),
        JsonOutput(data={"key": "value"}, caption="Data"),
    ]

    await handler.send_outputs(123, outputs)

    assert mock_telegram.send_message.await_count == 2
    first_call, second_call = mock_telegram.send_message.await_args_list
    assert first_call.args == (123, "Status")
    assert second_call.args[0] == 123
    assert "<b>Data</b>" in second_call.args[1]
    assert '"key": "value"' in second_call.args[1]
    mock_telegram.send_photo.assert_not_called()
    mock_telegram.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_message_handler_does_not_raise_when_text_delivery_fails():
    """Telegram network errors while sending text should not mark meta-agent run as failed."""
    from src.meta_agent.api_models import TextOutput
    from src.meta_agent.telegram.message_handler import MessageHandler

    mock_telegram = AsyncMock()
    mock_telegram.send_message = AsyncMock(side_effect=ConnectionError("telegram down"))
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)

    await handler.send_outputs(123, [TextOutput(text="Done")])

    mock_telegram.send_message.assert_awaited_once_with(123, "Done")


@pytest.mark.asyncio
async def test_message_handler_dispatches_image_output_with_fetched_bytes():
    """ImageOutput should be fetched from the API and uploaded as multipart photo."""
    from src.meta_agent.api_models import ImageOutput
    from src.meta_agent.telegram.message_handler import MessageHandler

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_meta_agent.fetch_artifact_bytes = AsyncMock(
        return_value=(b"image-bytes", "image/png", "chart.png")
    )
    mock_session = MagicMock()

    handler = MessageHandler(mock_telegram, mock_meta_agent, mock_session)

    await handler.send_outputs(
        123,
        [ImageOutput(url="/artifacts/chart.png", caption="Chart")],
    )

    mock_meta_agent.fetch_artifact_bytes.assert_awaited_once_with("/artifacts/chart.png")
    mock_telegram.send_photo.assert_awaited_once_with(
        123,
        content=b"image-bytes",
        filename="chart.png",
        mime_type="image/png",
        caption="Chart",
    )
    mock_telegram.send_message.assert_not_called()
