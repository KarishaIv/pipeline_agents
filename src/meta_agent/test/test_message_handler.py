"""Tests for Telegram message handler."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.meta_agent import MetaAgentApiResponse, TextOutput
from src.meta_agent.telegram.message_handler import MessageHandler
from src.meta_agent.telegram.update_parser import TelegramMessage


@pytest.fixture
def message_handler(mocker):
    """Create a message handler with mocked clients."""
    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    mock_session = MagicMock()

    return MessageHandler(mock_telegram, mock_meta_agent, mock_session)


@pytest.mark.asyncio
async def test_handle_start_command(message_handler):
    """Test /start command handler."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="start"
    )

    await message_handler.handle_command(msg, "start", "")

    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert call_args[0][0] == 123
    assert "Welcome" in call_args[0][1]


@pytest.mark.asyncio
async def test_handle_help_command(message_handler):
    """Test /help command handler."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="help"
    )

    await message_handler.handle_command(msg, "help", "")

    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert "Help" in call_args[0][1]


@pytest.mark.asyncio
async def test_handle_new_command(message_handler):
    """Test /new command handler with session creation."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="my_work_session", is_command=True, command="new"
    )

    from src.meta_agent.telegram.session_store import Session
    from datetime import datetime

    mock_session = Session(
        thread_id="-1",
        user_key="chat_123_user_456",
        name="my_work_session",
        is_active=True,
        has_messages=False,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    message_handler.session_store.create_session = MagicMock(return_value=mock_session)

    await message_handler.handle_command(msg, "new", "my_work_session")

    message_handler.session_store.create_session.assert_called_once()
    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert "my_work_session" in call_args[0][1]


@pytest.mark.asyncio
async def test_handle_unknown_command(message_handler):
    """Test unknown command handler."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="unknown"
    )

    await message_handler.handle_command(msg, "unknown", "")

    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert "Unknown command" in call_args[0][1]


@pytest.mark.asyncio
async def test_handle_question_new_session(message_handler):
    """Test handling a question with new session creation."""
    from src.meta_agent.telegram.session_store import Session
    from datetime import datetime

    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="What is the status?"
    )

    # No active session initially
    message_handler.session_store.get_active_session = MagicMock(return_value=None)

    # Mock session creation with auto-generated thread_id
    generated_thread_id = "test-thread-uuid7"
    mock_session = Session(
        thread_id=generated_thread_id,
        user_key="chat_123_user_456",
        name="default",
        is_active=True,
        has_messages=False,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    message_handler.session_store.create_session = MagicMock(return_value=mock_session)

    mock_response = MetaAgentApiResponse(
        thread_id="new-thread-1", outputs=[TextOutput(text="Status is good")]
    )
    message_handler.meta_agent.ask = AsyncMock(return_value=mock_response)

    await message_handler.handle_question(msg)

    message_handler.telegram.send_chat_action.assert_called_once_with(123, "typing")
    message_handler.session_store.create_session.assert_called_once()
    message_handler.meta_agent.ask.assert_called_once_with("What is the status?", generated_thread_id)
    # No replace_thread_id should be called since thread_id is already generated
    message_handler.session_store.replace_thread_id.assert_not_called()


@pytest.mark.asyncio
async def test_handle_question_existing_session(message_handler):
    """Test handling a question in an existing session."""
    from src.meta_agent.telegram.session_store import Session
    from datetime import datetime

    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=2, text="Tell me more"
    )

    # Mock existing active session
    mock_session = Session(
        thread_id="existing-thread",
        user_key="chat_123_user_456",
        name="default",
        is_active=True,
        has_messages=False,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    message_handler.session_store.get_active_session = MagicMock(return_value=mock_session)

    mock_response = MetaAgentApiResponse(
        thread_id="existing-thread", outputs=[TextOutput(text="More details")]
    )
    message_handler.meta_agent.ask = AsyncMock(return_value=mock_response)

    await message_handler.handle_question(msg)

    message_handler.meta_agent.ask.assert_called_once_with("Tell me more", "existing-thread")
    # Should not call replace_thread_id since session already has a real thread_id
    message_handler.session_store.replace_thread_id.assert_not_called()


@pytest.mark.asyncio
async def test_handle_question_empty_text(message_handler):
    """Test handling a question with empty text."""
    msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="")

    await message_handler.handle_question(msg)

    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert "Please provide a question" in call_args[0][1]


@pytest.mark.asyncio
async def test_handle_question_error(message_handler):
    """Test error handling in question."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="What is wrong?"
    )

    message_handler.session_store.get_thread_id.return_value = None
    message_handler.meta_agent.ask = AsyncMock(side_effect=Exception("API Error"))

    await message_handler.handle_question(msg)

    message_handler.telegram.send_message.assert_called()
    call_args = message_handler.telegram.send_message.call_args
    assert "Error" in call_args[0][1] or "error" in call_args[0][1].lower()


@pytest.mark.asyncio
async def test_send_outputs_text(message_handler):
    """Test sending text outputs."""
    outputs = [TextOutput(text="Response text")]

    await message_handler.send_outputs(123, outputs)

    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert call_args[0][0] == 123
    assert call_args[0][1] == "Response text"


@pytest.mark.asyncio
async def test_send_outputs_long_text(message_handler):
    """Test sending long text outputs (should be split)."""
    long_text = "x" * 5000
    outputs = [TextOutput(text=long_text)]

    await message_handler.send_outputs(123, outputs)

    # Long text should result in at least one send_message call
    assert message_handler.telegram.send_message.call_count >= 1


@pytest.mark.asyncio
async def test_handle_message_as_question(message_handler):
    """Test routing a message as a question."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="Question", is_command=False
    )

    message_handler.session_store.get_thread_id.return_value = None
    mock_response = MetaAgentApiResponse(
        thread_id="t-1", outputs=[TextOutput(text="Answer")]
    )
    message_handler.meta_agent.ask = AsyncMock(return_value=mock_response)

    await message_handler.handle_message(msg)

    message_handler.meta_agent.ask.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_as_command(message_handler):
    """Test routing a message as a command."""
    msg = TelegramMessage(
        chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="help"
    )

    await message_handler.handle_message(msg)

    message_handler.telegram.send_message.assert_called_once()
    call_args = message_handler.telegram.send_message.call_args
    assert "Help" in call_args[0][1]
