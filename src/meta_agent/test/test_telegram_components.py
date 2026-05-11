"""Tests for Telegram bot components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.meta_agent.configs import TelegramBotConfig
from src.meta_agent.telegram.session_store import TelegramSessionStore
from src.meta_agent.telegram.update_parser import parse_update, parse_message_text
from src.meta_agent.telegram.bot_client import TelegramBotClient
from src.meta_agent.telegram.meta_agent_client import MetaAgentClient


def test_telegram_config_from_env(monkeypatch):
    """Test loading TelegramBotConfig from environment."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setenv("META_AGENT_API_URL", "http://example.com:8000")
    monkeypatch.setenv("TELEGRAM_POLL_TIMEOUT", "60")

    config = TelegramBotConfig.from_env()

    assert config.token == "test-token-123"
    assert config.meta_agent_api_url == "http://example.com:8000"
    assert config.poll_timeout == 60


def test_telegram_config_missing_token(monkeypatch):
    """Test that TelegramBotConfig raises on missing token."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        TelegramBotConfig.from_env()


def test_session_store_get_set(tmp_path):
    """Test TelegramSessionStore get/set operations."""
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        # No active session initially
        assert store.get_active_thread_id(123, 456) is None

        # Create a session with auto-generated thread_id
        session = store.create_session(123, 456, "", "default")
        assert session.name == "default"
        assert session.thread_id != ""
        assert session.is_active is True

        # Get active thread ID (should be the generated one)
        active_thread = store.get_active_thread_id(123, 456)
        assert active_thread == session.thread_id

        # Verify session name was preserved
        active = store.get_active_session(123, 456)
        assert active.name == "default"
        assert active.thread_id == session.thread_id
    finally:
        store.close()


def test_session_store_multiple_sessions(tmp_path):
    """Test creating and managing multiple sessions."""
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        # Create first session
        session1 = store.create_session(123, 456, "thread-1", "work", activate=True)
        assert session1.is_active is True

        # Create second session (should deactivate first)
        session2 = store.create_session(123, 456, "thread-2", "personal", activate=True)
        assert session2.is_active is True

        # List sessions
        sessions = store.list_sessions(123, 456)
        assert len(sessions) == 2

        # Only second session should be active
        active = store.get_active_session(123, 456)
        assert active.name == "personal"
        assert active.thread_id == "thread-2"
    finally:
        store.close()


def test_session_store_create_existing_session_reactivates_without_duplicate(tmp_path):
    """Test creating an existing session name returns and activates it."""
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        existing = store.create_session(123, 456, "thread-1", "default", activate=False)
        other = store.create_session(123, 456, "thread-2", "work", activate=True)

        session = store.create_session(123, 456, "", "default", activate=True)

        assert session.thread_id == existing.thread_id
        assert session.name == "default"
        assert session.is_active is True

        sessions = store.list_sessions(123, 456)
        assert len(sessions) == 2
        assert [s for s in sessions if s.name == "default"][0].is_active is True
        assert [s for s in sessions if s.name == "work"][0].is_active is False
        assert store.get_active_thread_id(123, 456) == existing.thread_id
        assert store.get_active_thread_id(123, 456) != other.thread_id
    finally:
        store.close()


def test_session_store_switch(tmp_path):
    """Test switching between sessions."""
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        # Create two sessions
        store.create_session(123, 456, "thread-1", "session1", activate=True)
        store.create_session(123, 456, "thread-2", "session2", activate=False)

        # Switch to session2
        switched = store.switch_session(123, 456, "session2")
        assert switched is not None
        assert switched.name == "session2"
        # BUG FIX: returned Session should have is_active=True after successful switch
        assert switched.is_active is True, "Switched session should have is_active=True"

        # Verify active session changed
        active = store.get_active_session(123, 456)
        assert active.name == "session2"
        assert active.thread_id == "thread-2"

        # Try switching to non-existent session
        result = store.switch_session(123, 456, "nonexistent")
        assert result is None
    finally:
        store.close()


def test_session_store_delete(tmp_path):
    """Test deleting sessions."""
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        store.create_session(123, 456, "thread-1", "session1", activate=True)
        store.create_session(123, 456, "thread-2", "session2", activate=False)

        # Delete inactive session
        deleted = store.delete_session(123, 456, "session2")
        assert deleted is True

        sessions = store.list_sessions(123, 456)
        assert len(sessions) == 1
        assert sessions[0].name == "session1"

        # Try deleting non-existent session
        deleted = store.delete_session(123, 456, "nonexistent")
        assert deleted is False
    finally:
        store.close()


def test_session_store_user_scoped(tmp_path):
    """Test user-scoped session storage."""
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        store.create_session(100, 1, "thread-a", "default", activate=True)
        store.create_session(100, 2, "thread-b", "default", activate=True)

        assert store.get_active_thread_id(100, 1) == "thread-a"
        assert store.get_active_thread_id(100, 2) == "thread-b"
    finally:
        store.close()


def test_parse_update_simple_message():
    """Test parsing a simple text message update."""
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "date": 1234567890,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 456, "first_name": "Test"},
            "text": "Hello bot",
        },
    }

    msg = parse_update(update)

    assert msg is not None
    assert msg.chat_id == 123
    assert msg.user_id == 456
    assert msg.text == "Hello bot"
    assert msg.is_command is False


def test_parse_update_command():
    """Test parsing a command message."""
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "chat": {"id": 789},
            "from": {"id": 111},
            "text": "/start",
        },
    }

    msg = parse_update(update)

    assert msg is not None
    assert msg.is_command is True
    assert msg.command == "start"
    assert msg.text == ""


def test_parse_update_command_with_args():
    """Test parsing a command with arguments."""
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "chat": {"id": 789},
            "from": {"id": 111},
            "text": "/ask how are you",
        },
    }

    msg = parse_update(update)

    assert msg is not None
    assert msg.is_command is True
    assert msg.command == "ask"
    assert msg.text == "how are you"


def test_parse_update_non_message():
    """Test parsing a non-message update."""
    update = {"update_id": 1, "callback_query": {"data": "test"}}

    msg = parse_update(update)
    assert msg is None


def test_parse_update_missing_ids():
    """Test parsing a message with missing IDs."""
    update = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "text": "No chat/user IDs",
        },
    }

    msg = parse_update(update)
    assert msg is None


def test_parse_message_text_short():
    """Test splitting short message text."""
    text = "Short message"
    chunks = parse_message_text(text)

    assert len(chunks) == 1
    assert chunks[0] == "Short message"


def test_parse_message_text_long():
    """Test splitting long message text."""
    text = "x" * 5000
    chunks = parse_message_text(text, max_length=1000)

    assert len(chunks) >= 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    # Verify that all chunks joined equal original (or close to it)
    joined = "".join(chunks)
    assert len(joined) >= 4996  # Allow for edge case truncation


def test_parse_message_text_multiline():
    """Test splitting multiline message."""
    text = "Line 1\nLine 2\nLine 3"
    chunks = parse_message_text(text, max_length=10)

    assert len(chunks) > 0
    assert "\n".join(chunks).startswith("Line 1")


@pytest.mark.asyncio
async def test_telegram_bot_client_get_updates(mocker):
    """Test TelegramBotClient.get_updates."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": [{"update_id": 1}]}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        bot = TelegramBotClient("test-token")
        updates = await bot.get_updates(offset=0, timeout=30)

        assert len(updates) == 1
        assert updates[0]["update_id"] == 1
        await bot.close()


@pytest.mark.asyncio
async def test_telegram_bot_client_send_message(mocker):
    """Test TelegramBotClient.send_message."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        bot = TelegramBotClient("test-token")
        msg = await bot.send_message(123, "Test message")

        assert msg["message_id"] == 42
        await bot.close()


@pytest.mark.asyncio
async def test_meta_agent_client_ask(mocker):
    """Test MetaAgentClient.ask."""
    from meta_agent import MetaAgentApiResponse

    response_data = {
        "thread_id": "t-1",
        "outputs": [{"type": "text", "text": "Answer"}],
    }

    mock_response = MagicMock()
    mock_response.json.return_value = response_data
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        client = MetaAgentClient("http://localhost:8000")
        response = await client.ask("Test question", thread_id=None)

        assert isinstance(response, MetaAgentApiResponse)
        assert response.thread_id == "t-1"
        assert len(response.outputs) == 1
        await client.close()


@pytest.mark.asyncio
async def test_telegram_bot_client_send_message_with_reply_markup(mocker):
    """Test TelegramBotClient.send_message with reply_markup."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {"message_id": 42}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        bot = TelegramBotClient("test-token")
        reply_markup = {
            "keyboard": [[{"text": "/help"}]],
            "resize_keyboard": True,
        }
        msg = await bot.send_message(123, "Test message", reply_markup=reply_markup)

        assert msg["message_id"] == 42

        # Verify the reply_markup was included in the payload
        call_args = mock_client.post.call_args
        posted_json = call_args.kwargs["json"]
        assert "reply_markup" in posted_json
        assert posted_json["reply_markup"] == reply_markup

        await bot.close()


def test_get_commands_keyboard():
    """Test that get_commands_keyboard returns proper ReplyKeyboardMarkup."""
    from src.meta_agent.telegram.message_handler import get_commands_keyboard

    keyboard = get_commands_keyboard()

    assert "keyboard" in keyboard
    assert keyboard["resize_keyboard"] is True
    assert keyboard["is_persistent"] is True
    assert keyboard["one_time_keyboard"] is False

    # Check that all command buttons are present
    buttons_text = []
    for row in keyboard["keyboard"]:
        for button in row:
            buttons_text.append(button["text"])

    assert "/help" in buttons_text
    assert "/new" in buttons_text
    assert "/sessions" in buttons_text
    assert "/switch" in buttons_text
    assert "/delete" in buttons_text


@pytest.mark.asyncio
async def test_message_handler_start_command_sends_keyboard(tmp_path, mocker):
    """Test that /start command sends reply keyboard."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="start")

        await handler.handle_command(msg, "start", "")

        # Verify send_message was called with reply_markup
        mock_telegram.send_message.assert_called_once()
        call_args = mock_telegram.send_message.call_args

        assert call_args.kwargs.get("reply_markup") is not None
        assert "keyboard" in call_args.kwargs["reply_markup"]
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_help_command_sends_keyboard(tmp_path, mocker):
    """Test that /help command sends reply keyboard."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="help")

        await handler.handle_command(msg, "help", "")

        # Verify send_message was called with reply_markup
        mock_telegram.send_message.assert_called_once()
        call_args = mock_telegram.send_message.call_args

        assert call_args.kwargs.get("reply_markup") is not None
        assert "keyboard" in call_args.kwargs["reply_markup"]
    finally:
        session_store.close()


def test_get_session_switch_keyboard():
    """Test that get_session_switch_keyboard creates inline buttons with all sessions."""
    from src.meta_agent.telegram.message_handler import get_session_switch_keyboard
    from unittest.mock import MagicMock

    # Mock sessions
    session1 = MagicMock()
    session1.name = "session1"
    session1.is_active = True

    session2 = MagicMock()
    session2.name = "session2"
    session2.is_active = False

    sessions = [session1, session2]
    keyboard = get_session_switch_keyboard(sessions)

    assert "inline_keyboard" in keyboard
    assert len(keyboard["inline_keyboard"]) == 2

    # Check active session button
    assert "🟢" in keyboard["inline_keyboard"][0][0]["text"]
    assert "(active)" in keyboard["inline_keyboard"][0][0]["text"]
    assert keyboard["inline_keyboard"][0][0]["callback_data"] == "switch_active"

    # Check inactive session button
    assert "⚪️" in keyboard["inline_keyboard"][1][0]["text"]
    assert keyboard["inline_keyboard"][1][0]["callback_data"] == "switch:session2"


def test_get_session_delete_keyboard():
    """Test that get_session_delete_keyboard creates inline buttons for deletion."""
    from src.meta_agent.telegram.message_handler import get_session_delete_keyboard
    from unittest.mock import MagicMock

    # Mock sessions
    session1 = MagicMock()
    session1.name = "session1"
    session1.is_active = True

    session2 = MagicMock()
    session2.name = "session2"
    session2.is_active = False

    session3 = MagicMock()
    session3.name = "session3"
    session3.is_active = False

    sessions = [session1, session2, session3]
    keyboard = get_session_delete_keyboard(sessions)

    assert "inline_keyboard" in keyboard
    assert len(keyboard["inline_keyboard"]) == 3

    # Check active session button
    assert "🟢" in keyboard["inline_keyboard"][0][0]["text"]
    assert "(active)" in keyboard["inline_keyboard"][0][0]["text"]
    assert keyboard["inline_keyboard"][0][0]["callback_data"] == "delete_active"

    # Check inactive session buttons
    assert "⚪️" in keyboard["inline_keyboard"][1][0]["text"]
    assert keyboard["inline_keyboard"][1][0]["callback_data"] == "delete:session2"
    assert "⚪️" in keyboard["inline_keyboard"][2][0]["text"]
    assert keyboard["inline_keyboard"][2][0]["callback_data"] == "delete:session3"


@pytest.mark.asyncio
async def test_parse_callback_query():
    """Test parsing a callback query update."""
    from src.meta_agent.telegram.update_parser import parse_callback_query

    update = {
        "update_id": 1,
        "callback_query": {
            "id": "callback123",
            "from": {"id": 456},
            "message": {
                "message_id": 10,
                "chat": {"id": 123},
            },
            "data": "switch:session2",
        },
    }

    callback = parse_callback_query(update)

    assert callback is not None
    assert callback.callback_id == "callback123"
    assert callback.chat_id == 123
    assert callback.user_id == 456
    assert callback.data == "switch:session2"


@pytest.mark.asyncio
async def test_telegram_bot_client_answer_callback_query(mocker):
    """Test TelegramBotClient.answer_callback_query."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_client):
        bot = TelegramBotClient("test-token")
        result = await bot.answer_callback_query("callback123", text="Done")

        assert result is True

        # Verify the request was made correctly
        call_args = mock_client.post.call_args
        posted_json = call_args.kwargs["json"]
        assert posted_json["callback_query_id"] == "callback123"
        assert posted_json["text"] == "Done"

        await bot.close()


@pytest.mark.asyncio
async def test_message_handler_callback_switch(tmp_path):
    """Test callback query handler for switch command."""
    from src.meta_agent.telegram.message_handler import MessageHandler

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create two sessions
        session_store.create_session(123, 456, "thread-1", "session1", activate=True)
        session_store.create_session(123, 456, "thread-2", "session2", activate=False)

        await handler.handle_callback_query("cb123", 123, 456, "switch:session2")

        # Verify answer_callback_query and send_message were called
        mock_telegram.answer_callback_query.assert_called_once()
        mock_telegram.send_message.assert_called_once()

        # Verify the message indicates success
        send_msg_call = mock_telegram.send_message.call_args
        assert "Switched to session" in send_msg_call[0][1] or "Switched to session" in str(send_msg_call)
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_callback_switch_active(tmp_path):
    """Test callback query handler when trying to switch to already active session."""
    from src.meta_agent.telegram.message_handler import MessageHandler

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        session_store.create_session(123, 456, "thread-1", "session1", activate=True)

        await handler.handle_callback_query("cb123", 123, 456, "switch_active")

        # Verify answer_callback_query was called with alert message
        mock_telegram.answer_callback_query.assert_called_once()
        call_args = mock_telegram.answer_callback_query.call_args
        assert "already active" in call_args[1]["text"].lower()
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_callback_delete(tmp_path):
    """Test callback query handler for delete command."""
    from src.meta_agent.telegram.message_handler import MessageHandler

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create two sessions
        session_store.create_session(123, 456, "thread-1", "session1", activate=True)
        session_store.create_session(123, 456, "thread-2", "session2", activate=False)

        await handler.handle_callback_query("cb123", 123, 456, "delete:session2")

        # Verify answer_callback_query and send_message were called
        mock_telegram.answer_callback_query.assert_called_once()
        mock_telegram.send_message.assert_called_once()

        # Verify the message indicates success
        send_msg_call = mock_telegram.send_message.call_args
        assert "deleted" in send_msg_call[0][1].lower() or "deleted" in str(send_msg_call).lower()
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_callback_delete_active(tmp_path):
    """Test callback query handler when trying to delete active session."""
    from src.meta_agent.telegram.message_handler import MessageHandler

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        session_store.create_session(123, 456, "thread-1", "session1", activate=True)

        await handler.handle_callback_query("cb123", 123, 456, "delete_active")

        # Verify answer_callback_query was called with alert message
        mock_telegram.answer_callback_query.assert_called_once()
        call_args = mock_telegram.answer_callback_query.call_args
        assert "cannot delete" in call_args[1]["text"].lower()
    finally:
        session_store.close()


def test_session_store_new_session_replaces_empty_placeholder(tmp_path):
    """Test that creating new sessions generates unique thread IDs.

    Each new session should get its own unique thread ID, so multiple
    sessions can coexist with proper active status management.
    """
    db_path = tmp_path / "test_sessions.sqlite3"
    store = TelegramSessionStore(db_path)
    try:
        # First /new command creates session with auto-generated thread_id
        session1 = store.create_session(123, 456, "", "session_1", activate=True)
        assert session1.name == "session_1"
        assert session1.thread_id != ""
        assert session1.is_active is True
        thread_id_1 = session1.thread_id

        # List sessions - should have 1
        sessions = store.list_sessions(123, 456)
        assert len(sessions) == 1

        # Second /new command creates another session with different thread_id
        session2 = store.create_session(123, 456, "", "session_2", activate=True)
        assert session2.name == "session_2"
        assert session2.thread_id != ""
        assert session2.is_active is True
        assert session2.thread_id != thread_id_1

        # After second /new, we should have BOTH sessions
        # The active marker should be on session_2, not session_1
        sessions = store.list_sessions(123, 456)
        assert len(sessions) == 2

        # Only session_2 should be active
        active_sessions = [s for s in sessions if s.is_active]
        assert len(active_sessions) == 1
        assert active_sessions[0].name == "session_2"

        # session_1 should NOT be active
        inactive_sessions = [s for s in sessions if not s.is_active]
        assert len(inactive_sessions) == 1
        assert inactive_sessions[0].name == "session_1"
    finally:
        store.close()


def test_is_default_session_name():
    """Test checking if session name is default."""
    from src.meta_agent.telegram.message_handler import is_default_session_name

    # Default session names
    assert is_default_session_name("session_1") is True
    assert is_default_session_name("session_2") is True
    assert is_default_session_name("session_100") is True

    # Non-default session names
    assert is_default_session_name("work") is False
    assert is_default_session_name("personal") is False
    assert is_default_session_name("session_") is False  # No number
    assert is_default_session_name("Session_1") is False  # Capital S
    assert is_default_session_name("my_session_1") is False  # Different prefix


def test_should_delete_session():
    """Test checking if a session should be deleted."""
    from src.meta_agent.telegram.message_handler import should_delete_session
    from src.meta_agent.telegram.session_store import Session
    from datetime import datetime

    now = datetime.now()

    # Default session without messages - should be deleted
    session = Session(
        thread_id="thread1",
        user_key="user1",
        name="session_1",
        is_active=True,
        has_messages=False,
        created_at=now,
        updated_at=now
    )
    assert should_delete_session(session) is True

    # Default session with messages - should NOT be deleted
    session_with_messages = Session(
        thread_id="thread2",
        user_key="user1",
        name="session_2",
        is_active=False,
        has_messages=True,
        created_at=now,
        updated_at=now
    )
    assert should_delete_session(session_with_messages) is False

    # User-named session without messages - should NOT be deleted
    user_session = Session(
        thread_id="thread3",
        user_key="user1",
        name="work",
        is_active=True,
        has_messages=False,
        created_at=now,
        updated_at=now
    )
    assert should_delete_session(user_session) is False


@pytest.mark.asyncio
async def test_message_handler_marks_session_has_messages(tmp_path):
    """Test that sessions are marked as having messages after meta-agent response."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage
    from meta_agent import TextOutput, MetaAgentApiResponse

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create a session
        session = session_store.create_session(123, 456, "test-thread", "default", activate=True)
        assert session.has_messages is False

        # Mock meta-agent response
        mock_response = MetaAgentApiResponse(
            thread_id="test-thread",
            outputs=[TextOutput(text="Response")]
        )
        mock_meta_agent.ask = AsyncMock(return_value=mock_response)

        # Handle a question
        msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="What is this?")

        await handler.handle_question(msg)

        # Verify session is now marked as having messages
        updated_session = session_store.get_active_session(123, 456)
        assert updated_session.has_messages is True
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_question_reactivates_existing_default_session(tmp_path):
    """Test questions reuse an inactive default session instead of duplicating it."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage
    from meta_agent import TextOutput, MetaAgentApiResponse

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        session = session_store.create_session(123, 456, "default-thread", "default", activate=False)
        assert session_store.get_active_session(123, 456) is None

        mock_response = MetaAgentApiResponse(
            thread_id="default-thread",
            outputs=[TextOutput(text="Response")],
        )
        mock_meta_agent.ask = AsyncMock(return_value=mock_response)

        msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="What is this?")

        await handler.handle_question(msg)

        mock_meta_agent.ask.assert_called_once_with("What is this?", session.thread_id)
        active = session_store.get_active_session(123, 456)
        assert active.thread_id == session.thread_id
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_new_deletes_empty_sessions(tmp_path):
    """Test that /new command deletes the old session if it's default and empty."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create an existing default empty session
        session_store.create_session(123, 456, "", "session_1", activate=True)

        # Call /new via message handler
        msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="new")

        await handler.handle_command(msg, "new", "")

        # session_1 should be deleted (was active, is default, no messages)
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]

        assert "session_1" not in session_names  # Deleted
        assert len(sessions) == 1  # Only new session
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_new_keeps_old_session_with_messages(tmp_path):
    """Test that /new command keeps the old session if it has messages."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create a session and mark it as having messages
        session1 = session_store.create_session(123, 456, "", "session_1", activate=True)
        session_store.mark_session_has_messages(123, 456, session1.thread_id)

        # Call /new via message handler
        msg = TelegramMessage(chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="new")

        await handler.handle_command(msg, "new", "")

        # session_1 should NOT be deleted (has messages)
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]

        assert "session_1" in session_names  # Kept
        assert len(sessions) == 2  # Old and new sessions
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_switch_deletes_empty_old_session(tmp_path):
    """Test that /switch command deletes the old session if it's default and empty."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create two default empty sessions
        session1 = session_store.create_session(123, 456, "", "session_1", activate=True)
        session_store.create_session(123, 456, "", "session_2", activate=False)

        sessions = session_store.list_sessions(123, 456)
        assert len(sessions) == 2
        assert session1.is_active is True

        # Switch to session_2
        msg = TelegramMessage(
            chat_id=123, user_id=456, message_id=1, text="session_2",
            is_command=True, command="switch"
        )
        await handler.handle_command(msg, "switch", "session_2")

        # session_1 should be deleted (was active, is default, no messages)
        # session_2 should be active
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]

        assert "session_1" not in session_names  # Deleted
        assert "session_2" in session_names      # Still there, now active
        assert len(sessions) == 1
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_callback_switch_deletes_empty_old_session(tmp_path):
    """Test that /switch via callback button deletes the old session if it's default and empty."""
    from src.meta_agent.telegram.message_handler import MessageHandler

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create two default empty sessions
        session1 = session_store.create_session(123, 456, "", "session_1", activate=True)
        session_store.create_session(123, 456, "", "session_2", activate=False)

        sessions = session_store.list_sessions(123, 456)
        assert len(sessions) == 2
        assert session1.is_active is True

        # Switch via callback button (like inline keyboard button press)
        await handler.handle_callback_query("callback123", 123, 456, "switch:session_2")

        # session_1 should be deleted (was active, is default, no messages)
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]

        assert "session_1" not in session_names  # Deleted
        assert "session_2" in session_names      # Still there, now active
        assert len(sessions) == 1
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_switch_keeps_old_session_with_messages(tmp_path):
    """Test that /switch keeps the old session if it has messages."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create two sessions
        session1 = session_store.create_session(123, 456, "", "session_1", activate=True)
        session_store.create_session(123, 456, "", "session_2", activate=False)

        # Mark session1 as having messages
        session_store.mark_session_has_messages(123, 456, session1.thread_id)

        # Switch to session_2
        msg = TelegramMessage(
            chat_id=123, user_id=456, message_id=1, text="session_2",
            is_command=True, command="switch"
        )
        await handler.handle_command(msg, "switch", "session_2")

        # session_1 should NOT be deleted (has messages)
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]

        assert "session_1" in session_names  # Kept because has_messages
        assert "session_2" in session_names  # Active
        assert len(sessions) == 2
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_switch_keeps_user_named_sessions(tmp_path):
    """Test that /switch keeps user-named sessions even if empty."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create user-named and default session
        session_store.create_session(123, 456, "", "work", activate=True)
        session_store.create_session(123, 456, "", "session_2", activate=False)

        # Switch to session_2
        msg = TelegramMessage(
            chat_id=123, user_id=456, message_id=1, text="session_2",
            is_command=True, command="switch"
        )
        await handler.handle_command(msg, "switch", "session_2")

        # "work" should NOT be deleted (user-named)
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]

        assert "work" in session_names  # Kept because user-named
        assert "session_2" in session_names  # Active
        assert len(sessions) == 2
    finally:
        session_store.close()


@pytest.mark.asyncio
async def test_message_handler_new_with_session_gap_uses_next_number(tmp_path):
    """Test that /new generates the next unused default session number when gaps exist (regression for len-based naming)."""
    from src.meta_agent.telegram.message_handler import MessageHandler
    from src.meta_agent.telegram.update_parser import TelegramMessage

    db_path = tmp_path / "test_sessions.sqlite3"

    mock_telegram = AsyncMock()
    mock_meta_agent = AsyncMock()
    session_store = TelegramSessionStore(db_path)

    handler = MessageHandler(mock_telegram, mock_meta_agent, session_store)

    try:
        # Create session_1 and session_3 (gap at 2), both with messages so they are not auto-deleted on /new
        session1 = session_store.create_session(123, 456, "", "session_1", activate=True)
        session_store.mark_session_has_messages(123, 456, session1.thread_id)
        session3 = session_store.create_session(123, 456, "", "session_3", activate=True)
        session_store.mark_session_has_messages(123, 456, session3.thread_id)

        # session_3 should be the active one
        active_before = session_store.get_active_session(123, 456)
        assert active_before.name == "session_3"
        thread_id_3 = active_before.thread_id

        # Call /new with no name provided -> should create session_4, not reuse session_3
        msg = TelegramMessage(
            chat_id=123, user_id=456, message_id=1, text="", is_command=True, command="new"
        )
        await handler.handle_command(msg, "new", "")

        # Should now have 3 sessions: 1, 3, and new 4
        sessions = session_store.list_sessions(123, 456)
        session_names = [s.name for s in sessions]
        assert len(sessions) == 3, f"Expected 3 sessions, got {len(sessions)}. Names: {session_names}"
        assert "session_1" in session_names
        assert "session_3" in session_names
        assert "session_4" in session_names, f"session_4 missing, got: {session_names}"

        # session_4 should be active and a fresh session (new thread_id)
        active = session_store.get_active_session(123, 456)
        assert active.name == "session_4"
        assert active.thread_id != thread_id_3, "New session should have a different thread_id"
    finally:
        session_store.close()
