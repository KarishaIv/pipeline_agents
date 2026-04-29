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

        # Create a session with placeholder thread_id
        session = store.create_session(123, 456, "-1", "default")
        assert session.name == "default"
        assert session.thread_id == "-1"
        assert session.is_active is True

        # Get active thread ID (placeholder)
        assert store.get_active_thread_id(123, 456) == "-1"

        # Replace placeholder with real thread_id
        store.replace_thread_id(123, 456, "-1", "thread-xyz")
        assert store.get_active_thread_id(123, 456) == "thread-xyz"
        
        # Verify session name was preserved
        active = store.get_active_session(123, 456)
        assert active.name == "default"
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
