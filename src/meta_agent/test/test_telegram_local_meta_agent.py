"""Tests for the local Telegram-to-meta-agent development bypass."""

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_local_meta_agent_client_ask_invokes_graph_session(mocker):
    from src.meta_agent import MetaAgentResult, TextOutput
    from src.scripts.serve_telegram_bot import LocalMetaAgentClient

    mock_invoke = mocker.patch(
        "src.meta_agent.meta_graph_manager.invoke_graph_session",
        new=AsyncMock(
            return_value=MetaAgentResult(
                thread_id="thread-1",
                outputs=[TextOutput(text="local answer")],
            )
        ),
    )

    response = await LocalMetaAgentClient().ask("question", "thread-0")

    assert response.thread_id == "thread-1"
    assert response.outputs == [TextOutput(text="local answer")]
    mock_invoke.assert_awaited_once_with("question", "thread-0")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content", "expected_mime_type"),
    [
        ("chart.png", b"png-bytes", "image/png"),
        ("data.csv", b"name,score\nAlice,10\n", "text/csv"),
        ("data.json", b'{"name": "Alice"}', "application/json"),
    ],
)
async def test_local_meta_agent_client_fetches_artifacts_from_charts_dir(
    tmp_path,
    mocker,
    filename,
    content,
    expected_mime_type,
):
    from src.scripts.serve_telegram_bot import LocalMetaAgentClient

    mocker.patch("src.scripts.serve_telegram_bot.CHARTS_DIR", tmp_path)
    (tmp_path / filename).write_bytes(content)

    actual_content, mime_type, actual_filename = await LocalMetaAgentClient().fetch_artifact_bytes(
        f"/artifacts/{filename}"
    )

    assert actual_content == content
    assert mime_type == expected_mime_type
    assert actual_filename == filename


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_url", ["https://example.com/chart.png", "/files/chart.png", "/artifacts/../secret.txt"])
async def test_local_meta_agent_client_rejects_non_artifact_paths(artifact_url):
    from src.scripts.serve_telegram_bot import LocalMetaAgentClient

    with pytest.raises(ValueError):
        await LocalMetaAgentClient().fetch_artifact_bytes(artifact_url)


def test_build_meta_agent_client_uses_http_by_default(monkeypatch, mocker):
    from src.meta_agent.configs import TelegramBotConfig
    from src.scripts import serve_telegram_bot

    monkeypatch.delenv("TELEGRAM_LOCAL_META_AGENT", raising=False)
    mock_http_client = mocker.patch("src.scripts.serve_telegram_bot.MetaAgentClient")
    config = TelegramBotConfig(
        token="token",
        meta_agent_api_url="http://meta-agent.test",
        request_timeout=12.0,
    )

    client = serve_telegram_bot.build_meta_agent_client(config)

    assert client == mock_http_client.return_value
    mock_http_client.assert_called_once_with("http://meta-agent.test", 12.0)


def test_build_meta_agent_client_uses_local_client_when_enabled(monkeypatch):
    from src.meta_agent.configs import TelegramBotConfig
    from src.scripts.serve_telegram_bot import LocalMetaAgentClient, build_meta_agent_client

    monkeypatch.setenv("TELEGRAM_LOCAL_META_AGENT", "1")
    config = TelegramBotConfig(token="token", meta_agent_api_url="http://unused.test")

    assert isinstance(build_meta_agent_client(config), LocalMetaAgentClient)


@pytest.mark.asyncio
async def test_discard_pending_updates_skips_existing_telegram_backlog():
    from src.scripts.serve_telegram_bot import ALLOWED_UPDATES, discard_pending_updates

    telegram = AsyncMock()
    telegram.get_updates = AsyncMock(
        return_value=[
            {"update_id": 41, "message": {"text": "old"}},
            {"update_id": 42, "message": {"text": "also old"}},
        ]
    )

    next_offset = await discard_pending_updates(telegram)

    assert next_offset == 43
    telegram.get_updates.assert_awaited_once_with(
        offset=-1,
        timeout=0,
        allowed_updates=ALLOWED_UPDATES,
    )


@pytest.mark.asyncio
async def test_discard_pending_updates_returns_zero_when_no_backlog():
    from src.scripts.serve_telegram_bot import discard_pending_updates

    telegram = AsyncMock()
    telegram.get_updates = AsyncMock(return_value=[])

    assert await discard_pending_updates(telegram) == 0


@pytest.mark.asyncio
async def test_initial_update_offset_discards_backlog_only_in_local_mode(monkeypatch, mocker):
    from src.scripts import serve_telegram_bot

    telegram = AsyncMock()
    discard = mocker.patch(
        "src.scripts.serve_telegram_bot.discard_pending_updates",
        new=AsyncMock(return_value=43),
    )

    monkeypatch.setenv("TELEGRAM_LOCAL_META_AGENT", "1")
    assert await serve_telegram_bot.initial_update_offset(telegram) == 43
    discard.assert_awaited_once_with(telegram)

    discard.reset_mock()
    monkeypatch.delenv("TELEGRAM_LOCAL_META_AGENT", raising=False)
    assert await serve_telegram_bot.initial_update_offset(telegram) == 0
    discard.assert_not_called()
