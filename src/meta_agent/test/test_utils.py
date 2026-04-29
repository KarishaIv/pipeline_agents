"""Tests for meta_agent utils: history, state, and routing.

All test logic is contained within src/meta_agent/test/ as specified.
Tests cover pure functions with parametrized edge cases, reducers, truncation logic.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.meta_agent.configs import MAX_HISTORY_CHARS
from src.meta_agent.utils.history import (
    _default_history_summarizer,
    build_persisted_history,
    build_role_history_text_async,
    summarize_history_list,
    truncate_history,
    truncate_history_list,
    truncate_output_value,
)
from src.meta_agent.utils.routing import route_analyzer, route_supervisor
from src.meta_agent.utils.state import (
    MetaAgentState,
    append_history,
    build_turn_state_update,
    merge_dto_store,
    state_to_dict,
)


def test_truncate_output_value():
    """Test truncate_output_value for different types and lengths."""
    # String truncation
    long_str = "x" * 100
    assert truncate_output_value(long_str, 10) == "xxxxxxxxxx..."
    assert truncate_output_value("short", 10) == "short"

    # JSON for list/dict
    data = {"key": "value" * 20}
    truncated = truncate_output_value(data, 20)
    assert isinstance(truncated, str)
    assert "..." in truncated or len(truncated) <= 20

    # Other types unchanged
    assert truncate_output_value(42, 10) == 42
    assert truncate_output_value(None, 10) is None


@pytest.mark.parametrize(
    "history,expected_length,should_truncate",
    [
        ([], 0, False),
        ([{"role": "user", "content": "short"}], 1, False),
        # Long history should truncate
        ([{"role": "user", "content": "x" * 1000}] * 20, 1, True),
    ],
)
def test_truncate_history_and_list(history, expected_length, should_truncate):
    """Test history truncation functions with various inputs."""
    text_result = truncate_history(history)
    assert isinstance(text_result, str)

    list_result = truncate_history_list(history)
    assert isinstance(list_result, list)
    assert len(list_result) <= len(history) or len(history) == 0

    if should_truncate and history:
        assert "…" in text_result or "история обрезана" in text_result


@pytest.mark.asyncio
async def test_build_persisted_history():
    """Test build_persisted_history stores result history and assistant answer."""
    result = {
        "answer": "Test answer",
        "history": [{"role": "user", "content": "Previous"}],
    }
    persisted = await build_persisted_history(result)
    assert isinstance(persisted, list)
    assert len(persisted) >= 2  # existing history + assistant
    assert any("Test answer" in str(msg.get("content", "")) for msg in persisted)


@pytest.mark.asyncio
async def test_summarize_history_list_uses_llm_for_long_history():
    """Long history should be compressed with LLM summary instead of dropping old messages."""
    history = [{"role": "user", "content": f"message-{i}-" + ("x" * 3000)} for i in range(8)]

    async def fake_summarizer(_: str) -> str:
        return "Краткое резюме предыдущих шагов"

    summarized = await summarize_history_list(history, summarizer=fake_summarizer)
    assert summarized[0]["role"] == "history_summary"
    assert "Краткое резюме" in summarized[0]["content"]
    assert any("message-7-" in item["content"] for item in summarized)


@pytest.mark.asyncio
async def test_summarize_history_list_fallbacks_to_truncation_on_empty_summary():
    """If LLM summary is empty, code should keep deterministic truncation fallback."""
    history = [{"role": "user", "content": f"message-{i}-" + ("x" * 3000)} for i in range(8)]

    async def empty_summarizer(_: str) -> str:
        return "   "

    summarized = await summarize_history_list(history, summarizer=empty_summarizer)
    assert summarized
    assert summarized[0]["role"] != "history_summary"


@pytest.mark.asyncio
async def test_build_role_history_text_async_uses_summary_when_history_is_large():
    """Role-specific history text should include generated summary marker for oversized context."""
    history = [{"role": "data_extractor", "content": "x" * (MAX_HISTORY_CHARS // 2)} for _ in range(3)]

    async def fake_summarizer(_: str) -> str:
        return "Краткое резюме истории"

    text = await build_role_history_text_async(
        history,
        roles=("data_extractor",),
        summarizer=fake_summarizer,
    )
    assert "[HISTORY_SUMMARY]:" in text
    assert "Краткое резюме истории" in text


@pytest.mark.asyncio
async def test_build_role_history_text_async_keeps_existing_history_summary():
    """Existing history_summary entries should not be filtered out by role selection."""
    history = [
        {"role": "history_summary", "content": "already summarized context"},
        {"role": "data_extractor", "content": "fresh worker result"},
        {"role": "user", "content": "must be filtered"},
    ]

    text = await build_role_history_text_async(history, roles=("data_extractor",))
    assert "[HISTORY_SUMMARY]: already summarized context" in text
    assert "[DATA_EXTRACTOR]: fresh worker result" in text
    assert "[USER]:" not in text


@pytest.mark.asyncio
async def test_default_history_summarizer_passes_max_tokens_to_llm(mocker, monkeypatch):
    """Default summarizer should propagate max_tokens into chat completion request."""
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "test-folder")

    create_mock = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary text"))]
        )
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )

    mocker.patch("src.meta_agent.utils.history.make_openai_client", return_value=fake_client)

    out = await _default_history_summarizer("history body", max_tokens=321)
    assert out == "summary text"
    assert create_mock.await_args.kwargs["max_tokens"] == 321


def test_append_history_reducer():
    """Test append_history LangGraph reducer."""
    assert append_history([], None) == []
    assert append_history([{"role": "user", "content": "1"}], {"role": "assistant", "content": "2"}) == [
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "2"},
    ]
    new_msgs = [{"role": "user", "content": "3"}, {"role": "assistant", "content": "4"}]
    assert append_history([], new_msgs) == new_msgs
    assert append_history(None, new_msgs) == new_msgs


def test_merge_dto_store_reducer():
    """Test merge_dto_store LangGraph reducer."""
    from src.meta_agent.dto import DtoPayload
    
    # Create proper DtoPayload objects
    dto1_left = DtoPayload(
        summary_text="Left DTO 1",
        columns=["data"],
        num_rows=2,
        sample=[{"data": 1}],
        rows=[{"data": 1}, {"data": 2}],
    )
    dto1_right = DtoPayload(
        summary_text="Right DTO 1",
        columns=["data"],
        num_rows=2,
        sample=[{"data": 3}],
        rows=[{"data": 3}, {"data": 4}],
    )
    dto2 = DtoPayload(
        summary_text="DTO 2",
        columns=["data"],
        num_rows=1,
        sample=[{"data": 5}],
        rows=[{"data": 5}],
    )
    
    left = {"dto1": dto1_left}
    right = {"dto1": dto1_right, "dto2": dto2}
    merged = merge_dto_store(left, right)
    
    # Right wins for dto1
    assert isinstance(merged["dto1"], DtoPayload)
    assert merged["dto1"].rows == [{"data": 3}, {"data": 4}]
    # dto2 added
    assert isinstance(merged["dto2"], DtoPayload)
    assert merged["dto2"].rows == [{"data": 5}]
    
    # Test edge cases
    assert merge_dto_store({}, None) == {}
    assert merge_dto_store({}, right) == right


def test_state_to_dict():
    """Test state_to_dict handles both Pydantic and dict inputs."""
    state_dict = {"question": "test", "answer": "ok"}
    assert state_to_dict(state_dict) == state_dict

    state_obj = MetaAgentState(question="test")
    converted = state_to_dict(state_obj)
    assert isinstance(converted, dict)
    assert converted["question"] == "test"


def test_build_turn_state_update():
    """Test build_turn_state_update resets control fields and returns history delta."""
    snapshot = {
        "history": [{"role": "user", "content": "old"}],
        "dto_store": {"existing": {"data": [1]}},
    }
    question = "New analysis question"

    update = build_turn_state_update(question, snapshot)

    assert update["question"] == question
    assert update["iterations"] == 0
    assert update["delegated_attempts"] == 0
    assert update["next_worker"] == ""
    assert update["current_task"] == ""
    assert update["answer"] == ""
    assert "existing" in update["dto_store"]
    assert len(update["history"]) == 1
    assert update["history"][-1]["content"] == question


def test_route_supervisor():
    """Test supervisor routing based on next_worker."""
    assert route_supervisor({"next_worker": "data_extractor"}) == "data_extractor"
    assert route_supervisor({"next_worker": "analyzer"}) == "analyzer"
    assert route_supervisor({}) == "end"  # default
    assert route_supervisor(MetaAgentState(question="test", next_worker="end")) == "end"


def test_route_analyzer():
    """Test analyzer routing: code_writer or back to supervisor."""
    assert route_analyzer({"next_worker": "code_writer"}) == "code_writer"
    assert route_analyzer({"next_worker": "report"}) == "supervisor"
    assert route_analyzer({}) == "supervisor"  # default
    assert route_analyzer(MetaAgentState(question="test", next_worker="code_writer")) == "code_writer"


@pytest.mark.parametrize("max_chars", [100, None, 5000])
def test_truncate_history_list_edge_cases(max_chars):
    """Parametrized test for truncate_history_list edge cases."""
    long_history = [{"role": "user", "content": "x" * 1000}] * 5
    result = truncate_history_list(long_history, max_chars)
    assert isinstance(result, list)
    assert len(result) > 0 or not long_history


def test_extract_text_content_various_types():
    """Test _extract_text_content handles string, list of dicts, objects with .type, and fallback."""
    from src.meta_agent.utils.history import _extract_text_content

    assert _extract_text_content("plain string") == "plain string"
    assert _extract_text_content(None) == ""
    assert _extract_text_content(123) == "123"

    # List of dicts with type=text (OpenAI style)
    content_list = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " world"}
    ]
    assert _extract_text_content(content_list) == "Hello world"

    # List of objects with .type attribute
    class TextPart:
        def __init__(self, text):
            self.type = "text"
            self.text = text
    obj_list = [TextPart("foo"), TextPart("bar")]
    assert _extract_text_content(obj_list) == "foobar"

    # Mixed fallback
    mixed = [{"type": "image"}, "text only"]
    assert "text only" in _extract_text_content(mixed)


@pytest.mark.asyncio
async def test_default_history_summarizer_missing_credentials(monkeypatch):
    """Test that summarizer gracefully returns empty string when Yandex creds are missing."""
    from src.meta_agent.utils.history import _default_history_summarizer

    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)

    result = await _default_history_summarizer("some history")
    assert result == ""


@pytest.mark.asyncio
async def test_default_history_summarizer_empty_response(mocker, monkeypatch):
    """Test summarizer handles empty choices or no content from LLM."""
    from src.meta_agent.utils.history import _default_history_summarizer

    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "test-folder")

    # Case 1: no choices
    create_mock1 = AsyncMock(return_value=SimpleNamespace(choices=[]))
    fake_client1 = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock1))
    )
    mocker.patch("src.meta_agent.utils.history.make_openai_client", return_value=fake_client1)
    assert await _default_history_summarizer("history") == ""

    # Case 2: empty content
    create_mock2 = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
        )
    )
    fake_client2 = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock2))
    )
    mocker.patch("src.meta_agent.utils.history.make_openai_client", return_value=fake_client2)
    assert await _default_history_summarizer("history") == ""
