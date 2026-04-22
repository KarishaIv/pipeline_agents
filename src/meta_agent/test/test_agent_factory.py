"""Tests for agent_factory.py - make_agent, run_agent, _unwrap, client creation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.meta_agent.agent_factory import (
    AgentRunResult,
    _safe_get_custom_context,
    _unwrap,
    make_agent,
    make_openai_client,
    run_agent,
)


def test_unwrap_function():
    """Test _unwrap normalizes various agent outputs to string.

    None case now returns structured error JSON instead of raising,
    to catch the common 'NoneType' subscriptable error from sgr_agent_core.
    """
    assert _unwrap("direct string") == "direct string"
    assert _unwrap(SimpleNamespace(execution_result="from exec")) == "from exec"
    assert _unwrap(SimpleNamespace(answer="from answer")) == "from answer"
    assert _unwrap(123) == "123"  # fallback str()

    # Test the None case that previously raised RuntimeError (now gracefully returns error JSON)
    error_output = _unwrap(None)
    assert isinstance(error_output, str)
    assert "error" in error_output
    assert "no tool_call selected" in error_output
    assert "None" in error_output or "tool_call" in error_output


def test_safe_get_custom_context():
    """Test _safe_get_custom_context handles missing attrs gracefully."""
    mock_agent = MagicMock()
    mock_agent._context.custom_context = {"key": "value"}
    assert _safe_get_custom_context(mock_agent) == {"key": "value"}

    # Error case
    bad_agent = MagicMock()
    del bad_agent._context
    assert _safe_get_custom_context(bad_agent) is None


def test_safe_get_custom_context_handles_context_access_exception():
    """Test _safe_get_custom_context returns None on context access errors."""
    class BrokenContext:
        def __getattribute__(self, name):
            if name == "custom_context":
                raise RuntimeError("boom")
            return object.__getattribute__(self, name)

    bad_agent = SimpleNamespace(_context=BrokenContext())
    assert _safe_get_custom_context(bad_agent) is None


@patch("src.meta_agent.agent_factory.AsyncOpenAI")
@patch("src.meta_agent.agent_factory.wrap_openai")
def test_make_openai_client(mock_wrap, mock_asyncopenai, monkeypatch):
    """Test client creation with Yandex config and optional LangSmith wrapping."""
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_FOLDER_ID", "test-folder")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    client = make_openai_client()
    assert mock_asyncopenai.called
    # Wrapping happens when tracing enabled
    assert mock_wrap.called


@patch("src.meta_agent.agent_factory.AsyncOpenAI")
@patch("src.meta_agent.agent_factory.wrap_openai")
def test_make_openai_client_without_tracing(mock_wrap, mock_asyncopenai, monkeypatch):
    """Test client is not wrapped when tracing is disabled."""
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    make_openai_client()
    assert mock_asyncopenai.called
    assert not mock_wrap.called


def test_make_agent_wires_config_and_initial_context(mocker):
    """Test make_agent passes expected config and sets initial context."""
    fake_client = MagicMock()
    fake_agent = MagicMock()
    fake_agent._context = SimpleNamespace(custom_context={})
    tool_calling_agent = mocker.patch("src.meta_agent.agent_factory.ToolCallingAgent", return_value=fake_agent)
    mocker.patch("src.meta_agent.agent_factory.make_openai_client", return_value=fake_client)
    mock_get_model = mocker.patch("src.meta_agent.agent_factory.get_model_uri", return_value="model://x")

    agent = make_agent(
        task="Do task",
        system_prompt="System prompt",
        toolkit=["tool_a"],
        name="worker",
        model="aliceai-llm",
        initial_custom_context={"dto_store": {"d1": {}}},
    )

    assert agent is fake_agent
    assert fake_agent._context.custom_context == {"dto_store": {"d1": {}}}
    mock_get_model.assert_called_once_with("aliceai-llm")
    tool_calling_agent.assert_called_once()
    call_kwargs = tool_calling_agent.call_args.kwargs
    assert call_kwargs["def_name"] == "worker"
    assert call_kwargs["toolkit"] == ["tool_a"]
    assert call_kwargs["task_messages"][0]["content"] == "Do task"
    assert call_kwargs["agent_config"].prompts.system_prompt_str == "System prompt"


@pytest.mark.asyncio
async def test_run_agent_success(mocker):
    """Test happy path for run_agent."""
    agent = MagicMock()
    agent.execute = AsyncMock(return_value="Test answer from agent")
    agent._context.custom_context = {"dto_store": {"dto1": {"rows": []}}}
    mocker.patch("src.meta_agent.agent_factory.make_agent", return_value=agent)

    result = await run_agent(
        task="Test question",
        system_prompt="Test prompt",
        toolkit=[],
        name="test-agent",
    )

    assert isinstance(result, AgentRunResult)
    assert "Test answer" in result.output
    assert result.context == {"dto_store": {"dto1": {"rows": []}}}
    agent.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_error_handling(mocker):
    """Test error path returns JSON error payload."""
    agent = MagicMock()
    agent.execute = AsyncMock(side_effect=Exception("LLM failure"))
    agent._context.custom_context = {"dto_store": {}}
    mocker.patch("src.meta_agent.agent_factory.make_agent", return_value=agent)

    result = await run_agent(
        task="Failing question",
        system_prompt="Test",
        toolkit=[],
        name="test-agent",
    )

    assert isinstance(result.output, str)
    assert "error" in result.output.lower() or "LLM failure" in result.output


def test_agent_factory_metadata():
    """Basic checks for exported functions."""
    assert callable(make_agent)
    assert callable(run_agent)
    assert callable(_unwrap)
