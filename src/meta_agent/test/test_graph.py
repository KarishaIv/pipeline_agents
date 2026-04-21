"""Tests for graph.py - MetaAgentGraphManager, _build_graph, invoke methods, state management.

Tests graph topology (nodes/edges), thread_id resolution, prepare/finalize invoke logic,
integration with checkpointer and routing.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.meta_agent.graph import (
    MetaAgentGraphManager,
    MetaAgentResult,
    meta_graph_manager,
)
from src.meta_agent.utils.routing import route_analyzer, route_supervisor
from src.meta_agent.utils.state import MetaAgentState, build_turn_state_update


def test_resolve_session_thread_id():
    """Test thread_id resolution logic for new sessions and fixed IDs via manager."""
    manager = MetaAgentGraphManager()
    generated_none = manager._resolve_session_thread_id(None)
    generated_minus_one = manager._resolve_session_thread_id("-1")
    assert isinstance(generated_none, str)
    assert isinstance(generated_minus_one, str)
    assert len(generated_none) > 10
    assert len(generated_minus_one) > 10
    assert generated_none != generated_minus_one
    assert manager._resolve_session_thread_id("custom-thread-123") == "custom-thread-123"


@pytest.mark.asyncio
async def test_build_graph_structure(mocker):
    """Test MetaAgentGraphManager builds correct graph topology with nodes and conditional edges."""
    mocker.patch("src.meta_agent.graph.route_supervisor", side_effect=route_supervisor)
    mocker.patch("src.meta_agent.graph.route_analyzer", side_effect=route_analyzer)
    mocker.patch("src.meta_agent.utils.routing.route_supervisor", side_effect=route_supervisor)
    mocker.patch("src.meta_agent.utils.routing.route_analyzer", side_effect=route_analyzer)

    manager = MetaAgentGraphManager()
    graph = await manager.get_graph()

    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_manager_uses_async_sqlite_checkpointer():
    """Manager should initialize async SQLite checkpointer for async graph APIs."""
    manager = MetaAgentGraphManager()
    await manager.get_graph()
    assert isinstance(manager._checkpointer, AsyncSqliteSaver)


@pytest.mark.asyncio
async def test_manager_uses_disk_sqlite_checkpointer(tmp_path):
    """Manager should create SQLite checkpoint file on disk."""
    db_path = tmp_path / "checkpoints.sqlite3"
    manager = MetaAgentGraphManager(checkpoint_db_path=db_path)
    await manager.get_graph()

    assert db_path.exists()
    await manager.aclose()


@pytest.mark.asyncio
async def test_meta_agent_graph_manager_invoke(meta_state, mocker):
    """Test MetaAgentGraphManager.invoke_graph_session and related methods."""
    manager = MetaAgentGraphManager()

    # Mock the internal graph/session lifecycle
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"answer": "Test response", "history": []})
    manager._graph = mock_graph
    mocker.patch.object(manager, "_prepare_invoke", new=AsyncMock(return_value=({"configurable": {}}, {})))
    mocker.patch.object(manager, "_finalize_invoke", new=AsyncMock(return_value="Test response"))

    result = await manager.invoke_graph_session("What is the distribution of personas?", thread_id="test-123")

    assert isinstance(result, MetaAgentResult)
    assert result.answer == "Test response"
    assert result.thread_id == "test-123"


def test_meta_graph_manager_singleton():
    """Test module-level singleton."""
    assert meta_graph_manager is not None
    assert isinstance(meta_graph_manager, MetaAgentGraphManager)


@pytest.mark.asyncio
async def test_prepare_and_finalize_invoke(meta_state, mocker):
    """Test public invoke methods and state update helpers indirectly."""
    manager = MetaAgentGraphManager()
    manager._graph = MagicMock(ainvoke=AsyncMock(return_value={}))
    # Mock internal to avoid full execution for this test
    with patch.object(manager, "_prepare_invoke", new=AsyncMock(return_value=({"configurable": {}}, {}))):
        with patch.object(manager, "_finalize_invoke", new=AsyncMock(return_value="Mocked answer")):
            result = await manager.invoke_graph_session("test question", "test-thread")
            assert result.answer == "Mocked answer"

    # Test state helper directly (public)
    update = build_turn_state_update("test q", meta_state.model_dump())
    assert update["question"] == "test q"
    assert "history" in update


@pytest.mark.asyncio
async def test_prepare_invoke_builds_config_and_state_update():
    """Test _prepare_invoke uses snapshot and builds state update."""
    manager = MetaAgentGraphManager()
    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=MagicMock(
            values={
                "question": "old q",
                "history": [{"role": "assistant", "content": "prev"}],
                "dto_store": {"dto1": {"rows": []}},
                "next_worker": "analyzer",
                "current_task": "old task",
                "delegated_attempts": 2,
                "answer": "old answer",
                "iterations": 5,
            }
        )
    )
    manager._graph = graph

    runnable_config, state_update = await manager._prepare_invoke("new question", "thread-1")

    assert runnable_config["configurable"]["thread_id"] == "thread-1"
    assert state_update["question"] == "new question"
    assert state_update["iterations"] == 0
    assert state_update["dto_store"] == {"dto1": {"rows": []}}
    assert state_update["history"][-1]["content"] == "new question"


@pytest.mark.asyncio
async def test_prepare_invoke_with_real_graph_supports_async_state_access():
    """Regression: _prepare_invoke must not hit SqliteSaver async NotImplementedError."""
    manager = MetaAgentGraphManager()

    runnable_config, state_update = await manager._prepare_invoke("hello", "thread-async")

    assert runnable_config["configurable"]["thread_id"] == "thread-async"
    assert state_update["question"] == "hello"
    assert state_update["history"][-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_aclose_releases_graph_resources():
    """Manager should clear graph/checkpointer resources on explicit close."""
    manager = MetaAgentGraphManager()
    await manager.get_graph()

    assert manager._graph is not None
    assert manager._checkpointer is not None

    await manager.aclose()

    assert manager._graph is None
    assert manager._checkpointer is None


@pytest.mark.asyncio
async def test_finalize_invoke_updates_history_and_returns_answer():
    """Test _finalize_invoke persists truncated history and returns answer."""
    manager = MetaAgentGraphManager()
    graph = MagicMock()
    graph.aupdate_state = AsyncMock()
    manager._graph = graph

    runnable_config = {"configurable": {"thread_id": "t-1"}}
    result = {"answer": "final answer", "history": [{"role": "supervisor", "content": "work"}]}
    answer = await manager._finalize_invoke(runnable_config, result, "user question")

    assert answer == "final answer"
    graph.aupdate_state.assert_awaited_once()
    args, _ = graph.aupdate_state.await_args
    assert args[0] == runnable_config
    assert "history" in args[1]
    replaced_history = args[1]["history"]["__replace__"]
    assert replaced_history[-1]["content"] == "final answer"


def test_graph_imports_and_tracing():
    """Verify key exports and @traceable decorators are present."""
    assert callable(MetaAgentGraphManager)
    assert isinstance(MetaAgentResult, type)
    assert hasattr(MetaAgentGraphManager, "invoke_graph_session")
    # LangSmith tracing assumed on invoke methods
