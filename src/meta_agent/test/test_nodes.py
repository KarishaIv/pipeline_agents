"""Tests for nodes.py - supervisor_node, data_extractor_node, analyzer_node, code_writer_node.

Uses mock_run_agent fixture. Tests iteration limits, decision parsing, DTO extraction,
fallback JSON, routing decisions, MAX_* constants enforcement.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.meta_agent.configs import MAX_HISTORY_CHARS, MAX_SUPERVISOR_ITERATIONS
from src.meta_agent.nodes import (
    analyzer_node,
    code_writer_node,
    data_extractor_node,
    supervisor_node,
)
from src.meta_agent.utils.json_responses import json_node_failure
from src.meta_agent.utils.state import MetaAgentState
from src.meta_agent.workers import _extract_dto_store


def test_fallback_worker_payload():
    """Test fallback when JSON parsing of agent output fails."""
    payload = json_node_failure(
        worker="supervisor",
        raw_output="Invalid JSON output",
        expected_tool="supervisor_decision",
    )
    data = json.loads(payload)
    assert data["status"] == "failed"
    assert data["worker"] == "supervisor"
    assert data["expected_report_tool"] == "supervisor_decision"
    assert "Invalid JSON output" in data["raw_output"]


def test_extract_dto_store():
    """Test extraction of dto_store from agent run result context."""
    store = _extract_dto_store({"dto_store": {"test_dto": {"data": [1]}}})
    assert "test_dto" in store
    assert store["test_dto"]["data"] == [1]

    # No context case
    assert _extract_dto_store(None) == {}


@pytest.mark.asyncio
async def test_supervisor_node(mock_run_agent, meta_state, mocker):
    """Test supervisor_node with decision parsing and iteration cap."""
    mocker.patch(
        "src.meta_agent.nodes.summarize_history_text",
        new=AsyncMock(return_value="compressed history"),
    )
    mock_run_agent.return_value = MagicMock(
        output=json.dumps(
            {
                "reasoning": "Need data first",
                "next": "data_extractor",
                "task": "Extract data",
                "final_answer": "",
            }
        ),
        context={"custom_context": {}},
    )

    state = meta_state.model_copy()
    state.iterations = 2
    result = await supervisor_node(state, MagicMock())

    assert result["next_worker"] == "data_extractor"
    assert "Extract data" in result.get("current_task", "")
    assert result["iterations"] == 3
    mock_run_agent.assert_called()


@pytest.mark.asyncio
async def test_supervisor_node_max_iterations(meta_state, mocker):
    """Test MAX_SUPERVISOR_ITERATIONS enforcement leads to final answer."""
    mocker.patch(
        "src.meta_agent.nodes.summarize_history_text",
        new=AsyncMock(return_value="compressed history"),
    )
    state = meta_state.model_copy()
    state.iterations = MAX_SUPERVISOR_ITERATIONS + 1

    result = await supervisor_node(state, MagicMock())

    assert result["next_worker"] == "end"
    assert result["iterations"] == MAX_SUPERVISOR_ITERATIONS + 2
    assert result.get("outputs")


@pytest.mark.asyncio
async def test_data_extractor_node(mock_run_agent, meta_state, mocker):
    """Test data_extractor_node with Qdrant tools and DTO merging."""
    mock_run_agent.return_value = MagicMock(
        output=json.dumps(
            {
                "reasoning": "Collected required slices",
                "completed_steps": ["schema", "search"],
                "summary": "Data extracted",
                "dto_references": "dto1",
                "status": "COMPLETED",
            }
        ),
        context={"custom_context": {"dto_store": {"dto1": {}}}},
    )
    mocker.patch("src.meta_agent.workers._extract_dto_store", return_value={"dto1": {}})

    state = meta_state.model_copy()
    state.current_task = "Extract personas data"
    result = await data_extractor_node(state, MagicMock())

    assert "dto_store" in result
    assert "history" in result
    assert "Data extracted" in result["history"][-1]["content"]
    assert len(result["history"]) == 1


@pytest.mark.asyncio
async def test_analyzer_node_report_vs_delegate(mock_run_agent, meta_state, mocker):
    """Test analyzer_node decision logic for report vs delegate to code_writer."""
    # Report path
    mock_run_agent.return_value = MagicMock(
        output=json.dumps(
            {
                "reasoning": "Enough signal for conclusions",
                "decision": "report",
                "key_findings": ["pattern found"],
                "conclusions": "Main pattern is stable.",
                "status": "completed",
            }
        ),
        context={"dto_store": {}},
    )
    state = meta_state.model_copy()
    state.current_task = "Analyze data"
    result = await analyzer_node(state, MagicMock())
    assert result.get("next_worker") == "supervisor"
    assert "pattern found" in result["history"][-1]["content"]

    # Delegate path (within limits)
    mock_run_agent.return_value = MagicMock(
        output=json.dumps(
            {
                "reasoning": "Need chart for comparison",
                "decision": "delegate",
                "task": "Create chart",
                "delegate_reason": "Visualization required",
                "status": "completed",
            }
        ),
        context={"dto_store": {}},
    )
    state.delegated_attempts = 1
    result2 = await analyzer_node(state, MagicMock())
    assert result2.get("next_worker") == "code_writer"


@pytest.mark.asyncio
async def test_analyzer_node_truncates_large_prior_data(mock_run_agent, meta_state):
    """Regression: analyzer task should use summary for large prior worker history."""
    mock_run_agent.return_value = MagicMock(
        output=json.dumps(
            {
                "reasoning": "Enough signal for conclusions",
                "decision": "report",
                "key_findings": ["ok"],
                "conclusions": "done",
                "status": "completed",
            }
        ),
        context={"dto_store": {}},
    )

    state = meta_state.model_copy()
    state.current_task = "Analyze huge context"
    state.history = [
        {"role": "data_extractor", "content": f"chunk-{i}-" + ("x" * 3000)}
        for i in range(12)
    ]

    with patch(
        "src.meta_agent.nodes.build_role_history_text_async",
        new=AsyncMock(return_value="[HISTORY_SUMMARY]: compressed history"),
    ):
        await analyzer_node(state, MagicMock())

    task_text = mock_run_agent.await_args.kwargs["task"]
    assert "[HISTORY_SUMMARY]:" in task_text
    assert len(task_text) <= MAX_HISTORY_CHARS + 500


@pytest.mark.asyncio
async def test_code_writer_node(mock_run_agent, meta_state, mocker):
    """Test code_writer_node with BIG_MODEL, task validation, and return to analyzer."""
    mock_run_agent.return_value = MagicMock(
        output=json.dumps(
            {
                "reasoning": "Validated and executed",
                "task": "Compute statistics",
                "code": "print('ok')",
                "validation": '{"is_runnable": true}',
                "execution": '{"output": "ok"}',
                "findings": ["done"],
                "status": "COMPLETED",
            }
        ),
        context={"custom_context": {}},
    )
    mocker.patch("src.meta_agent.workers._extract_dto_store", return_value={})

    state = meta_state.model_copy()
    state.current_task = "Compute statistics"
    result = await code_writer_node(state, MagicMock())

    assert result["next_worker"] == "analyzer"
    assert result["current_task"] == "Compute statistics"
    mock_run_agent.assert_called()  # with BIG_MODEL


def test_nodes_import_and_structure():
    """Verify all nodes and helpers are importable and have expected signatures."""
    import src.meta_agent.nodes as nodes_mod

    assert hasattr(nodes_mod, "supervisor_node")
    assert hasattr(nodes_mod, "data_extractor_node")
    assert hasattr(nodes_mod, "analyzer_node")
    assert hasattr(nodes_mod, "code_writer_node")
    assert nodes_mod.MAX_SUPERVISOR_ITERATIONS == MAX_SUPERVISOR_ITERATIONS


@pytest.mark.asyncio
async def test_supervisor_node_non_json_fallback(mock_run_agent, meta_state, mocker):
    """Regression test: when supervisor worker returns non-JSON (parse failure),
    it should treat the raw output as the final answer and end the graph.
    """
    mocker.patch(
        "src.meta_agent.nodes.summarize_history_text",
        new=AsyncMock(return_value="compressed history"),
    )
    plain_output = "This is a plain text final answer from the supervisor."
    mock_run_agent.return_value = MagicMock(
        output=plain_output,
        context={"custom_context": {}},
    )

    state = meta_state.model_copy()
    state.iterations = 3
    result = await supervisor_node(state, MagicMock())

    assert result["next_worker"] == "end"
    assert len(result["outputs"]) == 1
    assert result["outputs"][0]["text"] == plain_output
    assert result["iterations"] == 4
    assert len(result.get("history", [])) == 1
    assert result["history"][0]["role"] == "supervisor"
    assert result["history"][0]["content"] == plain_output
    mock_run_agent.assert_called()
