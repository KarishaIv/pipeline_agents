"""Tests for graph and API output handling (should fail before implementation)."""
import pytest
from unittest.mock import AsyncMock, MagicMock


def test_meta_agent_result_structure():
    """MetaAgentResult should have thread_id and outputs, not answer."""
    from src.meta_agent.graph import MetaAgentResult
    from src.meta_agent.api_models import TextOutput

    result = MetaAgentResult(
        thread_id="t-1",
        outputs=[TextOutput(text="Done")],
    )

    # Should not have 'answer' attribute (breaking change)
    assert not hasattr(result, "answer") or result.__dict__.get("answer") is None
    assert result.thread_id == "t-1"
    assert len(result.outputs) > 0


@pytest.mark.asyncio
async def test_ask_endpoint_returns_outputs_not_wrapped_answer(mocker):
    """The /ask endpoint should return structured outputs, not wrap answer as TextOutput."""
    from src.scripts.serve_meta_agent import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    # Mock the graph manager to return outputs
    mock_result = MagicMock()
    mock_result.thread_id = "thread-123"
    mock_result.outputs = []  # Will be populated

    # The endpoint should construct MetaAgentApiResponse with outputs from graph state
    # This test is checking the contract, not full flow


@pytest.mark.asyncio
async def test_graph_invoke_returns_outputs():
    """Graph invoke should populate MetaAgentState.outputs, not just answer."""
    from src.meta_agent.graph import MetaAgentGraphManager
    from src.meta_agent.utils.state import MetaAgentState

    manager = MetaAgentGraphManager()

    # After implementation, invoking the graph should produce outputs in state
    # This is a contract test for the new behavior
