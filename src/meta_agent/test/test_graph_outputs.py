"""Tests for graph and API structured output handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def test_meta_agent_result_structure():
    """MetaAgentResult should have thread_id and outputs, not answer."""
    from src.meta_agent.graph import MetaAgentResult
    from src.meta_agent.api_models import TextOutput

    result = MetaAgentResult(
        thread_id="t-1",
        outputs=[TextOutput(text="Done")],
    )

    assert not hasattr(result, "answer")
    assert result.thread_id == "t-1"
    assert result.outputs[0].text == "Done"


@pytest.mark.asyncio
async def test_ask_endpoint_returns_outputs_not_wrapped_answer(mocker):
    """The /ask endpoint should return structured outputs, not wrap answer as TextOutput."""
    from src.meta_agent import MetaAgentResult, TextOutput
    from src.scripts.serve_meta_agent import ask_json
    from src.meta_agent.api_models import AskRequest, MetaAgentApiResponse

    mock_result = MetaAgentResult(thread_id="thread-123", outputs=[TextOutput(text="Done")])
    mock_invoke = AsyncMock(return_value=mock_result)
    mocker.patch(
        "src.scripts.serve_meta_agent.meta_graph_manager.invoke_graph_session",
        new=mock_invoke,
    )

    response = await ask_json(AskRequest(question="Summarize"))

    assert isinstance(response, MetaAgentApiResponse)
    assert response.thread_id == "thread-123"
    assert response.outputs == [TextOutput(text="Done")]
    mock_invoke.assert_awaited_once_with("Summarize", None)


@pytest.mark.asyncio
async def test_graph_invoke_returns_outputs():
    """invoke_graph should return finalized structured outputs."""
    from src.meta_agent.graph import MetaAgentGraphManager
    from src.meta_agent.api_models import TextOutput

    manager = MetaAgentGraphManager()
    manager.invoke_graph_session = AsyncMock(
        return_value=MagicMock(outputs=[TextOutput(text="Final answer")])
    )

    outputs = await manager.invoke_graph("Question")

    assert outputs == [TextOutput(text="Final answer")]
    manager.invoke_graph_session.assert_awaited_once_with("Question", "-1")
