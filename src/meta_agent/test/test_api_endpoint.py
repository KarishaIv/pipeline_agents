"""Tests for the structured POST /ask API endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from meta_agent import (
    AskRequest,
    MetaAgentApiResponse,
    MetaAgentResult,
    TextOutput,
)


@pytest.mark.asyncio
async def test_ask_endpoint_success(mocker):
    """Test successful POST /ask request."""
    from src.scripts.serve_meta_agent import ask_json

    mock_result = MetaAgentResult(answer="Test answer", thread_id="thread-123")
    mocker.patch(
        "src.scripts.serve_meta_agent.meta_graph_manager.invoke_graph_session",
        new=AsyncMock(return_value=mock_result),
    )

    request = AskRequest(question="Test question", thread_id=None)
    response = await ask_json(request)

    assert isinstance(response, MetaAgentApiResponse)
    assert response.thread_id == "thread-123"
    assert len(response.outputs) == 1
    assert isinstance(response.outputs[0], TextOutput)
    assert response.outputs[0].text == "Test answer"


@pytest.mark.asyncio
async def test_ask_endpoint_with_existing_thread(mocker):
    """Test POST /ask with existing thread ID."""
    from src.scripts.serve_meta_agent import ask_json

    mock_result = MetaAgentResult(answer="Follow-up answer", thread_id="thread-123")
    mock_invoke = AsyncMock(return_value=mock_result)
    mocker.patch("src.scripts.serve_meta_agent.meta_graph_manager.invoke_graph_session", new=mock_invoke)

    request = AskRequest(question="Follow-up question", thread_id="thread-123")
    response = await ask_json(request)

    mock_invoke.assert_called_once_with("Follow-up question", "thread-123")
    assert response.thread_id == "thread-123"
    assert response.outputs[0].text == "Follow-up answer"


def test_ask_request_validation():
    """Test AskRequest Pydantic validation."""
    valid = AskRequest(question="Valid question")
    assert valid.question == "Valid question"
    assert valid.thread_id is None

    with pytest.raises(ValueError):
        AskRequest(question="")

    with pytest.raises(ValueError):
        AskRequest(question="x" * 10001)


def test_ask_response_structure():
    """Test MetaAgentApiResponse structure."""
    text_output = TextOutput(text="Hello")
    response = MetaAgentApiResponse(thread_id="t-1", outputs=[text_output])

    assert response.thread_id == "t-1"
    assert len(response.outputs) == 1
    assert response.outputs[0].text == "Hello"

    response_dict = response.model_dump()
    assert "thread_id" in response_dict
    assert "outputs" in response_dict
    assert response_dict["outputs"][0]["type"] == "text"
