"""Tests for qdrant_tools.py - all Qdrant interaction tools and failure_payload.

Uses mock_qdrant_service fixture from conftest. Tests error paths, DTO registration,
schema, search, filter, scroll, retrieve.
"""
import json
from unittest.mock import MagicMock

import pytest

from src.meta_agent.tools.qdrant_tools import (
    QdrantCollectionSchema,
    QdrantFilterTool,
    QdrantRetrieveTool,
    QdrantScrollTool,
    QdrantSearchTool,
    failure_payload,
    get_qdrant_service,
)


def test_failure_payload():
    """Test standardized error JSON for Qdrant operations."""
    payload = failure_payload("search", "test error")
    data = json.loads(payload)
    assert data["error"] == "ошибка_запроса_qdrant"
    assert data["operation"] == "search"
    assert "test error" in data["detail"]

    # Test with Exception
    exc = ValueError("bad query")
    payload2 = failure_payload("filter", exc)
    data2 = json.loads(payload2)
    assert "bad query" in data2["detail"]

    # Non-exception
    payload3 = failure_payload("scroll", 123)
    data3 = json.loads(payload3)
    assert "123" in data3["detail"]


def test_get_qdrant_service(mock_qdrant_service):
    """Test service getter returns the mocked singleton."""
    import src.meta_agent.tools.qdrant_tools as qtools

    service = qtools.get_qdrant_service()
    assert service == mock_qdrant_service


@pytest.mark.asyncio
async def test_collection_schema_tool(mock_qdrant_service):
    """Test QdrantCollectionSchema tool."""
    tool = QdrantCollectionSchema(reasoning="Need schema", collection="personas")
    mock_qdrant_service.get_collection_schema.return_value = {"name": "personas", "points_count": 100}

    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)
    assert "personas" in result
    assert "points_count" in result
    mock_qdrant_service.get_collection_schema.assert_called_with("personas")


@pytest.mark.asyncio
async def test_qdrant_search_tool(mock_qdrant_service, sample_dto_data):
    """Test semantic search tool with DTO registration."""
    tool = QdrantSearchTool(reasoning="Need matches", collection="personas", query="test query", limit=5)
    mock_qdrant_service.search.return_value = sample_dto_data

    mock_context = MagicMock()
    mock_context.custom_context = {}
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    data = json.loads(result)
    assert "dto_name" in str(data) or isinstance(data, dict)
    mock_qdrant_service.search.assert_called_with(
        collection="personas",
        query="test query",
        vector_name="embedding",
        limit=5,
    )


@pytest.mark.asyncio
async def test_qdrant_filter_and_scroll_tools(mock_qdrant_service):
    """Test filter_points and scroll_points tools."""
    tool_filter = QdrantFilterTool(
        reasoning="Need approved records",
        collection="simulations",
        field="status",
        value="approved",
    )
    mock_qdrant_service.filter_points.return_value = [{"id": 1}]

    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool_filter(mock_context, mock_config)
    assert "points" in result or "dto" in result.lower()

    tool_scroll = QdrantScrollTool(reasoning="Browse collection", collection="questions", limit=10)
    mock_qdrant_service.scroll_points.return_value = {"points": [], "next_offset": None}
    result_scroll = await tool_scroll(mock_context, mock_config)
    assert isinstance(result_scroll, str)


@pytest.mark.asyncio
async def test_qdrant_retrieve_tool(mock_qdrant_service):
    """Test retrieve_by_id with empty ids edge case."""
    tool = QdrantRetrieveTool(reasoning="Need specific point", collection="personas", ids=[])
    mock_context = MagicMock()
    mock_config = MagicMock()

    # Empty ids
    result_empty = await tool(mock_context, mock_config)
    data = json.loads(result_empty)
    assert "error" in data
    assert "ids_empty" in str(data).lower()

    # Normal case
    mock_qdrant_service.retrieve_by_id.return_value = [{"id": "uuid1"}]
    tool_with_ids = QdrantRetrieveTool(reasoning="Need point", collection="personas", ids=["uuid1"])
    result = await tool_with_ids(mock_context, mock_config)
    assert result


@pytest.mark.asyncio
async def test_collection_schema_tool_exception_returns_failure_payload(mock_qdrant_service):
    """Test collection_schema wraps service exceptions into failure_payload."""
    tool = QdrantCollectionSchema(reasoning="Need schema", collection="personas")
    mock_qdrant_service.get_collection_schema.side_effect = RuntimeError("schema failed")

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert payload["error"] == "ошибка_запроса_qdrant"
    assert payload["operation"] == "collection_schema"


@pytest.mark.asyncio
async def test_qdrant_search_tool_exception_returns_failure_payload(mock_qdrant_service):
    """Test search wraps service exceptions into failure_payload."""
    tool = QdrantSearchTool(reasoning="Need matches", collection="personas", query="q")
    mock_qdrant_service.search.side_effect = RuntimeError("search failed")

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert payload["error"] == "ошибка_запроса_qdrant"
    assert payload["operation"] == "search"


@pytest.mark.asyncio
async def test_qdrant_filter_tool_exception_returns_failure_payload(mock_qdrant_service):
    """Test filter_points wraps service exceptions into failure_payload."""
    tool = QdrantFilterTool(reasoning="Need filtered", collection="simulations", field="status", value="ok")
    mock_qdrant_service.filter_points.side_effect = RuntimeError("filter failed")
    result = await tool(MagicMock(), MagicMock())

    payload = json.loads(result)
    assert payload["error"] == "ошибка_запроса_qdrant"
    assert payload["operation"] == "filter_points"


@pytest.mark.asyncio
async def test_qdrant_scroll_tool_exception_returns_failure_payload(mock_qdrant_service):
    """Test scroll_points wraps service exceptions into failure_payload."""
    tool = QdrantScrollTool(reasoning="Need scroll", collection="questions")
    mock_qdrant_service.scroll_points.side_effect = RuntimeError("scroll failed")

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert payload["error"] == "ошибка_запроса_qdrant"
    assert payload["operation"] == "scroll_points"


@pytest.mark.asyncio
async def test_qdrant_retrieve_tool_exception_returns_failure_payload(mock_qdrant_service):
    """Test retrieve_by_id wraps service exceptions into failure_payload."""
    tool = QdrantRetrieveTool(reasoning="Need id", collection="personas", ids=["u1"])
    mock_qdrant_service.retrieve_by_id.side_effect = RuntimeError("retrieve failed")
    
    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert payload["error"] == "ошибка_запроса_qdrant"
    assert payload["operation"] == "retrieve_by_id"


def test_all_qdrant_tools_registered():
    """Verify all Qdrant tools have proper metadata."""
    tools = [QdrantCollectionSchema, QdrantSearchTool, QdrantFilterTool, QdrantScrollTool, QdrantRetrieveTool]
    tool_names = {t.tool_name for t in tools}
    expected = {"collection_schema", "search", "filter_points", "scroll_points", "retrieve_by_id"}
    assert tool_names == expected
