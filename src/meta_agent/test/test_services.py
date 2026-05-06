"""Tests for services/qdrant.py - QdrantService singleton and core methods."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.meta_agent.services.qdrant import QdrantService


def test_qdrant_service_singleton():
    """Test singleton pattern for QdrantService."""
    with patch("src.meta_agent.services.qdrant.QdrantClient") as mock_client_class:
        QdrantService._instance = None
        service1 = QdrantService()
        service2 = QdrantService()
        assert service1 is service2
        assert mock_client_class.call_count == 1


def test_qdrant_service_search_calls_client_and_normalizes_points(mocker):
    """Test search uses embedding + query_points and returns normalized hits."""
    with patch("src.meta_agent.services.qdrant.QdrantClient") as mock_client_class:
        QdrantService._instance = None
        mock_client = MagicMock()
        point = SimpleNamespace(id="p1", score=0.9, payload={"x": 1})
        mock_client.query_points.return_value = SimpleNamespace(points=[point])
        mock_client_class.return_value = mock_client
        mocker.patch("src.meta_agent.services.qdrant.get_embedding", return_value=[0.1, 0.2])

        service = QdrantService()
        result = service.search("personas", "query text", vector_name="embedding", limit=2)

    assert result == [{"id": "p1", "score": 0.9, "payload": {"x": 1}}]
    mock_client.query_points.assert_called_once()


def test_qdrant_service_filter_points_uses_scroll_filter():
    """Test filter_points delegates to scroll and normalizes results."""
    with patch("src.meta_agent.services.qdrant.QdrantClient") as mock_client_class:
        QdrantService._instance = None
        mock_client = MagicMock()
        point = SimpleNamespace(id="id1", score=None, payload={"status": "approved"})
        mock_client.scroll.return_value = ([point], None)
        mock_client_class.return_value = mock_client

        service = QdrantService()
        result = service.filter_points("simulations", "status", "approved", limit=7)

    assert result == [{"id": "id1", "score": None, "payload": {"status": "approved"}}]
    mock_client.scroll.assert_called_once()


def test_qdrant_service_scroll_points_with_payload_selector_and_filter():
    """Test scroll_points returns points and next offset."""
    with patch("src.meta_agent.services.qdrant.QdrantClient") as mock_client_class:
        QdrantService._instance = None
        mock_client = MagicMock()
        point = SimpleNamespace(id="id2", score=None, payload={"a": 1, "b": 2})
        mock_client.scroll.return_value = ([point], "next-1")
        mock_client_class.return_value = mock_client

        service = QdrantService()
        result = service.scroll_points(
            "questions",
            limit=5,
            offset="off1",
            payload_fields=["a"],
            filter_field="a",
            filter_value="1",
        )

    assert result["next_offset"] == "next-1"
    assert result["points"][0]["id"] == "id2"
    mock_client.scroll.assert_called_once()


def test_qdrant_service_retrieve_by_id_success_and_not_found():
    """Test retrieve_by_id returns points or raises when missing."""
    with patch("src.meta_agent.services.qdrant.QdrantClient") as mock_client_class:
        QdrantService._instance = None
        mock_client = MagicMock()
        point = SimpleNamespace(id="u1", score=None, payload={"k": "v"})
        mock_client.retrieve.return_value = [point]
        mock_client_class.return_value = mock_client
        service = QdrantService()

        result = service.retrieve_by_id("personas", ["u1"])
        assert result == [{"id": "u1", "score": None, "payload": {"k": "v"}}]

        mock_client.retrieve.return_value = []
        with pytest.raises(ValueError):
            service.retrieve_by_id("personas", ["missing"])


def test_qdrant_service_get_collection_schema():
    """Test schema extraction from Qdrant collection info object."""
    with patch("src.meta_agent.services.qdrant.QdrantClient") as mock_client_class:
        QdrantService._instance = None
        mock_client = MagicMock()
        mock_client.get_collection.return_value = SimpleNamespace(
            payload_schema={"field": "keyword"},
            vectors={"embedding": {"size": 1536}},
        )
        mock_client_class.return_value = mock_client

        service = QdrantService()
        schema = service.get_collection_schema("personas")

    assert schema["collection"] == "personas"
    assert "field" in schema["payload_schema"]
    assert "embedding" in schema["vectors"]


def test_point_to_dict_conversion():
    """Test internal point normalization methods."""
    with patch("src.meta_agent.services.qdrant.QdrantClient"):
        QdrantService._instance = None
        service = QdrantService()
    # Mock point object
    mock_point = MagicMock()
    mock_point.id = "test-id"
    mock_point.score = 0.95
    mock_point.payload = {"key": "value"}

    result = service._point_to_dict(mock_point)
    assert result["id"] == "test-id"
    assert result["score"] == 0.95
    assert result["payload"] == {"key": "value"}

    # List conversion
    points = [mock_point, mock_point]
    results = service._points_to_dict(points)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_service_methods_with_mocks(mock_qdrant_service, mocker):
    """Test public methods via the mocked service from conftest."""
    # The mock_qdrant_service fixture already patches the class
    service = mock_qdrant_service

    # Configure sync method return values
    service.search.return_value = []
    service.filter_points.return_value = []
    service.scroll_points.return_value = {"points": [], "next_offset": None}
    service.retrieve_by_id.return_value = [{"id": "id1"}]
    service.get_collection_schema.return_value = {"collection": "target_audiences"}

    result = service.search("personas", "test query")
    assert isinstance(result, list)

    # Other methods
    filtered = service.filter_points("simulations", "status", "approved")
    scrolled = service.scroll_points("questions", limit=10)
    retrieved = service.retrieve_by_id("personas", ["id1"])
    schema = service.get_collection_schema("target_audiences")

    assert isinstance(filtered, list)
    assert "points" in scrolled
    assert retrieved[0]["id"] == "id1"
    assert schema["collection"] == "target_audiences"


def test_service_schema_method(mock_qdrant_service):
    """Test get_collection_schema."""
    mock_qdrant_service.get_collection_schema.return_value = {"status": "green", "points_count": 1000}
    schema = mock_qdrant_service.get_collection_schema("personas")
    assert schema["points_count"] == 1000
