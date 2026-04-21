"""Shared test fixtures and configuration for meta_agent tests.

All tests are contained within src/meta_agent/test/ per the requirements.
"""
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from src.meta_agent.catalog import AVAILABLE_COLLECTIONS, COLLECTION_DESCRIPTIONS
from src.meta_agent.utils.state import MetaAgentState


@pytest.fixture
def meta_state():
    """Fixture providing a basic MetaAgentState instance."""
    return MetaAgentState(
        question="Test question about personas",
        history=[],
        dto_store={},
        next_worker="",
        current_task="",
        delegated_attempts=0,
        answer="",
        iterations=0,
    )


@pytest.fixture
def mock_qdrant_service(mocker):
    """Mock for QdrantService and its client."""
    mock_service = MagicMock()
    mock_client = MagicMock()
    mock_service.client = mock_client
    mocker.patch("src.meta_agent.services.qdrant.QdrantService", return_value=mock_service)
    mocker.patch("src.meta_agent.tools.qdrant_tools.get_qdrant_service", return_value=mock_service)
    return mock_service


@pytest.fixture
def mock_run_agent(mocker):
    """Mock for run_agent from agent_factory."""
    async_mock = AsyncMock()
    mocker.patch("src.meta_agent.nodes.run_agent", new=async_mock)
    mocker.patch("src.meta_agent.agent_factory.run_agent", new=async_mock)
    return async_mock


@pytest.fixture
def mock_openai_client(mocker):
    """Mock for AsyncOpenAI client used in tools and agent_factory."""
    mock_client = AsyncMock()
    mocker.patch("src.meta_agent.agent_factory.AsyncOpenAI", return_value=mock_client)
    mocker.patch("src.meta_agent.tools.analyzer_tools.AsyncOpenAI", return_value=mock_client)
    return mock_client


@pytest.fixture
def temp_charts_dir(tmp_path):
    """Temporary directory for charts created by analyzer tools."""
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir(exist_ok=True)
    with patch("src.meta_agent.config.CHARTS_DIR", charts_dir):
        yield charts_dir


@pytest.fixture
def mock_catalog(mocker):
    """Mock catalog functions."""
    mocker.patch("src.meta_agent.catalog.get_collection_catalog", return_value="Mocked catalog:\n  • questions — test")
    mocker.patch("src.meta_agent.prompts.get_collection_catalog", return_value="Mocked catalog:\n  • questions — test")
    return AVAILABLE_COLLECTIONS


@pytest.fixture
def mock_sqlite_saver(tmp_path):
    """SQLite checkpointer fixture for graph tests."""
    conn = sqlite3.connect(tmp_path / "test-checkpoints.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    try:
        yield saver
    finally:
        conn.close()


# Common mock responses
@pytest.fixture
def sample_dto_data():
    """Sample DTO data for testing DTO tools."""
    return [
        {"id": "1", "age": 25, "income": 50000, "text": "Sample text"},
        {"id": "2", "age": 35, "income": 75000, "text": "Another sample"},
    ]
