"""Tests for dto_tools.py - DTO management, registration, sampling, and pandas conversion.

Covers _normalize_rows, register_dto, resolve_dto_or_error, ListDtoNamesTool, SampleDtoTool,
and all helper functions with edge cases.
"""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.meta_agent.dto import DtoPayload
from src.meta_agent.tools.dto_tools import (
    DTO_STORE_KEY,
    ListDtoNamesTool,
    SampleDtoTool,
    _infer_columns,
    _next_dto_name,
    _normalize_rows,
    _sanitize_source,
    dto_summary_view,
    dto_to_dataframe,
    get_dto,
    get_dto_store,
    register_dto,
    resolve_dto_or_error,
    set_dto_store,
)


def test_normalize_rows_various_inputs():
    """Test _normalize_rows handles dicts with points, lists, scalars, None."""
    assert _normalize_rows(None) == []
    assert _normalize_rows([]) == []
    assert _normalize_rows({"points": [{"a": 1}, {"b": 2}]}) == [{"a": 1}, {"b": 2}]
    assert _normalize_rows([{"id": 1}, {"id": 2}]) == [{"id": 1}, {"id": 2}]
    assert _normalize_rows("scalar") == [{"value": "scalar"}]
    assert _normalize_rows(42) == [{"value": 42}]


def test_sanitize_source_and_next_dto_name():
    """Test name sanitization and unique naming logic."""
    assert _sanitize_source("Personas Data!") == "personas_data"
    assert _sanitize_source("   Invalid @# Name   ") == "invalid_name"
    assert _sanitize_source("") == "dto"

    mock_context = MagicMock()
    mock_context.custom_context = {DTO_STORE_KEY: {"personas_1": {}, "personas_2": {}}}

    name = _next_dto_name(mock_context, "personas")
    assert name == "personas_3"


def test_dto_store_helpers():
    """Test get_dto_store, set_dto_store, _ensure_context_dict behaviors."""
    mock_context = MagicMock()
    mock_context.custom_context = None

    store = get_dto_store(mock_context)
    assert isinstance(store, dict)
    assert DTO_STORE_KEY in mock_context.custom_context

    test_store = {"test_dto": {"data": [1, 2]}}
    set_dto_store(mock_context, test_store)
    assert mock_context.custom_context[DTO_STORE_KEY] == test_store

    # Strict validation for custom_context type (no legacy coercion)
    mock_invalid = MagicMock()
    mock_invalid.custom_context = "legacy_string"
    with pytest.raises(
        TypeError, match="AgentContext.custom_context must be a dict or None"
    ):
        get_dto_store(mock_invalid)


def test_register_dto_and_summary(sample_dto_data):
    """Test full DTO registration pipeline and summary view."""
    mock_context = MagicMock()
    mock_context.custom_context = {}

    name, payload = register_dto(
        mock_context,
        source="test_source",
        data=sample_dto_data,
        summary_text="Test registration",
        meta={"source": "test"},
    )

    assert name.startswith("test_source_")
    assert isinstance(payload, DtoPayload)
    assert payload.num_rows == 2
    assert len(payload.columns) >= 1
    assert payload.meta["source"] == "test"

    summary = dto_summary_view(name, payload, max_len=50)
    assert summary.dto_name == name
    assert summary.columns
    assert isinstance(summary.sample, (list, str))


def test_get_dto_and_resolve(sample_dto_data):
    """Test get_dto, resolve_dto_or_error success and error paths."""
    mock_context = MagicMock()
    mock_context.custom_context = {DTO_STORE_KEY: {}}

    name, _ = register_dto(mock_context, source="test", data=sample_dto_data)

    dto = get_dto(mock_context, name)
    assert isinstance(dto, DtoPayload)
    assert dto.num_rows == 2

    # Resolve success
    df, payload, error = resolve_dto_or_error(mock_context, name)
    assert isinstance(df, pd.DataFrame)
    assert payload is not None
    assert isinstance(payload, DtoPayload)
    assert error is None
    assert len(df) == 2

    # Resolve error - missing DTO
    df2, payload2, error2 = resolve_dto_or_error(mock_context, "missing_dto")
    assert df2 is None
    assert payload2 is None
    assert error2 is not None
    assert "не найден" in error2
    assert "list_dtos" in error2


def test_dto_to_dataframe():
    """Test DataFrame conversion from various DTO payloads."""
    # From rows
    payload1 = DtoPayload(
        summary_text="test",
        columns=["a", "b"],
        rows=[{"a": 1, "b": 2}, {"a": 3, "b": 4}],
    )
    df1 = dto_to_dataframe(payload1)
    assert isinstance(df1, pd.DataFrame)
    assert len(df1) == 2
    assert list(df1.columns) == ["a", "b"]

    # From columns only
    payload2 = DtoPayload(
        summary_text="test",
        columns=["x", "y"],
        rows=[],
    )
    df2 = dto_to_dataframe(payload2)
    assert len(df2.columns) == 2
    assert len(df2) == 0

    # Empty
    payload3 = DtoPayload(
        summary_text="empty",
        columns=[],
        rows=[],
    )
    assert len(dto_to_dataframe(payload3)) == 0


def test_dto_summary_derives_sample_from_rows():
    """DtoPayload stores rows only; summaries derive a small preview from rows."""
    payload = DtoPayload(
        summary_text="test",
        columns=["id"],
        rows=[{"id": i} for i in range(6)],
    )

    assert "sample" not in payload.model_dump()
    assert "num_rows" not in payload.model_dump()
    assert payload.num_rows == 6
    assert payload.get_summary("test_dto").sample == [{"id": i} for i in range(5)]
    assert payload.get_summary("test_dto", sample_size=2).sample == [{"id": 0}, {"id": 1}]


@pytest.mark.asyncio
async def test_list_dtos_tool(sample_dto_data):
    """Test ListDtoNamesTool returns summary of all DTOs."""
    tool = ListDtoNamesTool(reasoning="Need to see available data")
    mock_context = MagicMock()
    mock_context.custom_context = {DTO_STORE_KEY: {}}
    register_dto(mock_context, source="test", data=sample_dto_data)
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)
    payload = json.loads(result)

    assert payload["dto_count"] == 1
    assert len(payload["dtos"]) == 1
    assert payload["dtos"][0]["dto_name"].startswith("test_")


@pytest.mark.asyncio
async def test_list_dtos_tool_accepts_checkpoint_restored_dicts():
    """DTOs restored from checkpoints can arrive in tool context as plain dicts."""
    tool = ListDtoNamesTool(reasoning="Need to see restored DTOs")
    mock_context = MagicMock()
    mock_context.custom_context = {
        DTO_STORE_KEY: {
            "checkpoint_dto": {
                "summary_text": "Restored data",
                "columns": ["id", "name"],
                "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "meta": {"source": "checkpoint"},
            }
        }
    }
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)
    payload = json.loads(result)

    assert payload["dto_count"] == 1
    assert payload["dtos"][0]["dto_name"] == "checkpoint_dto"
    assert isinstance(mock_context.custom_context[DTO_STORE_KEY]["checkpoint_dto"], DtoPayload)


@pytest.mark.asyncio
async def test_sample_dto_tool():
    """Test SampleDtoTool for pagination and error on missing DTO."""
    tool = SampleDtoTool(
        reasoning="Get more data",
        dto_name="test_dto",
        sample_size=2,
        start=0,
    )
    mock_context = MagicMock()
    test_payload = DtoPayload(
        summary_text="test",
        columns=["id"],
        rows=[{"id": i} for i in range(10)],
    )
    mock_context.custom_context = {
        DTO_STORE_KEY: {
            "test_dto": test_payload,
        }
    }
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)
    payload = json.loads(result)
    assert payload["dto_name"] == "test_dto"
    assert payload["sample_size"] == 2
    assert len(payload["sample"]) == 2
    assert payload["start"] == 0

    # Test error path
    tool_error = SampleDtoTool(
        reasoning="Bad dto", dto_name="missing", sample_size=5
    )
    error_result = await tool_error(mock_context, mock_config)
    error_payload = json.loads(error_result)
    assert "error" in error_payload
    assert "missing" in error_payload["error"]
    assert "available_dto_names" in error_payload
    assert "test_dto" in error_payload["available_dto_names"]


def test_infer_columns():
    """Test column inference from list of row dicts."""
    rows = [{"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 5}]
    columns = _infer_columns(rows)
    assert columns == ["a", "b", "c"]
    assert len(columns) == len(set(columns))  # unique and ordered by first appearance
