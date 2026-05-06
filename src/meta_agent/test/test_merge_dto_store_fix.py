"""Regression test for merge_dto_store bug.

Tests that merge_dto_store properly converts dict values to DtoPayload objects
when merging state updates from nodes.
"""

import pytest

from src.meta_agent.dto import DtoPayload
from src.meta_agent.utils.state import merge_dto_store


def test_merge_dto_store_converts_dict_to_dtopayload():
    """Test that merge_dto_store converts dict values to DtoPayload.

    This is a regression test for the bug where merge_dto_store would preserve
    dict objects instead of converting them to DtoPayload, causing
    AttributeError: 'dict' object has no attribute 'get_summary'
    """
    # Initial state with DtoPayload (left side from previous state)
    left_store = {
        "existing_dto": DtoPayload(
            summary_text="Existing data",
            columns=["id", "name"],
            num_rows=2,
            sample=[{"id": 1, "name": "Alice"}],
            rows=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            meta={"source": "checkpoint"},
        )
    }

    # Node returns update with dicts (from model_dump)
    right_store = {
        "existing_dto": {
            "summary_text": "Updated data",
            "columns": ["id", "name"],
            "num_rows": 2,
            "sample": [{"id": 1, "name": "Alice"}],
            "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "meta": {"updated": True},
        },
        "new_dto": {
            "summary_text": "New data from node",
            "columns": ["x", "y"],
            "num_rows": 3,
            "sample": [{"x": 1, "y": 10}],
            "rows": [{"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 30}],
            "meta": {"source": "node"},
        },
    }

    # Merge the stores
    merged = merge_dto_store(left_store, right_store)

    # All values should be DtoPayload, not dict
    assert isinstance(merged["existing_dto"], DtoPayload), \
        f"Overwritten value should be DtoPayload, got {type(merged['existing_dto'])}"
    assert isinstance(merged["new_dto"], DtoPayload), \
        f"New value should be DtoPayload, got {type(merged['new_dto'])}"

    # Verify we can call DtoPayload methods
    summary1 = merged["existing_dto"].get_summary("existing_dto", 50)
    assert summary1.dto_name == "existing_dto"

    summary2 = merged["new_dto"].get_summary("new_dto", 50)
    assert summary2.dto_name == "new_dto"


def test_merge_dto_store_preserves_existing_dtopayload():
    """Test that merge_dto_store preserves existing DtoPayload objects when not overwritten."""
    dto1 = DtoPayload(
        summary_text="First",
        columns=["a"],
        num_rows=1,
        sample=[{"a": 1}],
        rows=[{"a": 1}],
    )

    left_store = {"dto1": dto1}

    # Right store with different dto
    right_store = {
        "dto2": {
            "summary_text": "Second",
            "columns": ["b"],
            "num_rows": 1,
            "sample": [{"b": 2}],
            "rows": [{"b": 2}],
        }
    }

    merged = merge_dto_store(left_store, right_store)

    # Existing DtoPayload should be preserved
    assert merged["dto1"] is dto1, "Existing DtoPayload should be preserved"
    # New dict should be converted
    assert isinstance(merged["dto2"], DtoPayload), "New dict should be converted to DtoPayload"


def test_merge_dto_store_handles_invalid_dict():
    """Test that merge_dto_store raises ValueError for dict that can't be converted to DtoPayload."""
    left_store = {}

    # Dict with missing required fields
    invalid_dict = {
        "incomplete_dto": {
            "summary_text": "Incomplete",
            # Missing required fields: columns, num_rows, sample, rows
        }
    }

    # Should raise ValueError for malformed DTO
    with pytest.raises(ValueError, match="Cannot convert dto_store.*to DtoPayload"):
        merge_dto_store(left_store, invalid_dict)


def test_merge_dto_store_with_none_right():
    """Test that merge_dto_store handles None as right side."""
    left_store = {
        "dto1": DtoPayload(
            summary_text="Data",
            columns=["a"],
            num_rows=1,
            sample=[{"a": 1}],
            rows=[{"a": 1}],
        )
    }

    merged = merge_dto_store(left_store, None)

    assert merged == left_store
    assert isinstance(merged["dto1"], DtoPayload)


def test_merge_dto_store_empty_stores():
    """Test merge_dto_store with empty stores."""
    merged1 = merge_dto_store({}, {})
    assert merged1 == {}

    merged2 = merge_dto_store({}, None)
    assert merged2 == {}

    dto = DtoPayload(
        summary_text="Data",
        columns=["a"],
        num_rows=1,
        sample=[{"a": 1}],
        rows=[{"a": 1}],
    )
    merged3 = merge_dto_store({"dto": dto}, {})
    assert isinstance(merged3["dto"], DtoPayload)
