"""Test to investigate checkpoint serialization/deserialization of DtoPayload objects.

This test reproduces the issue where DtoPayload objects stored in dto_store
are not reconstructed when loading from checkpoint.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.meta_agent.dto import DtoPayload
from src.meta_agent.utils.state import MetaAgentState


def test_pydantic_direct_validation_with_nested_dto():
    """Test that Pydantic correctly validates nested DtoPayload in dto_store."""
    # Create a DtoPayload object
    dto_payload = DtoPayload(
        summary_text="Test data",
        columns=["id", "name"],
        num_rows=2,
        sample=[{"id": 1, "name": "Alice"}],
        rows=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        meta={"source": "test"},
    )

    # Create a MetaAgentState with the DtoPayload
    state = MetaAgentState(
        question="Test question",
        history=[],
        dto_store={"test_dto": dto_payload},
        next_worker="",
        current_task="",
        delegated_attempts=0,
        answer="",
        iterations=0,
    )

    # Verify we have DtoPayload objects
    assert isinstance(state.dto_store["test_dto"], DtoPayload)
    assert state.dto_store["test_dto"].num_rows == 2

    print("\n✓ Direct state creation works with DtoPayload objects")


def test_pydantic_validation_from_dict_with_nested_dto():
    """Test Pydantic validation when dto_store contains dicts instead of DtoPayload objects."""
    # Simulate what comes from checkpoint: plain dicts
    state_dict = {
        "question": "Test question",
        "history": [],
        "dto_store": {
            "test_dto": {
                "summary_text": "Test data",
                "columns": ["id", "name"],
                "num_rows": 2,
                "sample": [{"id": 1, "name": "Alice"}],
                "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "meta": {"source": "test"},
            }
        },
        "next_worker": "",
        "current_task": "",
        "delegated_attempts": 0,
        "answer": "",
        "iterations": 0,
    }

    # Try to validate and convert to MetaAgentState
    state = MetaAgentState.model_validate(state_dict)

    # Check what type we actually got
    dto_store_item = state.dto_store.get("test_dto")
    print(f"\nType of dto_store['test_dto']: {type(dto_store_item)}")
    print(f"Is it DtoPayload? {isinstance(dto_store_item, DtoPayload)}")
    print(f"Is it dict? {isinstance(dto_store_item, dict)}")
    print(f"Value: {dto_store_item}")

    # This is where the test fails - we expect DtoPayload but get dict
    assert isinstance(
        dto_store_item, DtoPayload
    ), f"Expected DtoPayload, got {type(dto_store_item)}"


def test_pydantic_model_dump_and_reload():
    """Test the full cycle: create state -> model_dump -> model_validate."""
    # Create initial state with DtoPayload
    original_dto = DtoPayload(
        summary_text="Test data",
        columns=["id", "name"],
        num_rows=2,
        sample=[{"id": 1, "name": "Alice"}],
        rows=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        meta={"source": "test"},
    )

    state = MetaAgentState(
        question="Test question",
        history=[],
        dto_store={"test_dto": original_dto},
        next_worker="",
        current_task="",
        delegated_attempts=0,
        answer="",
        iterations=0,
    )

    # model_dump (what checkpoint stores)
    dumped = state.model_dump()
    print(f"\nAfter model_dump, dto_store['test_dto'] type: {type(dumped['dto_store']['test_dto'])}")
    assert isinstance(
        dumped["dto_store"]["test_dto"], dict
    ), "model_dump should convert to dicts"

    # Simulate JSON serialization (what checkpoint does)
    json_str = json.dumps(dumped, default=str)
    loaded_dict = json.loads(json_str)
    print(f"After JSON roundtrip, dto_store['test_dto'] type: {type(loaded_dict['dto_store']['test_dto'])}")

    # Try to reconstruct state
    reconstructed = MetaAgentState.model_validate(loaded_dict)
    dto_item = reconstructed.dto_store.get("test_dto")

    print(f"After model_validate, dto_store['test_dto'] type: {type(dto_item)}")
    print(f"Is it DtoPayload? {isinstance(dto_item, DtoPayload)}")
    print(f"Is it dict? {isinstance(dto_item, dict)}")

    # This is where the bug manifests
    assert isinstance(
        dto_item, DtoPayload
    ), f"Expected DtoPayload after model_validate, got {type(dto_item)}"


def test_merge_dto_store_reducer_with_dicts():
    """Test that the merge_dto_store reducer converts dicts to DtoPayload (the fix)."""
    from src.meta_agent.utils.state import merge_dto_store

    # Create initial DtoPayload
    dto1 = DtoPayload(
        summary_text="First",
        columns=["a"],
        num_rows=1,
        sample=[{"a": 1}],
        rows=[{"a": 1}],
    )

    # Simulate checkpoint loading: left has DtoPayload
    left_store = {"dto1": dto1}

    # New update from node with dict (from model_dump)
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

    print(f"\nAfter merge, dto1 type: {type(merged['dto1'])}")
    print(f"After merge, dto2 type: {type(merged['dto2'])}")

    # Check that both are DtoPayload (after fix)
    assert isinstance(merged["dto1"], DtoPayload), "Left DtoPayload should remain DtoPayload"
    assert isinstance(
        merged["dto2"], DtoPayload
    ), "Right dict should be converted to DtoPayload (FIX APPLIED)"
