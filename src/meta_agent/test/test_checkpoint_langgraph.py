"""Test to investigate how LangGraph's checkpointer handles DtoPayload serialization.

This test uses the actual SqliteSaver to see if the issue happens during 
checkpoint storage/retrieval.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from src.meta_agent.dto import DtoPayload
from src.meta_agent.utils.state import MetaAgentState


def test_model_dump_converts_dto_to_dict():
    """Test that model_dump() converts DtoPayload objects to dicts.
    
    This is KEY: model_dump() is lossy for DtoPayload objects.
    """
    # Create a MetaAgentState with DtoPayload
    dto_payload = DtoPayload(
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
        dto_store={"test_dto": dto_payload},
        next_worker="",
        current_task="",
        delegated_attempts=0,
        answer="",
        iterations=0,
    )

    print(f"\nBefore model_dump():")
    print(f"  state.dto_store['test_dto'] type: {type(state.dto_store['test_dto'])}")
    print(f"  Is DtoPayload? {isinstance(state.dto_store['test_dto'], DtoPayload)}")

    # Convert to dict (what would be saved to checkpoint)
    state_dict = state.model_dump()
    print(f"\nAfter model_dump():")
    print(f"  state_dict['dto_store']['test_dto'] type: {type(state_dict['dto_store']['test_dto'])}")
    print(f"  Is DtoPayload? {isinstance(state_dict['dto_store']['test_dto'], DtoPayload)}")
    print(f"  Is dict? {isinstance(state_dict['dto_store']['test_dto'], dict)}")

    # The issue: model_dump converts to dict!
    assert isinstance(state_dict["dto_store"]["test_dto"], dict), "model_dump should convert to dict"
    
    # But can we get it back?
    reconstructed = MetaAgentState.model_validate(state_dict)
    dto_item = reconstructed.dto_store.get("test_dto")
    print(f"\nAfter model_validate on dumped state:")
    print(f"  reconstructed.dto_store['test_dto'] type: {type(dto_item)}")
    print(f"  Is DtoPayload? {isinstance(dto_item, DtoPayload)}")
    
    # This should work since Pydantic validates
    assert isinstance(dto_item, DtoPayload), "model_validate should reconstruct DtoPayload"


def test_checkpoint_save_with_json_encoding(tmp_path):
    """Test what gets stored in the checkpoint database (as JSON)."""
    db_path = tmp_path / "test-checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    try:
        # Create state with DtoPayload
        dto_payload = DtoPayload(
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
            dto_store={"test_dto": dto_payload},
            next_worker="",
            current_task="",
            delegated_attempts=0,
            answer="",
            iterations=0,
        )

        # Save to checkpoint using correct API
        config = {"configurable": {"thread_id": "test_thread_2", "checkpoint_ns": ""}}
        state_dict = state.model_dump()
        checkpoint = {
            "v": 1,
            "ts": datetime.now().isoformat(),
            "id": str(uuid4()),
            "channel_values": state_dict,
            "channel_versions": {},
            "versions_seen": {},
        }
        saver.put(config, checkpoint, {}, {})

        # Retrieve checkpoint to verify it was saved
        read_config = {"configurable": {"thread_id": "test_thread_2"}}
        retrieved = saver.get(read_config)
        assert retrieved is not None
        # retrieved is a dict with channel_values
        assert "channel_values" in retrieved
        assert retrieved["channel_values"].get("dto_store") is not None
        print(f"\nCheckpoint saved and retrieved successfully")
        print(f"  dto_store keys: {list(retrieved['channel_values'].get('dto_store', {}).keys())}")
    finally:
        conn.close()


def test_checkpoint_list_and_retrieve(tmp_path):
    """Test checkpoint list and tuple structure."""
    db_path = tmp_path / "test-checkpoints.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    try:
        # Create state with DtoPayload
        dto_payload = DtoPayload(
            summary_text="Test data",
            columns=["id"],
            num_rows=1,
            sample=[{"id": 1}],
            rows=[{"id": 1}],
        )

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

        # Save to checkpoint using correct API
        config = {"configurable": {"thread_id": "test_thread_3", "checkpoint_ns": ""}}
        state_dict = state.model_dump()
        checkpoint = {
            "v": 1,
            "ts": datetime.now().isoformat(),
            "id": str(uuid4()),
            "channel_values": state_dict,
            "channel_versions": {},
            "versions_seen": {},
        }
        saver.put(config, checkpoint, {}, {})

        # Use list() API
        checkpoints = list(saver.list(config))
        print(f"\nCheckpoint list API:")
        print(f"  Number of checkpoints: {len(checkpoints)}")
        if checkpoints:
            cp = checkpoints[0]
            print(f"  Checkpoint type: {type(cp)}")
            # cp is a CheckpointTuple, access via dict interface
            if isinstance(cp, dict):
                channel_values = cp.get("channel_values", {})
            else:
                channel_values = cp.channel_values if hasattr(cp, "channel_values") else {}
            print(f"  channel_values keys: {list(channel_values.keys())}")
            if channel_values.get("dto_store"):
                print(f"  dto_store keys: {list(channel_values['dto_store'].keys())}")
    finally:
        conn.close()
