"""Test to demonstrate the ACTUAL bug: merge_dto_store with mixed types.

The real issue is that LangGraph's reducer merges the state updates.
When a node returns dto_store with dicts (from model_dump), they get merged
with left values, but the reducer doesn't guarantee they'll be DtoPayload objects.
"""

from src.meta_agent.dto import DtoPayload
from src.meta_agent.utils.state import merge_dto_store, MetaAgentState


def test_merge_dto_store_loses_type_information():
    """The merge_dto_store reducer now PRESERVES type information (after fix).

    After the fix, merge_dto_store converts dicts to DtoPayload objects.
    """
    # Initial state: has DtoPayload
    left_store = {
        "existing_dto": DtoPayload(
            summary_text="Existing",
            columns=["a"],
            num_rows=1,
            sample=[{"a": 1}],
            rows=[{"a": 1}],
        )
    }

    # Node returns state update with dto_store as dicts (from model_dump)
    right_store = {
        "existing_dto": {
            "summary_text": "Updated",
            "columns": ["a"],
            "num_rows": 1,
            "sample": [{"a": 1}],
            "rows": [{"a": 1}],
        },
        "new_dto": {
            "summary_text": "New",
            "columns": ["b"],
            "num_rows": 2,
            "sample": [{"b": 2}],
            "rows": [{"b": 2}, {"b": 3}],
        },
    }

    # Merge is called
    merged = merge_dto_store(left_store, right_store)

    print(f"\nAfter merge_dto_store (with fix):")
    for name, value in merged.items():
        print(f"  {name}: type={type(value).__name__}, is_dict={isinstance(value, dict)}, is_DtoPayload={isinstance(value, DtoPayload)}")

    # After fix: all values should be DtoPayload
    assert isinstance(merged["existing_dto"], DtoPayload), "Overwritten value should be DtoPayload"
    assert isinstance(merged["new_dto"], DtoPayload), "New value should be DtoPayload"


def test_issue_happens_during_graph_invoke():
    """Simulate what happens during actual graph execution (after fix).

    After the fix, the merge_dto_store reducer properly converts dicts to DtoPayload,
    so dicts don't leak into the store.
    """
    print("\n=== Simulating graph execution (after fix) ===")

    # Step 1: Initial state from checkpoint is dicts
    checkpoint_state = {
        "question": "Test",
        "history": [],
        "dto_store": {
            "personas_1": {
                "summary_text": "Personas",
                "columns": ["id", "name"],
                "num_rows": 2,
                "sample": [{"id": 1, "name": "Alice"}],
                "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "meta": {},
            }
        },
        "next_worker": "",
        "current_task": "",
        "delegated_attempts": 0,
        "answer": "",
        "iterations": 0,
    }

    print("\n1. Checkpoint returns dicts:")
    print(f"   dtype_store['personas_1']: {type(checkpoint_state['dto_store']['personas_1'])}")

    # Step 2: Validate with MetaAgentState
    state = MetaAgentState.model_validate(checkpoint_state)
    print(f"\n2. After model_validate:")
    print(f"   state.dto_store['personas_1']: {type(state.dto_store['personas_1'])}")

    # Step 3: model_dump for nodes
    state_dict = state.model_dump()
    print(f"\n3. After model_dump (send to nodes):")
    print(f"   state_dict['dto_store']['personas_1']: {type(state_dict['dto_store']['personas_1'])}")

    # Step 4: Node processes and returns update
    node_update = {
        "dto_store": {
            "personas_1": {  # Modified from node
                "summary_text": "Personas - Updated",
                "columns": ["id", "name"],
                "num_rows": 2,
                "sample": [{"id": 1, "name": "Alice"}],
                "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
                "meta": {"updated": True},
            },
            "new_data_1": {  # New from node
                "summary_text": "New data",
                "columns": ["x"],
                "num_rows": 1,
                "sample": [{"x": 1}],
                "rows": [{"x": 1}],
                "meta": {},
            }
        }
    }
    print(f"\n4. Node returns update with dicts:")
    for name, value in node_update["dto_store"].items():
        print(f"   {name}: {type(value)}")

    # Step 5: LangGraph calls the reducer (NOW WITH FIX)
    merged = merge_dto_store(state.dto_store, node_update["dto_store"])
    print(f"\n5. After merge_dto_store (WITH FIX):")
    for name, value in merged.items():
        print(f"   {name}: {type(value)}")

    # Step 6: Try to get one of the merged DTOs
    personas = merged["personas_1"]
    print(f"\n6. Trying to call .get_summary() on personas_1:")
    print(f"   Type: {type(personas)}")
    if isinstance(personas, dict):
        print(f"   ERROR: It's still a dict! This shouldn't happen now.")
        try:
            personas.get_summary("personas_1", 50)
        except AttributeError as e:
            print(f"   ❌ AttributeError: {e}")
    else:
        print(f"   ✓ OK: It's a DtoPayload")
        summary = personas.get_summary("personas_1", 50)
        print(f"   ✓ Successfully called .get_summary(): {summary.dto_name}")


def test_the_real_fix_needed():
    """The fix should be in merge_dto_store to ensure all values are DtoPayload.

    OR in get_dto_store to ensure proper type conversion when loading from checkpoint.
    """
    # The merged dict contains plain dicts
    mixed_store = {
        "dto1": {
            "summary_text": "Data 1",
            "columns": ["a"],
            "num_rows": 1,
            "sample": [{"a": 1}],
            "rows": [{"a": 1}],
            "meta": {},
        }
    }

    print("\n=== Proposed Fix ===")
    print("Option 1: Make merge_dto_store validate the right dict:")

    # Proposed fix
    def merge_dto_store_fixed(
        left: dict, right: dict | None
    ) -> dict:
        """Fixed version that ensures all values are DtoPayload."""
        merged = dict(left) if left else {}
        if isinstance(right, dict):
            for key, value in right.items():
                # Convert dict to DtoPayload if needed
                if isinstance(value, dict) and not isinstance(value, DtoPayload):
                    try:
                        merged[key] = DtoPayload(**value)
                    except Exception as e:
                        print(f"    Warning: Could not convert {key}: {e}")
                        merged[key] = value  # Keep as dict if conversion fails
                else:
                    merged[key] = value
        return merged

    fixed_store = merge_dto_store_fixed({}, mixed_store)
    for name, value in fixed_store.items():
        print(f"   After fix - {name}: {type(value).__name__}")
        assert isinstance(value, DtoPayload), f"Should be DtoPayload, got {type(value)}"
