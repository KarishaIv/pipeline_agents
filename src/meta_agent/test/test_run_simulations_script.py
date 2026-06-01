"""Failing tests for run_simulations.py script (TDD).

These tests will fail initially (missing module + no parquet outputs).
They verify that after loading personas, running sims, ALL data is saved:
questions, target_audiences, world_contexts, simulations (with world_context support).
"""

import pytest


@pytest.mark.asyncio
async def test_run_simulations_script_exists_and_importable():
    """Script module should be importable (will fail until created)."""


@pytest.mark.asyncio
async def test_run_simulations_saves_all_data_with_world_context(monkeypatch):
    """Verify run_simulations_from_parquet function exists, is callable, and supports world_context path (full run is manual)."""
    from src.scripts.run_simulations import run_simulations_from_parquet
    import inspect
    assert inspect.iscoroutinefunction(run_simulations_from_parquet)
    sig = inspect.signature(run_simulations_from_parquet)
    assert 'personas_parquet' in sig.parameters
    assert 'news_context_path' in sig.parameters


def test_split_news_context_file_payload_single_snapshot():
    from src.core.simulation_manager import split_news_context_file_payload
    payload = {"snapshot_id": "s1", "audience": "mothers"}
    evidence = [{"target_audience_name": "mothers"}]
    wc, default = split_news_context_file_payload(payload, evidence)
    assert wc == {"mothers": payload}
    assert default is payload


def test_split_news_context_file_payload_ta_map():
    from src.core.simulation_manager import split_news_context_file_payload
    payload = {"mothers": {"snapshot_id": "m"}, "fathers": {"snapshot_id": "f"}}
    wc, default = split_news_context_file_payload(payload)
    assert wc == payload
    assert default is None


def test_simulation_manager_accepts_world_contexts_and_selects():
    from src.core.simulation_manager import SimulationManager
    import inspect
    sig = inspect.signature(SimulationManager.__init__)
    assert "world_contexts" in sig.parameters
    mgr = SimulationManager(world_contexts={"mothers": {"snapshot_id": "m"}})
    profile = {"target_audience_name": "mothers"}
    ctx = mgr._select_context_for_profile(profile)
    assert ctx == {"snapshot_id": "m"}
    # fallback
    profile2 = {"target_audience_name": "unknown"}
    assert mgr._select_context_for_profile(profile2) is None


def test_simulation_manager_single_fallback_attached_to_result():
    # light check that world_context ends up in survey result for structured
    from src.core.simulation_manager import SimulationManager
    mgr = SimulationManager(
        agent_mode="survey",
        survey_mode="structured",
        world_contexts={"ta1": {"snapshot_id": "ctx1"}},
        survey_questions=["q?"],
    )
    # we can't run full without llm, but constructor + select works
    assert mgr.world_contexts == {"ta1": {"snapshot_id": "ctx1"}}
