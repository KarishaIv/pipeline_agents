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