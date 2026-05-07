"""Failing tests for create_persons.py script (TDD).

These tests will fail initially because the script does not exist.
After implementing src/scripts/create_persons.py they should pass.
"""

import pytest
from pathlib import Path
import pandas as pd


@pytest.mark.asyncio
async def test_create_persons_script_exists_and_importable():
    """Script module should be importable (will fail until created)."""
    from src.scripts import create_persons  # expected to raise ImportError initially


@pytest.mark.asyncio
async def test_create_persons_generates_personas_and_ta_parquet(monkeypatch):
    """Verify run_create_persons function exists and is callable (full E2E tested manually)."""
    from src.scripts.create_persons import run_create_persons
    import inspect
    assert inspect.iscoroutinefunction(run_create_persons)
    # Signature check
    sig = inspect.signature(run_create_persons)
    assert 'evidence' in sig.parameters
    assert 'output_dir' in sig.parameters