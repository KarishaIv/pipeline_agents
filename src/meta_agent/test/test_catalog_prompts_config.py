"""Tests for catalog, prompts, and config modules.

All tests are self-contained in src/meta_agent/test/. Covers constants, prompt construction,
catalog generation, and env-based config.
"""
import importlib
from typing import get_args

import pytest

from src.meta_agent.configs import (
    AVAILABLE_COLLECTIONS,
    COLLECTION_DESCRIPTIONS,
    CollectionName,
    get_collection_catalog,
)
from src.meta_agent.configs import (
    CHARTS_DIR,
)
from src.meta_agent.prompts import (
    ANALYZER_SYSTEM,
    CODE_WRITER_SYSTEM,
    EXTRACTOR_SYSTEM,
    SUPERVISOR_SYSTEM,
    _ANALYZER_TOOLS_BLOCK,
    _CODE_WRITER_TOOLS_BLOCK,
    _COLLECTION_CATALOG,
    _EXTRACTOR_TOOLS_BLOCK,
)


def test_catalog_constants_and_function():
    """Test catalog definitions and get_collection_catalog output."""
    assert len(AVAILABLE_COLLECTIONS) == 5
    assert all(c in ["questions", "personas", "target_audiences", "simulations", "world_contexts"] for c in AVAILABLE_COLLECTIONS)
    assert set(get_args(CollectionName)) == set(AVAILABLE_COLLECTIONS)

    assert "questions" in COLLECTION_DESCRIPTIONS
    assert "персоны" in COLLECTION_DESCRIPTIONS["personas"].lower()
    assert "world_contexts" in COLLECTION_DESCRIPTIONS
    assert "новостной" in COLLECTION_DESCRIPTIONS["world_contexts"].lower() or "контекст" in COLLECTION_DESCRIPTIONS["world_contexts"].lower()

    catalog = get_collection_catalog()
    assert isinstance(catalog, str)
    assert "• questions —" in catalog
    assert "• personas —" in catalog
    assert "• world_contexts —" in catalog
    assert len(catalog.split("\n")) == 5


def test_prompts_contain_key_elements():
    """Test that system prompts contain critical instructions, tool lists, and catalog."""
    prompts = [SUPERVISOR_SYSTEM, EXTRACTOR_SYSTEM, ANALYZER_SYSTEM, CODE_WRITER_SYSTEM]

    for prompt in prompts:
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantial content
        assert "супервайзер" in prompt.lower() or "агент" in prompt.lower()

    # Specific tool blocks embedded
    assert "remaining_steps" in _EXTRACTOR_TOOLS_BLOCK
    assert "collection_schema" in _EXTRACTOR_TOOLS_BLOCK
    assert "compute_stats" in _ANALYZER_TOOLS_BLOCK
    assert "create_chart" in _ANALYZER_TOOLS_BLOCK
    assert "validate_code" in _CODE_WRITER_TOOLS_BLOCK
    assert "execute_code" in _CODE_WRITER_TOOLS_BLOCK

    # Catalog is injected
    assert _COLLECTION_CATALOG in EXTRACTOR_SYSTEM
    assert "DTOS_DATA_JSON" in CODE_WRITER_SYSTEM

    # Decision tools mentioned
    assert "supervisor_decision" in SUPERVISOR_SYSTEM.lower()
    assert "analyzer_decision" in ANALYZER_SYSTEM.lower()
    assert "data_extraction_report" in EXTRACTOR_SYSTEM.lower()


def test_config_env_overrides(monkeypatch):
    """Test that config respects environment variables."""
    import src.meta_agent.configs.runtime as config_module

    monkeypatch.setenv("QDRANT_HOST", "test-host")
    monkeypatch.setenv("QDRANT_PORT", "9999")

    reloaded = importlib.reload(config_module)
    assert reloaded.QDRANT_HOST == "test-host"
    assert reloaded.QDRANT_PORT == 9999


def test_config_charts_dir_resolution():
    """Test CHARTS_DIR uses the data directory under PROJECT_ROOT."""
    from config import PROJECT_ROOT

    assert CHARTS_DIR == PROJECT_ROOT / "data" / "charts"
    assert CHARTS_DIR.parent.name == "data"


@pytest.mark.parametrize("collection", AVAILABLE_COLLECTIONS)
def test_collection_descriptions_complete(collection):
    """Ensure every collection has a description."""
    assert collection in COLLECTION_DESCRIPTIONS
    assert len(COLLECTION_DESCRIPTIONS[collection]) > 10
