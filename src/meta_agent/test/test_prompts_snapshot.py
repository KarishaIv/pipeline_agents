"""Snapshot tests for system prompts.

Uses syrupy to ensure prompt strings don't change unexpectedly.
"""

import pytest

from src.meta_agent.prompts import (
    SUPERVISOR_SYSTEM,
    EXTRACTOR_SYSTEM,
    ANALYZER_SYSTEM,
    CODE_WRITER_SYSTEM,
    HISTORY_SUMMARIZER_SYSTEM,
    _EXTRACTOR_TOOLS_BLOCK,
    _ANALYZER_TOOLS_BLOCK,
    _CODE_WRITER_TOOLS_BLOCK,
    _COLLECTION_CATALOG,
    _CODE_WRITER_DTO_ENV_VAR,
)


class TestPromptSnapshots:
    """Snapshot tests ensure prompts remain stable across refactoring."""

    def test_supervisor_system_prompt_snapshot(self, snapshot):
        assert snapshot == SUPERVISOR_SYSTEM

    def test_extractor_system_prompt_snapshot(self, snapshot):
        assert snapshot == EXTRACTOR_SYSTEM

    def test_analyzer_system_prompt_snapshot(self, snapshot):
        assert snapshot == ANALYZER_SYSTEM

    def test_code_writer_system_prompt_snapshot(self, snapshot):
        assert snapshot == CODE_WRITER_SYSTEM

    def test_history_summarizer_system_prompt_snapshot(self, snapshot):
        assert snapshot == HISTORY_SUMMARIZER_SYSTEM


class TestPromptComponents:
    """Test individual prompt components and their formatting."""

    def test_extractor_tools_block_formatting(self):
        """Verify extractor tools are properly formatted."""
        assert "remaining_steps" in _EXTRACTOR_TOOLS_BLOCK
        assert "collection_schema" in _EXTRACTOR_TOOLS_BLOCK
        assert "data_extraction_report" in _EXTRACTOR_TOOLS_BLOCK
        lines = _EXTRACTOR_TOOLS_BLOCK.split("\n")
        assert all(line.startswith("- ") for line in lines)

    def test_analyzer_tools_block_formatting(self):
        """Verify analyzer tools are properly formatted."""
        assert "compute_stats" in _ANALYZER_TOOLS_BLOCK
        assert "create_chart" in _ANALYZER_TOOLS_BLOCK
        assert "summarize_texts" in _ANALYZER_TOOLS_BLOCK
        assert "analyzer_decision" in _ANALYZER_TOOLS_BLOCK
        lines = _ANALYZER_TOOLS_BLOCK.split("\n")
        assert all(line.startswith("- ") for line in lines)

    def test_code_writer_tools_block_formatting(self):
        """Verify code writer tools are properly formatted."""
        assert "validate_code" in _CODE_WRITER_TOOLS_BLOCK
        assert "execute_code" in _CODE_WRITER_TOOLS_BLOCK
        assert "code_execution_report" in _CODE_WRITER_TOOLS_BLOCK
        lines = _CODE_WRITER_TOOLS_BLOCK.split("\n")
        assert all(line.startswith("- ") for line in lines)

    def test_collection_catalog_injected_in_extractor(self):
        """Verify collection catalog is injected into extractor prompt."""
        assert _COLLECTION_CATALOG in EXTRACTOR_SYSTEM
        assert "Доступные коллекции Qdrant:" in EXTRACTOR_SYSTEM

    def test_code_writer_env_var_injected(self):
        """Verify DTO environment variable is injected into code writer prompt."""
        assert _CODE_WRITER_DTO_ENV_VAR == "DTO_DATA_JSON"
        assert _CODE_WRITER_DTO_ENV_VAR in CODE_WRITER_SYSTEM

    def test_all_prompts_have_minimum_length(self):
        """Ensure all system prompts have substantial content."""
        min_length = 100
        prompts = {
            "SUPERVISOR_SYSTEM": SUPERVISOR_SYSTEM,
            "EXTRACTOR_SYSTEM": EXTRACTOR_SYSTEM,
            "ANALYZER_SYSTEM": ANALYZER_SYSTEM,
            "CODE_WRITER_SYSTEM": CODE_WRITER_SYSTEM,
            "HISTORY_SUMMARIZER_SYSTEM": HISTORY_SUMMARIZER_SYSTEM,
        }
        for name, prompt in prompts.items():
            assert len(prompt) > min_length, f"{name} is too short ({len(prompt)} chars)"

    def test_all_prompts_reference_tools(self):
        """Verify system prompts reference their tools."""
        assert "remaining_steps" in SUPERVISOR_SYSTEM or "supervisor_decision" in SUPERVISOR_SYSTEM
        assert "remaining_steps" in EXTRACTOR_SYSTEM
        assert "remaining_steps" in ANALYZER_SYSTEM
        assert "remaining_steps" in CODE_WRITER_SYSTEM

    def test_russian_language_in_prompts(self):
        """Verify prompts are in Russian as expected."""
        russian_keywords = {
            "SUPERVISOR_SYSTEM": "супервайзер",
            "EXTRACTOR_SYSTEM": "извлекатель",
            "ANALYZER_SYSTEM": "аналитик",
            "CODE_WRITER_SYSTEM": "написания и выполнения",
            "HISTORY_SUMMARIZER_SYSTEM": "сжатию истории",
        }
        prompts = {
            "SUPERVISOR_SYSTEM": SUPERVISOR_SYSTEM,
            "EXTRACTOR_SYSTEM": EXTRACTOR_SYSTEM,
            "ANALYZER_SYSTEM": ANALYZER_SYSTEM,
            "CODE_WRITER_SYSTEM": CODE_WRITER_SYSTEM,
            "HISTORY_SUMMARIZER_SYSTEM": HISTORY_SUMMARIZER_SYSTEM,
        }
        for name, keyword in russian_keywords.items():
            assert keyword.lower() in prompts[name].lower(), (
                f"Expected Russian keyword '{keyword}' in {name}"
            )
