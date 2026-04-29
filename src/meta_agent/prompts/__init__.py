"""Public API for system prompts across all meta-agent workers.

This module re-exports prompts and related constants from sub-modules,
providing a unified interface for all prompt access.
"""

from src.meta_agent.prompts.supervisor import SUPERVISOR_SYSTEM
from src.meta_agent.prompts.extractor import (
    EXTRACTOR_SYSTEM,
    _EXTRACTOR_TOOLS,
    _EXTRACTOR_TOOLS_BLOCK,
    _COLLECTION_CATALOG,
)
from src.meta_agent.prompts.analyzer import (
    ANALYZER_SYSTEM,
    _ANALYZER_TOOLS,
    _ANALYZER_TOOLS_BLOCK,
)
from src.meta_agent.prompts.code_writer import (
    CODE_WRITER_SYSTEM,
    _CODE_WRITER_TOOLS,
    _CODE_WRITER_TOOLS_BLOCK,
    _CODE_WRITER_DTO_ENV_VAR,
)
from src.meta_agent.prompts.history import HISTORY_SUMMARIZER_SYSTEM

__all__ = [
    "SUPERVISOR_SYSTEM",
    "EXTRACTOR_SYSTEM",
    "ANALYZER_SYSTEM",
    "CODE_WRITER_SYSTEM",
    "HISTORY_SUMMARIZER_SYSTEM",
    # Private constants (for tests and configuration)
    "_EXTRACTOR_TOOLS",
    "_EXTRACTOR_TOOLS_BLOCK",
    "_COLLECTION_CATALOG",
    "_ANALYZER_TOOLS",
    "_ANALYZER_TOOLS_BLOCK",
    "_CODE_WRITER_TOOLS",
    "_CODE_WRITER_TOOLS_BLOCK",
    "_CODE_WRITER_DTO_ENV_VAR",
]
