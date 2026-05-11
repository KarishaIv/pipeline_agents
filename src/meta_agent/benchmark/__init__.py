"""Meta-agent qualitative benchmark (no gold oracles; manual/LLM 0-1 scoring)."""

from .cases import BenchmarkCase, CaseScore, BenchmarkResult
from .runner import BenchmarkRunner
from .report import generate_report
from .suites import (
    get_command_following_suite,
    get_data_extraction_suite,
    get_analysis_correctness_suite,
    get_graph_artifact_quality_suite,
    get_session_context_behavior_suite,
)

__all__ = [
    "BenchmarkCase",
    "CaseScore",
    "BenchmarkResult",
    "BenchmarkRunner",
    "generate_report",
    "get_command_following_suite",
    "get_data_extraction_suite",
    "get_analysis_correctness_suite",
    "get_graph_artifact_quality_suite",
    "get_session_context_behavior_suite",
]
