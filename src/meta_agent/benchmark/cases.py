"""Benchmark case definitions for qualitative evaluation."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class BenchmarkCase:
    """A single benchmark case with qualitative expectations for manual or LLM scoring."""

    id: str
    prompt: str
    section: str  # e.g. "command_following", "data_extraction", etc.
    description: str  # High-level intent of the case
    expected_answer: str  # Narrative description of a good/correct response
    success_criteria: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    rubric: Optional[str] = None  # Optional detailed scoring guidance (0.0-1.0 scale)
    tags: list[str] = field(default_factory=list)
    expected_output_types: list[Literal["text", "json", "image", "file"]] = field(default_factory=list)
    expected_collections: list[str] = field(default_factory=list)
    thread_policy: Literal["new", "fixed", "followup"] = "new"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "section": self.section,
            "description": self.description,
            "expected_answer": self.expected_answer,
            "success_criteria": self.success_criteria,
            "failure_modes": self.failure_modes,
            "rubric": self.rubric,
            "tags": self.tags,
            "expected_output_types": self.expected_output_types,
            "expected_collections": self.expected_collections,
            "thread_policy": self.thread_policy,
            "metadata": self.metadata,
        }


@dataclass
class CaseScore:
    """Score assigned to a benchmark case (manual or future LLM-as-judge)."""

    case_id: str
    score: float  # 0.0 to 1.0
    comment: Optional[str] = None
    scored_by: str = "human"  # "human" | "llm" | ...
    timestamp: Optional[str] = None  # ISO format when scored


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case, with performance data."""

    case_id: str
    thread_id: str
    prompt: str
    outputs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    iterations: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_type_counts: dict[str, int] = field(default_factory=dict)
    artifact_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
