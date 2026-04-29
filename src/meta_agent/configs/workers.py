"""Worker configuration for meta-agent.

Defines worker types, configurations, and the centralized worker definitions dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from src.meta_agent.configs.runtime import BIG_MODEL
from src.meta_agent.prompts import (
    ANALYZER_SYSTEM,
    CODE_WRITER_SYSTEM,
    EXTRACTOR_SYSTEM,
    SUPERVISOR_SYSTEM,
)
from src.meta_agent.tools import (
    AnalyzerDecisionTool,
    CodeExecutionReportTool,
    ComputeStatsTool,
    CreateChartTool,
    DataExtractionReportTool,
    ExecuteCodeTool,
    ListDtoNamesTool,
    QdrantCollectionSchema,
    QdrantFilterTool,
    QdrantRetrieveTool,
    QdrantScrollTool,
    QdrantSearchTool,
    RemainingStepsTool,
    SampleDtoTool,
    SummarizeTextsTool,
    SupervisorDecisionTool,
    ValidateCodeTool,
)

WorkerName = Literal["supervisor", "data_extractor", "analyzer", "code_writer"]


@dataclass(frozen=True)
class WorkerDefinition:
    """Unified configuration for a meta-agent worker.

    Combines LLM setup (tools, prompts, model) with structured execution policy
    (parsing, formatting, fallback behavior).

    Attributes:
        worker_name: Identifier for the worker (supervisor, data_extractor, etc.)
        tools: Tool classes passed to the agent
        system_prompt: System prompt for the agent
        report_tool: Pydantic model to parse worker output (e.g., SupervisorDecisionTool)
        fallback_on_parse_error: Whether to generate json_node_failure on parse failure
        format_content: Optional function to format parsed report into history content.
                       If None, uses raw output.
        model_override: Optional model override (e.g., BIG_MODEL for code_writer)
    """
    worker_name: str
    tools: list
    system_prompt: str
    report_tool: type
    fallback_on_parse_error: bool = True
    format_content: Callable[[Any], str] | None = field(default=None)
    model_override: str | None = field(default=None)


WORKER_DEFINITIONS: dict[WorkerName, WorkerDefinition] = {
    "supervisor": WorkerDefinition(
        worker_name="supervisor",
        tools=[RemainingStepsTool, SupervisorDecisionTool],
        system_prompt=SUPERVISOR_SYSTEM,
        report_tool=SupervisorDecisionTool,
        fallback_on_parse_error=False,
    ),
    "data_extractor": WorkerDefinition(
        worker_name="data_extractor",
        tools=[
            QdrantCollectionSchema,
            QdrantSearchTool,
            QdrantFilterTool,
            QdrantScrollTool,
            QdrantRetrieveTool,
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            DataExtractionReportTool,
        ],
        system_prompt=EXTRACTOR_SYSTEM,
        report_tool=DataExtractionReportTool,
        fallback_on_parse_error=True,
        format_content=lambda p: f"Кратко: {p.summary}\n\nДанные: {p.dto_references}",
    ),
    "analyzer": WorkerDefinition(
        worker_name="analyzer",
        tools=[
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            SummarizeTextsTool,
            ComputeStatsTool,
            CreateChartTool,
            AnalyzerDecisionTool,
        ],
        system_prompt=ANALYZER_SYSTEM,
        report_tool=AnalyzerDecisionTool,
        fallback_on_parse_error=True,
    ),
    "code_writer": WorkerDefinition(
        worker_name="code_writer",
        tools=[
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            ValidateCodeTool,
            ExecuteCodeTool,
            CodeExecutionReportTool,
        ],
        system_prompt=CODE_WRITER_SYSTEM,
        report_tool=CodeExecutionReportTool,
        fallback_on_parse_error=True,
        model_override=BIG_MODEL,
        format_content=lambda p: (
            f"Задача: {p.task}\n"
            f"Найдено:\n{chr(10).join(f'- {item}' for item in p.findings)}\n\n"
            f"Валидация: {p.validation}\n"
            f"Выполнение: {p.execution}"
        ),
    ),
}
