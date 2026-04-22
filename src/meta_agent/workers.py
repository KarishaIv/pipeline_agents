"""Worker configurations for meta-agent nodes.

Centralized, typed, extensible configuration for each worker (supervisor, data_extractor,
analyzer, code_writer). This separates config from node implementation logic.
"""

from dataclasses import dataclass
from typing import Literal

from src.meta_agent.config import BIG_MODEL
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
class WorkerConfig:
    """Configuration for each meta-agent worker.
    """
    tools: list
    system_prompt: str
    default_model: str | None = None


WORKER_CONFIGS: dict[WorkerName, WorkerConfig] = {
    "supervisor": WorkerConfig(
        tools=[RemainingStepsTool, SupervisorDecisionTool],
        system_prompt=SUPERVISOR_SYSTEM,
    ),
    "data_extractor": WorkerConfig(
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
    ),
    "analyzer": WorkerConfig(
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
    ),
    "code_writer": WorkerConfig(
        tools=[
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            ValidateCodeTool,
            ExecuteCodeTool,
            CodeExecutionReportTool,
        ],
        system_prompt=CODE_WRITER_SYSTEM,
        default_model=BIG_MODEL,
    ),
}


def _get_worker_config(worker_name: str) -> WorkerConfig:
    """Return structured config for a worker.
    """
    if worker_name not in WORKER_CONFIGS:
        raise ValueError(f"Unknown worker: {worker_name}. Must be one of {list(WORKER_CONFIGS.keys())}")
    return WORKER_CONFIGS[worker_name]
