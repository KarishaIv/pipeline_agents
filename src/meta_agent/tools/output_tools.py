"""Structured output SystemBaseTool classes — one per agent role.

Each tool terminates the agent loop by setting context.state and
context.execution_result, then returns a validated JSON payload.
"""

from __future__ import annotations

from typing import List, Literal, TYPE_CHECKING

from pydantic import Field

from sgr_agent_core.base_tool import SystemBaseTool
from sgr_agent_core.models import AgentStatesEnum

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext


class SupervisorDecisionTool(SystemBaseTool):
    """Record the supervisor's routing decision and terminate the supervisor step."""

    tool_name = "supervisor_decision"
    description = "Record the supervisor's routing decision and terminate the supervisor step."

    reasoning: str = Field(
        description="Analysis of the current state and justification for this decision"
    )
    next: Literal["data_extractor", "analyzer", "end"] = Field(
        description=(
            "Which worker to call next: "
            "'data_extractor' to fetch more data, "
            "'analyzer' to synthesise what was collected, "
            "'end' when the question is fully answered"
        )
    )
    task: str = Field(
        description=(
            "High-level task description for the chosen worker. "
            "State WHAT needs to be done, not HOW. "
            "Leave empty when next='end'."
        )
    )
    final_answer: str = Field(
        default="",
        description="Complete answer for the user — filled only when next='end'",
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = AgentStatesEnum.COMPLETED
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload


class DataExtractionReportTool(SystemBaseTool):
    """Report structured findings after Qdrant data extraction and terminate the extractor step."""

    tool_name = "data_extraction_report"
    description = "Report structured findings after Qdrant data extraction and terminate the extractor step."

    reasoning: str = Field(description="What was searched, which collections, and why")
    completed_steps: List[str] = Field(
        description="Ordered list of extraction steps performed",
        min_length=1,
        max_length=10,
    )
    summary: str = Field(
        description="Concise human-readable summary of what was found"
    )
    raw_data: str = Field(
        description="All retrieved records serialised as a JSON string"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        description="Extraction status"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload


class AnalysisReportTool(SystemBaseTool):
    """Report structured analytical conclusions and terminate the analyzer step."""

    tool_name = "analysis_report"
    description = "Report structured analytical conclusions and terminate the analyzer step."

    reasoning: str = Field(description="Analytical approach and methodology used")
    completed_steps: List[str] = Field(
        description="Ordered list of analysis steps performed",
        min_length=1,
        max_length=10,
    )
    key_findings: List[str] = Field(
        description="Key findings and patterns identified from the data"
    )
    conclusions: str = Field(
        description="Comprehensive analytical conclusions in Russian"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        description="Analysis status"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload
