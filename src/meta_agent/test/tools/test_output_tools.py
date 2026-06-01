"""Tests for output_tools.py - all SystemBaseTool decision and report classes.

Verifies Pydantic models, context.state/execution_result mutation, JSON serialization.
"""
import json
from unittest.mock import MagicMock

import pytest
from sgr_agent_core.models import AgentStatesEnum

from src.meta_agent.tools.output_tools import (
    AnalyzerDecisionTool,
    CodeExecutionReportTool,
    DataExtractionReportTool,
    SupervisorDecisionTool,
)


@pytest.mark.asyncio
async def test_supervisor_decision_tool():
    """Test SupervisorDecisionTool sets COMPLETED state and returns full payload."""
    tool = SupervisorDecisionTool(
        reasoning="Test reasoning",
        next="analyzer",
        task="Analyze the data",
        final_answer="",
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    assert mock_context.state == AgentStatesEnum.COMPLETED
    assert mock_context.execution_result == result
    payload = json.loads(result)
    assert payload["next"] == "analyzer"
    assert payload["reasoning"] == "Test reasoning"
    assert payload["task"] == "Analyze the data"


@pytest.mark.asyncio
async def test_data_extraction_report_tool():
    """Test DataExtractionReportTool with various statuses and fields."""
    tool = DataExtractionReportTool(
        reasoning="Extracted from personas",
        completed_steps=["schema", "search", "sample"],
        summary='{"dtos": ["personas_df"]}',
        dto_references='{"dto_names": ["personas_1"]}',
        status=AgentStatesEnum.COMPLETED,
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    assert mock_context.state == AgentStatesEnum.COMPLETED
    payload = json.loads(result)
    assert len(payload["completed_steps"]) == 3
    assert payload["status"] == "completed"
    assert "personas" in str(payload)


@pytest.mark.asyncio
async def test_code_execution_report_tool():
    """Test CodeExecutionReportTool for code_writer completion."""
    tool = CodeExecutionReportTool(
        reasoning="Validated and executed stats computation",
        task="Compute correlation matrix",
        code="import pandas as pd; ...",
        validation='{"is_runnable": true}',
        execution='{"output": "correlation=0.85"}',
        findings=["strong correlation found"],
        status=AgentStatesEnum.COMPLETED,
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    assert mock_context.state == AgentStatesEnum.COMPLETED
    payload = json.loads(result)
    assert "correlation" in payload["execution"]
    assert payload["code"]  # not empty
    assert len(payload["findings"]) == 1


@pytest.mark.asyncio
async def test_analyzer_decision_tool_report():
    """Test AnalyzerDecisionTool with 'report' decision."""
    tool = AnalyzerDecisionTool(
        reasoning="Data shows clear patterns",
        decision="report",
        key_findings=["high income correlation", "age distribution normal"],
        conclusions="Target audience prefers conservative credit products.",
        status=AgentStatesEnum.COMPLETED,
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    assert mock_context.state == AgentStatesEnum.COMPLETED
    payload = json.loads(result)
    assert payload["decision"] == "report"
    assert len(payload["key_findings"]) == 2
    assert "conservative" in payload["conclusions"]


@pytest.mark.asyncio
async def test_analyzer_decision_tool_delegate():
    """Test AnalyzerDecisionTool with 'delegate' decision for code_writer."""
    tool = AnalyzerDecisionTool(
        reasoning="Complex computation needed",
        decision="delegate",
        task="Create correlation heatmap from personas DTO",
        delegate_reason="Requires matplotlib visualization beyond simple stats",
        status=AgentStatesEnum.COMPLETED,
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    assert mock_context.state == AgentStatesEnum.COMPLETED
    payload = json.loads(result)
    assert payload["decision"] == "delegate"
    assert "heatmap" in payload["task"]
    assert "matplotlib" in payload["delegate_reason"]


def test_all_tools_have_correct_names_and_descriptions():
    """Verify tool metadata for registration."""
    tools = [
        SupervisorDecisionTool(reasoning="meta", next="end", final_answer="ok"),
        DataExtractionReportTool(reasoning="meta"),
        CodeExecutionReportTool(reasoning="meta"),
        AnalyzerDecisionTool(reasoning="meta", decision="report"),
    ]
    expected_names = {
        "supervisor_decision",
        "data_extraction_report",
        "code_execution_report",
        "analyzer_decision",
    }

    for instance in tools:
        assert instance.tool_name in expected_names
        assert len(instance.description) > 20
        assert hasattr(instance, "model_dump_json")
