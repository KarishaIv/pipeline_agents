"""Tests for analyzer_tools.py - ComputeStatsTool, CreateChartTool, SummarizeTextsTool."""
import json
from typing import get_args
from unittest.mock import MagicMock

import pandas as pd
import pytest

@pytest.mark.asyncio
async def test_compute_stats_tool(sample_dto_data, mocker):
    """Test ComputeStatsTool with pandas DataFrame stats."""
    from src.meta_agent.tools.analyzer_tools import ComputeStatsTool

    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(
        pd.DataFrame(sample_dto_data), {"num_rows": 2}, None
    ))

    tool = ComputeStatsTool(
        reasoning="Need descriptive stats",
        dto_name="test_dto",
        columns=["age", "income"],
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    data = json.loads(result)
    assert "describe" in data
    assert "error" not in data


@pytest.mark.asyncio
async def test_compute_stats_tool_without_numeric_columns_returns_error(mocker):
    """Test stats tool returns error when DTO has no numeric columns."""
    from src.meta_agent.tools.analyzer_tools import ComputeStatsTool

    mocker.patch(
        "src.meta_agent.tools.analyzer_tools.resolve_dto_or_error",
        return_value=(pd.DataFrame([{"text": "a"}, {"text": "b"}]), {"num_rows": 2}, None),
    )
    tool = ComputeStatsTool(reasoning="Try stats on text only", dto_name="text_dto")

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert "error" in payload


@pytest.mark.asyncio
async def test_compute_stats_tool_exception_branch_returns_error(sample_dto_data, mocker):
    """Test ComputeStatsTool catches unexpected exceptions and returns error JSON."""
    from src.meta_agent.tools.analyzer_tools import ComputeStatsTool

    mocker.patch(
        "src.meta_agent.tools.analyzer_tools.resolve_dto_or_error",
        return_value=(pd.DataFrame(sample_dto_data), {"num_rows": 2}, None),
    )
    mocker.patch("src.meta_agent.tools.analyzer_tools.dto_summary_view", side_effect=RuntimeError("summary fail"))
    tool = ComputeStatsTool(reasoning="Force exception", dto_name="test_dto")

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert payload["error"] == "summary fail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chart_type",
    ["bar", "line", "scatter", "histogram", "pie", "box", "heatmap"],
)
async def test_create_chart_tool_success_by_type(chart_type, temp_charts_dir, sample_dto_data, mocker):
    """Test CreateChartTool success path for every supported chart type."""
    from src.meta_agent.tools.analyzer_tools import CreateChartTool

    allowed_chart_types = set(get_args(CreateChartTool.model_fields["chart_type"].annotation))
    tested_chart_types = {"bar", "line", "scatter", "histogram", "pie", "box", "heatmap"}
    assert tested_chart_types == allowed_chart_types

    df = pd.DataFrame(sample_dto_data)
    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(df, {}, None))
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")

    tool = CreateChartTool(
        reasoning=f"Visualize using {chart_type}",
        dto_name="test_dto",
        chart_type=chart_type,
        title="Test Chart",
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    data = json.loads(result)
    assert "chart_saved" in data
    assert data["type"] == chart_type


@pytest.mark.asyncio
async def test_create_chart_tool_scatter_requires_numeric_columns(mocker):
    """Test scatter returns error when insufficient numeric columns."""
    from src.meta_agent.tools.analyzer_tools import CreateChartTool

    df = pd.DataFrame([{"label": "a"}, {"label": "b"}])
    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(df, {}, None))
    
    tool = CreateChartTool(reasoning="Scatter fail", dto_name="test_dto", chart_type="scatter", title="Scatter")
    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert "error" in payload


@pytest.mark.asyncio
async def test_summarize_texts_tool(mock_openai_client, sample_dto_data, mocker):
    """Test SummarizeTextsTool with mocked LLM response."""
    from src.meta_agent.tools.analyzer_tools import SummarizeTextsTool

    # Mock successful LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Summary: High income group prefers X."))]
    mock_openai_client.chat.completions.create.return_value = mock_response

    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(
        pd.DataFrame(sample_dto_data), {}, None
    ))
    mocker.patch("src.meta_agent.tools.analyzer_tools.get_model_uri", return_value="test-model")

    tool = SummarizeTextsTool(
        reasoning="Extract key patterns",
        dto_name="test_dto",
        text_columns=["text"],
        max_items=10,
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    data = json.loads(result)
    assert "summary" in data or "High income" in str(data)
    mock_openai_client.chat.completions.create.assert_called()


@pytest.mark.asyncio
async def test_summarize_texts_tool_returns_error_when_no_text(mock_openai_client, mocker):
    """Test summarize_texts returns error when DTO has no text content."""
    from src.meta_agent.tools.analyzer_tools import SummarizeTextsTool

    mocker.patch(
        "src.meta_agent.tools.analyzer_tools.resolve_dto_or_error",
        return_value=(pd.DataFrame([{"age": 20}, {"age": 30}]), {}, None),
    )
    tool = SummarizeTextsTool(reasoning="Try summary", dto_name="test_dto", text_columns=["text"])

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert "error" in payload


@pytest.mark.asyncio
async def test_summarize_texts_tool_llm_exception_returns_error(mock_openai_client, sample_dto_data, mocker):
    """Test summarize_texts gracefully handles LLM exceptions."""
    from src.meta_agent.tools.analyzer_tools import SummarizeTextsTool

    mock_openai_client.chat.completions.create.side_effect = RuntimeError("llm down")
    mocker.patch(
        "src.meta_agent.tools.analyzer_tools.resolve_dto_or_error",
        return_value=(pd.DataFrame(sample_dto_data), {}, None),
    )
    tool = SummarizeTextsTool(reasoning="Try summary", dto_name="test_dto", text_columns=["text"])

    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)
    assert "error" in payload
    assert "llm down" in payload["error"]


@pytest.mark.asyncio
async def test_analyzer_tools_error_paths(mocker):
    """Test error handling in analyzer tools (empty data, invalid columns)."""
    from src.meta_agent.tools.analyzer_tools import ComputeStatsTool, CreateChartTool

    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(None, None, '{"error": "DTO not found"}'))

    tool_stats = ComputeStatsTool(reasoning="Check error path", dto_name="missing")
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool_stats(mock_context, mock_config)
    assert "error" in result.lower() or "DTO" in result

    tool_chart = CreateChartTool(
        reasoning="Check error path",
        dto_name="missing",
        chart_type="bar",
        title="Error chart",
    )
    result_chart = await tool_chart(mock_context, mock_config)
    assert "error" in result_chart.lower()


def test_analyzer_tool_metadata():
    """Verify tool names and descriptions."""
    from src.meta_agent.tools.analyzer_tools import ComputeStatsTool, CreateChartTool, SummarizeTextsTool

    tools = [
        ComputeStatsTool(reasoning="meta", dto_name="dto"),
        CreateChartTool(reasoning="meta", dto_name="dto", chart_type="bar", title="t"),
        SummarizeTextsTool(reasoning="meta", dto_name="dto"),
    ]
    for tool in tools:
        assert tool.tool_name in ["compute_stats", "create_chart", "summarize_texts"]
        assert len(tool.description) > 30
