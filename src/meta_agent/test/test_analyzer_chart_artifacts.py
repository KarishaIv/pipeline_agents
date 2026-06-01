"""Tests for analyzer chart tools with artifact registration."""
import json
import pytest
from unittest.mock import MagicMock
import pandas as pd

from src.meta_agent.dto import DtoPayload


@pytest.mark.asyncio
async def test_create_chart_tool_registers_artifact(temp_charts_dir, sample_dto_data, mocker):
    """CreateChartTool should register a chart AgentArtifact and return save metadata."""
    from src.meta_agent.tools.analyzer_tools import CreateChartTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "income"],
        rows=sample_dto_data,
    )
    df = pd.DataFrame(sample_dto_data)
    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(df, test_payload, None))

    tool = CreateChartTool(
        reasoning="Visualize data",
        dto_name="test_dto",
        chart_type="bar",
        title="Test Chart",
    )

    mock_context = MagicMock()
    mock_context.custom_context = {}
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    data = json.loads(result)
    assert data["success"] is True
    assert data["title"] == "Test Chart"
    assert data["type"] == "bar"
    assert data["dto"]["dto_name"] == "test_dto"

    artifacts = mock_context.custom_context["artifacts"]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.kind == "chart"
    assert artifact.caption == "Test Chart"
    assert artifact.mime_type == "image/png"
    assert artifact.metadata == {"chart_type": "bar", "dto_name": "test_dto"}
    assert artifact.filename.endswith(".png")
    artifact_path = temp_charts_dir / artifact.filename
    assert artifact.path == str(artifact_path)
    assert artifact_path.exists()


@pytest.mark.asyncio
async def test_create_chart_tool_artifact_has_required_fields(temp_charts_dir, sample_dto_data, mocker):
    """Registered chart artifact should have id, kind, path, filename, mime_type."""
    from src.meta_agent.tools.analyzer_tools import CreateChartTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "income"],
        rows=sample_dto_data,
    )
    df = pd.DataFrame(sample_dto_data)
    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(df, test_payload, None))

    tool = CreateChartTool(
        reasoning="Visualize",
        dto_name="test_dto",
        chart_type="line",
        title="Trend",
    )

    mock_context = MagicMock()
    mock_context.custom_context = {"artifacts": []}
    mock_config = MagicMock()

    await tool(mock_context, mock_config)

    artifacts = mock_context.custom_context.get("artifacts", [])
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.id
    assert artifact.kind == "chart"
    artifact_path = temp_charts_dir / artifact.filename
    assert artifact.path == str(artifact_path)
    assert artifact_path.exists()
    assert artifact.mime_type == "image/png"
    assert artifact.caption == "Trend"
    assert artifact.metadata == {"chart_type": "line", "dto_name": "test_dto"}
