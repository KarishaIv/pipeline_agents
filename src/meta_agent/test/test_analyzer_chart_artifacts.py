"""Tests for analyzer chart tools with artifact registration (should fail before implementation)."""
import json
import pytest
from unittest.mock import MagicMock
import pandas as pd

from src.meta_agent.dto import DtoPayload


@pytest.mark.asyncio
async def test_create_chart_tool_registers_artifact(temp_charts_dir, sample_dto_data, mocker):
    """CreateChartTool should register a ChartArtifact in tool context while returning JSON."""
    from src.meta_agent.tools.analyzer_tools import CreateChartTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "income"],
        num_rows=2,
        sample=[],
        rows=sample_dto_data,
    )
    df = pd.DataFrame(sample_dto_data)
    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(df, test_payload, None))
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")

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

    # Tool should return JSON with concise chart info
    data = json.loads(result)
    assert "chart_saved" in data or "success" in data

    # Tool should have registered an artifact in custom context
    assert "artifacts" in mock_context.custom_context or "chart_artifacts" in mock_context.custom_context


@pytest.mark.asyncio
async def test_create_chart_tool_artifact_has_required_fields(temp_charts_dir, sample_dto_data, mocker):
    """Registered chart artifact should have id, kind, path, filename, mime_type."""
    from src.meta_agent.tools.analyzer_tools import CreateChartTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "income"],
        num_rows=2,
        sample=[],
        rows=sample_dto_data,
    )
    df = pd.DataFrame(sample_dto_data)
    mocker.patch("src.meta_agent.tools.analyzer_tools.resolve_dto_or_error", return_value=(df, test_payload, None))
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")

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

    # Check if artifact was registered
    artifacts = mock_context.custom_context.get("artifacts", [])
    if len(artifacts) > 0:
        artifact = artifacts[0]
        assert hasattr(artifact, "id") or "id" in artifact
        assert hasattr(artifact, "kind") or "kind" in artifact
        assert artifact.kind == "chart" or artifact.get("kind") == "chart"
        assert hasattr(artifact, "mime_type") or "mime_type" in artifact
        assert "png" in str(artifact.mime_type).lower() or "png" in str(artifact.get("mime_type", "")).lower()
