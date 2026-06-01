"""Tests for converting artifacts to ImageOutput in _finalize_invoke (regression tests for bug fix)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.meta_agent.graph import MetaAgentGraphManager
from src.meta_agent.output_models import ImageOutput, AgentArtifact


@pytest.mark.asyncio
async def test_finalize_invoke_converts_chart_artifacts_to_imageoutput():
    """_finalize_invoke should convert AgentArtifact with kind='chart' into ImageOutput objects.

    This ensures that chart artifacts saved during execution are exposed to users
    via the /ask API endpoint as ImageOutput with proper URLs.
    """
    manager = MetaAgentGraphManager()
    graph = MagicMock()
    graph.aupdate_state = AsyncMock()
    manager._graph = graph

    # Simulate final graph state with a chart artifact
    artifact = AgentArtifact(
        id="chart-001",
        kind="chart",
        path="/Users/aleksandr/HSE/Diplom/pipeline_agents/charts/chart_20260502_120000_000000.png",
        filename="chart_20260502_120000_000000.png",
        mime_type="image/png",
        caption="Revenue Trend",
        metadata={"chart_type": "bar", "dto_name": "sales_data"},
    )

    result = {
        "outputs": [],  # No text outputs
        "artifacts": [artifact],  # But we have a chart artifact
        "history": [{"role": "supervisor", "content": "Analysis complete"}],
        "dto_store": {},
    }

    runnable_config = {"configurable": {"thread_id": "t-1"}}

    with patch(
        "src.meta_agent.graph.build_persisted_history",
        new=AsyncMock(return_value=[{"role": "history_summary", "content": "summary"}]),
    ):
        outputs = await manager._finalize_invoke(runnable_config, result)

    # Should return outputs with ImageOutput created from the chart artifact
    assert len(outputs) >= 1
    # Find ImageOutput in the outputs
    image_outputs = [o for o in outputs if isinstance(o, ImageOutput)]
    assert len(image_outputs) >= 1, "Chart artifact should be converted to ImageOutput"

    image_output = image_outputs[0]
    assert image_output.url  # Should have a URL (e.g., http://api/artifacts/chart-001.png)
    assert "chart" in image_output.url.lower()
    assert image_output.caption == "Revenue Trend"
    assert image_output.mime_type == "image/png"


@pytest.mark.asyncio
async def test_finalize_invoke_mixes_text_outputs_and_chart_artifacts():
    """_finalize_invoke should handle both TextOutput and ImageOutput from artifacts in same response."""
    from src.meta_agent.output_models import TextOutput

    manager = MetaAgentGraphManager()
    graph = MagicMock()
    graph.aupdate_state = AsyncMock()
    manager._graph = graph

    # Both text and chart
    artifact = AgentArtifact(
        id="chart-002",
        kind="chart",
        path="/app/charts/pie_chart.png",
        filename="pie_chart.png",
        mime_type="image/png",
        caption="Sales Distribution",
    )

    result = {
        "outputs": [TextOutput(text="Here is your analysis:")],
        "artifacts": [artifact],
        "history": [],
        "dto_store": {},
    }

    runnable_config = {"configurable": {"thread_id": "t-2"}}

    with patch(
        "src.meta_agent.graph.build_persisted_history",
        new=AsyncMock(return_value=[]),
    ):
        outputs = await manager._finalize_invoke(runnable_config, result)

    # Should have both text and image
    assert len(outputs) >= 2
    text_outputs = [o for o in outputs if isinstance(o, TextOutput)]
    image_outputs = [o for o in outputs if isinstance(o, ImageOutput)]

    assert len(text_outputs) >= 1
    assert len(image_outputs) >= 1
    assert text_outputs[0].text == "Here is your analysis:"
    assert "Sales Distribution" in image_outputs[0].caption or image_outputs[0].caption == "Sales Distribution"


@pytest.mark.asyncio
async def test_finalize_invoke_converts_json_and_csv_artifacts(tmp_path):
    """_finalize_invoke should expose JSON as JsonOutput and CSV as FileOutput."""
    from src.meta_agent.output_models import JsonOutput, FileOutput

    manager = MetaAgentGraphManager()
    graph = MagicMock()
    graph.aupdate_state = AsyncMock()
    manager._graph = graph

    json_path = tmp_path / "raw.json"
    json_path.write_text('[{"name": "Alice", "score": 10}]', encoding="utf-8")
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text("name,score\nAlice,10\n", encoding="utf-8")

    result = {
        "outputs": [],
        "artifacts": [
            AgentArtifact(
                kind="data",
                path=str(json_path),
                filename="raw.json",
                mime_type="application/json",
                caption="Raw JSON",
            ),
            AgentArtifact(
                kind="csv",
                path=str(csv_path),
                filename="raw.csv",
                mime_type="text/csv",
                caption="Raw CSV",
            ),
        ],
        "history": [],
        "dto_store": {},
    }

    with patch(
        "src.meta_agent.graph.build_persisted_history",
        new=AsyncMock(return_value=[]),
    ):
        outputs = await manager._finalize_invoke({"configurable": {"thread_id": "t-3"}}, result)

    json_outputs = [o for o in outputs if isinstance(o, JsonOutput)]
    file_outputs = [o for o in outputs if isinstance(o, FileOutput)]

    assert json_outputs[0].data == [{"name": "Alice", "score": 10}]
    assert json_outputs[0].caption == "Raw JSON"
    assert file_outputs[0].download_url == "/artifacts/raw.csv"
    assert file_outputs[0].mime_type == "text/csv"
