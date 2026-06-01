"""Tests for ExecuteCodeTool artifact registration (regression test for code_writer chart bug)."""

import json

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.meta_agent.dto import DtoPayload


@pytest.mark.asyncio
async def test_execute_code_tool_registers_saved_charts_as_artifacts(sample_dto_data, tmp_path, mocker):
    """ExecuteCodeTool should detect and register files created by save_chart() as artifacts.

    When user code calls save_chart(), the tool should:
    1. Capture the returned file path
    2. Create an AgentArtifact for it
    3. Register it in context.custom_context['artifacts']
    """
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["id", "age", "income", "text"],
        rows=sample_dto_data,
    )

    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, test_payload, None),
    )
    mocker.patch("src.meta_agent.tools.code_writer_tools.CHARTS_DIR", tmp_path)

    mock_executor = MagicMock()
    mock_result = MagicMock()
    mock_result.stdout = "Chart saved"
    mock_result.stderr = ""
    mock_result.exit_code = 0

    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.CodeExecutionService",
        return_value=mock_executor,
    )

    async def execute_and_create_file(_code: str):
        (tmp_path / "output.png").write_bytes(b"fake-png")
        return mock_result

    mock_executor.execute_async = AsyncMock(side_effect=execute_and_create_file)

    tool = ExecuteCodeTool(
        reasoning="Create visualization",
        dto_names=["test_dto"],
        code="import matplotlib.pyplot as plt\nplt.plot([1,2,3])\nsave_chart('output.png')\nprint('Chart saved')"
    )

    mock_context = MagicMock()
    mock_context.custom_context = {"artifacts": []}
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    payload = json.loads(result)
    artifacts = mock_context.custom_context["artifacts"]

    assert payload["success"] is True
    assert payload["output"] == "Chart saved"
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.kind == "chart"
    assert artifact.filename == "output.png"
    assert artifact.path == str(tmp_path / "output.png")
    assert artifact.mime_type == "image/png"
    assert artifact.caption == "CHART: output.png"
    assert artifact.metadata == {"source": "code_execution"}
