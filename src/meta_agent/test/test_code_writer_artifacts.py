"""Tests for code writer artifact propagation."""

import json
from pathlib import Path

import pytest

from src.meta_agent.dto import DtoPayload


@pytest.mark.asyncio
async def test_execute_code_sandbox_has_save_chart():
    """The code execution sandbox should inject a safe save_chart(filename) function."""
    from src.meta_agent.services.code_execution import CodeExecutionService, CodeExecutionConfig

    config = CodeExecutionConfig(timeout=30)
    service = CodeExecutionService(config)

    # Code that uses save_chart should run without NameError
    code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
save_chart('test.png')
print("Chart saved")
"""

    result = await service.execute_async(code)

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "Chart saved" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_tool_registers_chart_artifact(sample_dto_data, tmp_path, mocker):
    """ExecuteCodeTool should register charts created by save_chart()."""
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

    tool = ExecuteCodeTool(
        reasoning="Create visualization",
        dto_names=["test_dto"],
        code="plt.plot([1, 2, 3])\nsave_chart('output.png')\nprint('done')",
    )

    mock_context = type("Context", (), {})()
    mock_context.custom_context = {}

    result = await tool(mock_context, object())
    payload = json.loads(result)
    artifacts = mock_context.custom_context["artifacts"]

    assert payload["output"] == "done"
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.kind == "chart"
    assert artifact.mime_type == "image/png"
    assert artifact.filename.endswith(".png")
    assert artifact.metadata["source"] == "code_execution"
    assert Path(artifact.path).exists()


@pytest.mark.asyncio
async def test_execute_code_tool_does_not_register_artifact_without_created_files(sample_dto_data, tmp_path, mocker):
    """ExecuteCodeTool should leave artifact list empty when code creates no files."""
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

    tool = ExecuteCodeTool(
        reasoning="Generate chart",
        dto_names=["test_dto"],
        code="print('Analysis complete')",
    )

    mock_context = type("Context", (), {})()
    mock_context.custom_context = {"artifacts": []}

    result = await tool(mock_context, object())
    payload = json.loads(result)

    assert payload["output"] == "Analysis complete"
    assert mock_context.custom_context["artifacts"] == []
