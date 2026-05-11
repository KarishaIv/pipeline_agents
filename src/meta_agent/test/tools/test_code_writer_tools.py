"""Tests for code_writer_tools.py - ValidateCodeTool and ExecuteCodeTool tool wrappers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.meta_agent.dto import DtoPayload


@pytest.mark.asyncio
async def test_validate_code_tool_safe_code(mocker):
    """Test ValidateCodeTool with import-free safe code."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    test_payload = DtoPayload(
        summary_text="test",
        columns=[],
        num_rows=0,
        sample=[],
        rows=[],
    )
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, test_payload, None),
    )

    safe_code = "values = [1, 2, 3]; print(sum(values) / len(values))"
    safe_tool = ValidateCodeTool(
        reasoning="Validate safe snippet",
        dto_names=["test_dto"],
        code=safe_code,
    )
    result_safe = await safe_tool(mock_context, mock_config)
    data_safe = json.loads(result_safe)
    assert data_safe.get("is_runnable", False) is True
    assert "warnings" in data_safe
    assert not any("import" in warning.lower() for warning in data_safe.get("warnings", []))


@pytest.mark.asyncio
async def test_validate_code_tool_pandas_import_is_warned(mocker):
    """Test ValidateCodeTool flags pandas import with warning."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    pandas_import_code = "import pandas as pd; df = pd.DataFrame(); print(df.describe())"
    pandas_import_tool = ValidateCodeTool(
        reasoning="Validate legacy pandas import snippet",
        dto_names=["test_dto"],
        code=pandas_import_code,
    )
    result_pandas_import = await pandas_import_tool(mock_context, mock_config)
    data_pandas_import = json.loads(result_pandas_import)
    assert data_pandas_import.get("is_runnable", False) is True
    assert any("импорт" in warning.lower() for warning in data_pandas_import.get("warnings", []))


@pytest.mark.asyncio
async def test_validate_code_tool_dangerous_code_is_warned(mocker):
    """Test ValidateCodeTool flags dangerous patterns with warnings."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    dangerous_code = "import os; os.system('rm -rf /'); exec('print(1)')"
    dangerous_tool = ValidateCodeTool(
        reasoning="Validate dangerous snippet",
        dto_names=["test_dto"],
        code=dangerous_code,
    )
    result_danger = await dangerous_tool(mock_context, mock_config)
    data_danger = json.loads(result_danger)
    assert data_danger.get("is_runnable", False) is True
    assert any("запрещ" in w.lower() or "import" in w.lower() for w in data_danger.get("warnings", []))


@pytest.mark.asyncio
async def test_validate_code_tool_empty_code_returns_error(mocker):
    """Test ValidateCodeTool rejects empty code."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    tool = ValidateCodeTool(reasoning="Validate empty code", dto_names=["test_dto"], code="   ")
    result = await tool(mock_context, mock_config)
    payload = json.loads(result)

    assert payload["is_runnable"] is False
    assert any("пустой" in err.lower() for err in payload.get("errors", []))


@pytest.mark.asyncio
async def test_validate_code_tool_syntax_error_returns_error(mocker):
    """Test ValidateCodeTool reports syntax errors."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    tool = ValidateCodeTool(
        reasoning="Validate syntax failure",
        dto_names=["test_dto"],
        code="for i in range(3) print(i)",
    )
    result = await tool(mock_context, mock_config)
    payload = json.loads(result)

    assert payload["is_runnable"] is False
    assert any("синтакс" in err.lower() for err in payload.get("errors", []))


@pytest.mark.asyncio
async def test_execute_code_tool_with_dto(temp_charts_dir, sample_dto_data, mocker):
    """Test ExecuteCodeTool delegates to CodeExecutionService."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "id", "income", "text"],
        num_rows=len(sample_dto_data),
        sample=[],
        rows=sample_dto_data,
    )
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, test_payload, None),
    )

    # Mock CodeExecutionService
    mock_executor = AsyncMock()
    mock_executor.execute_async.return_value = MagicMock(
        stdout="Stats computed",
        stderr="",
        exit_code=0,
        timeout_occurred=False,
    )

    with patch(
        "src.meta_agent.tools.code_writer_tools.CodeExecutionService",
        return_value=mock_executor,
    ):
        tool = ExecuteCodeTool(
            reasoning="Run stats snippet",
            dto_names=["test_dto"],
            code="df.describe(); print('Stats computed')",
        )
        mock_context = MagicMock()
        mock_config = MagicMock()

        result = await tool(mock_context, mock_config)

        data = json.loads(result)
        assert "output" in data
        assert "Stats computed" in data["output"]
        assert data["dto_names"] == ["test_dto"]


@pytest.mark.asyncio
async def test_execute_code_tool_sets_default_output_when_silent(sample_dto_data, mocker):
    """Test ExecuteCodeTool returns default output when code is silent."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "id", "income", "text"],
        num_rows=len(sample_dto_data),
        sample=[],
        rows=sample_dto_data,
    )
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, test_payload, None),
    )

    # Mock CodeExecutionService
    mock_executor = AsyncMock()
    mock_executor.execute_async.return_value = MagicMock(
        stdout="",
        stderr="",
        exit_code=0,
        timeout_occurred=False,
    )

    with patch(
        "src.meta_agent.tools.code_writer_tools.CodeExecutionService",
        return_value=mock_executor,
    ):
        tool = ExecuteCodeTool(reasoning="Silent run", dto_names=["test_dto"], code="x = 1")
        result = await tool(MagicMock(), MagicMock())
        payload = json.loads(result)

        assert payload["dto_names"] == ["test_dto"]
        assert payload["output"] == "(нет вывода)"


@pytest.mark.asyncio
async def test_execute_code_tool_returns_error_output(sample_dto_data, mocker):
    """Test ExecuteCodeTool includes error text in response."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["age", "id", "income", "text"],
        num_rows=len(sample_dto_data),
        sample=[],
        rows=sample_dto_data,
    )
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, test_payload, None),
    )

    # Mock CodeExecutionService
    mock_executor = AsyncMock()
    mock_executor.execute_async.return_value = MagicMock(
        stdout="",
        stderr="ValueError: boom",
        exit_code=1,
        timeout_occurred=False,
    )

    with patch(
        "src.meta_agent.tools.code_writer_tools.CodeExecutionService",
        return_value=mock_executor,
    ):
        tool = ExecuteCodeTool(reasoning="Failing run", dto_names=["test_dto"], code="raise ValueError('x')")
        result = await tool(MagicMock(), MagicMock())
        payload = json.loads(result)

        assert payload["dto_names"] == ["test_dto"]
        assert payload["error"] == "ValueError: boom"


def test_code_writer_tool_metadata():
    """Verify tool names and descriptions are correct."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool, ValidateCodeTool

    tools = [
        ValidateCodeTool(reasoning="meta", dto_names=["dto"], code="print(1)"),
        ExecuteCodeTool(reasoning="meta", dto_names=["dto"], code="print(1)"),
    ]
    for tool in tools:
        name = tool.tool_name
        assert name in ["validate_code", "execute_code"]
        assert len(tool.description) > 20


@pytest.mark.asyncio
async def test_execute_code_tool_creates_service_with_correct_config(sample_dto_data, mocker):
    """Test ExecuteCodeTool creates CodeExecutionService with correct configuration."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    test_payload = DtoPayload(
        summary_text="test",
        columns=["x"],
        num_rows=1,
        sample=[],
        rows=[{"x": 1}],
    )
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, test_payload, None),
    )

    # Track CodeExecutionService initialization
    init_calls = []

    def track_init(config):
        init_calls.append(config)
        executor = AsyncMock()
        executor.execute_async.return_value = MagicMock(
            stdout="test",
            stderr="",
            exit_code=0,
            timeout_occurred=False,
        )
        return executor

    with patch(
        "src.meta_agent.tools.code_writer_tools.CodeExecutionService",
        side_effect=track_init,
    ):
        tool = ExecuteCodeTool(reasoning="test", dto_names=["test_dto"], code="print('test')")
        await tool(MagicMock(), MagicMock())

        assert len(init_calls) == 1
        config = init_calls[0]
        assert "test_dto" in config.dto_payloads
        assert config.dto_payloads["test_dto"] == test_payload
        assert config.timeout > 0  # Should have a timeout value
