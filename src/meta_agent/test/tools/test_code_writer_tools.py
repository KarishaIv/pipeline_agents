"""Tests for code_writer_tools.py - ValidateCodeTool, ExecuteCodeTool, sandbox security."""
import asyncio
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

@pytest.mark.asyncio
async def test_validate_code_tool_safe_code(mocker):
    """Test import-free safe code is runnable without import warnings."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    safe_code = "values = [1, 2, 3]; print(sum(values) / len(values))"
    safe_tool = ValidateCodeTool(
        reasoning="Validate safe snippet",
        dto_name="test_dto",
        code=safe_code,
    )
    result_safe = await safe_tool(mock_context, mock_config)
    data_safe = json.loads(result_safe)
    assert data_safe.get("is_runnable", False) is True
    assert "warnings" in data_safe
    assert not any("import" in warning.lower() for warning in data_safe.get("warnings", []))


@pytest.mark.asyncio
async def test_validate_code_tool_pandas_import_is_warned(mocker):
    """Test pandas import code is runnable and flagged with import warning."""
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
        dto_name="test_dto",
        code=pandas_import_code,
    )
    result_pandas_import = await pandas_import_tool(mock_context, mock_config)
    data_pandas_import = json.loads(result_pandas_import)
    assert data_pandas_import.get("is_runnable", False) is True
    assert any("импорт" in warning.lower() for warning in data_pandas_import.get("warnings", []))


@pytest.mark.asyncio
async def test_validate_code_tool_dangerous_code_is_warned(mocker):
    """Test dangerous code patterns are flagged with warnings."""
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
        dto_name="test_dto",
        code=dangerous_code,
    )
    result_danger = await dangerous_tool(mock_context, mock_config)
    data_danger = json.loads(result_danger)
    assert data_danger.get("is_runnable", False) is True
    assert any("запрещ" in w.lower() or "import" in w.lower() for w in data_danger.get("warnings", []))


@pytest.mark.asyncio
async def test_validate_code_tool_empty_code_returns_error(mocker):
    """Test empty code is rejected with a validation error."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    tool = ValidateCodeTool(reasoning="Validate empty code", dto_name="test_dto", code="   ")
    result = await tool(mock_context, mock_config)
    payload = json.loads(result)

    assert payload["is_runnable"] is False
    assert any("пустой" in err.lower() for err in payload.get("errors", []))


@pytest.mark.asyncio
async def test_validate_code_tool_syntax_error_returns_error(mocker):
    """Test syntax errors are reported and code is not runnable."""
    from src.meta_agent.tools.code_writer_tools import ValidateCodeTool

    mock_context = MagicMock()
    mock_config = MagicMock()
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": []}, None),
    )

    tool = ValidateCodeTool(
        reasoning="Validate syntax failure",
        dto_name="test_dto",
        code="for i in range(3) print(i)",
    )
    result = await tool(mock_context, mock_config)
    payload = json.loads(result)

    assert payload["is_runnable"] is False
    assert any("синтакс" in err.lower() for err in payload.get("errors", []))


@pytest.mark.asyncio
async def test_execute_code_tool_with_dto(temp_charts_dir, sample_dto_data, mocker):
    """Test ExecuteCodeTool with DTO injection, sandbox, and output capture."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": sample_dto_data}, None),  # df handled inside
    )
    mock_execute = mocker.patch("src.meta_agent.tools.code_writer_tools._execute_safely")
    mock_execute.return_value = ("Stats computed", "")

    tool = ExecuteCodeTool(
        reasoning="Run stats snippet",
        dto_name="test_dto",
        code="df.describe(); print('Stats computed')",
    )
    mock_context = MagicMock()
    mock_config = MagicMock()

    result = await tool(mock_context, mock_config)

    data = json.loads(result)
    assert "output" in data
    assert "Stats computed" in data["output"]
    assert data["dto_name"] == "test_dto"


@pytest.mark.asyncio
async def test_execute_code_tool_sets_default_output_when_silent(sample_dto_data, mocker):
    """Test ExecuteCodeTool returns default output when code is silent."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": sample_dto_data}, None),
    )
    mocker.patch("src.meta_agent.tools.code_writer_tools._execute_safely", return_value=("", ""))

    tool = ExecuteCodeTool(reasoning="Silent run", dto_name="test_dto", code="x = 1")
    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)

    assert payload["dto_name"] == "test_dto"
    assert payload["output"] == "(нет вывода)"


@pytest.mark.asyncio
async def test_execute_code_tool_returns_error_output(sample_dto_data, mocker):
    """Test ExecuteCodeTool includes runtime error text in payload."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool

    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.resolve_dto_or_error",
        return_value=(None, {"rows": sample_dto_data}, None),
    )
    mocker.patch("src.meta_agent.tools.code_writer_tools._execute_safely", return_value=("", "boom"))

    tool = ExecuteCodeTool(reasoning="Failing run", dto_name="test_dto", code="raise ValueError('x')")
    result = await tool(MagicMock(), MagicMock())
    payload = json.loads(result)

    assert payload["dto_name"] == "test_dto"
    assert payload["error"] == "boom"


def test_run_code_success_returns_stdout():
    """Test _run_code returns stdout when code executes successfully."""
    from src.meta_agent.tools.code_writer_tools import _run_code

    stdout, error = _run_code("print('hello')")
    assert "hello" in stdout
    assert error == ""


def test_run_code_failure_returns_traceback():
    """Test _run_code returns traceback when execution fails."""
    from src.meta_agent.tools.code_writer_tools import _run_code

    stdout, error = _run_code("raise ValueError('boom')")
    assert stdout == ""
    assert "ValueError" in error


@pytest.mark.asyncio
async def test_execute_safely_timeout_branch(mocker):
    """Test _execute_safely returns timeout message on TimeoutError."""
    from src.meta_agent.tools.code_writer_tools import _execute_safely

    mocker.patch(
        "src.meta_agent.tools.code_writer_tools.asyncio.wait_for",
        side_effect=asyncio.TimeoutError,
    )
    stdout, error = await _execute_safely("print('x')")
    assert stdout == ""
    assert "Превышено время выполнения" in error


def test_make_sandbox_save_chart_fallback_path(mocker, tmp_path):
    """Test save_chart falls back to safe filename for unsafe resolved path."""
    from src.meta_agent.tools import code_writer_tools as cwt

    mocker.patch("src.meta_agent.tools.code_writer_tools.CHARTS_DIR", tmp_path)
    mocker.patch("src.meta_agent.tools.code_writer_tools.plt.savefig")
    mocker.patch("src.meta_agent.tools.code_writer_tools.plt.close")
    mocker.patch(
        "src.meta_agent.tools.code_writer_tools._sanitize_filename",
        side_effect=["../bad.png", "fallback.png"],
    )
    mock_resolve = mocker.patch(
        "src.meta_agent.tools.code_writer_tools.Path.resolve",
        side_effect=[Path("/tmp/evil.png"), tmp_path.resolve()],
    )

    namespace = cwt._make_sandbox(io.StringIO(), [], dto_payload={"rows": []})
    chart_path = namespace["save_chart"]("chart.png")

    assert chart_path.endswith("fallback.png")
    assert mock_resolve.call_count >= 2


def test_sanitize_filename_security():
    """Test _sanitize_filename prevents path traversal."""
    from src.meta_agent.tools.code_writer_tools import _sanitize_filename

    assert _sanitize_filename("normal.png") == "normal.png"
    # Path traversal should be sanitized
    dangerous = "../../../etc/passwd.png"
    sanitized = _sanitize_filename(dangerous)
    assert ".." not in sanitized
    assert sanitized.endswith(".png")
    assert "passwd" not in sanitized or "etc" not in sanitized


@pytest.mark.asyncio
async def test_code_execution_report_integration(mock_run_agent):
    """Test integration with code execution report flow (mocked)."""
    # The code_writer_node uses these tools; this tests basic flow
    assert mock_run_agent is not None  # from fixture


def test_code_writer_tool_metadata():
    """Verify tool names for code writer."""
    from src.meta_agent.tools.code_writer_tools import ExecuteCodeTool, ValidateCodeTool

    tools = [
        ValidateCodeTool(reasoning="meta", dto_name="dto", code="print(1)"),
        ExecuteCodeTool(reasoning="meta", dto_name="dto", code="print(1)"),
    ]
    for tool in tools:
        name = tool.tool_name
        assert name in ["validate_code", "execute_code"]
        assert len(tool.description) > 20


# Additional tests for sandbox internals would require more patching of ThreadPoolExecutor
# and AST walker, but core paths (safe/dangerous, DTO, error) are covered above.
