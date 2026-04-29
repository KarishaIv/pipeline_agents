"""Tests for CodeExecutionService - subprocess execution, timeout handling, and sandbox isolation."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from src.meta_agent.dto import DtoPayload
from src.meta_agent.services.code_execution import (
    CodeExecutionConfig,
    CodeExecutionService,
    ExecutionResult,
)


@pytest.fixture
def sample_dto():
    """Create sample DTO payload for testing."""
    return DtoPayload(
        summary_text="Test data",
        columns=["x", "y"],
        num_rows=3,
        sample=[],
        rows=[{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5, "y": 6}],
    )


def test_execution_result_dataclass():
    """Test ExecutionResult dataclass has correct attributes."""
    result = ExecutionResult(
        stdout="output",
        stderr="error",
        exit_code=0,
        timeout_occurred=False,
    )

    assert result.stdout == "output"
    assert result.stderr == "error"
    assert result.exit_code == 0
    assert result.timeout_occurred is False


def test_code_execution_config_defaults():
    """Test CodeExecutionConfig has correct defaults."""
    config = CodeExecutionConfig()

    assert config.timeout == 30
    assert config.max_stdout == 102400
    assert config.dto_payload is None
    assert config.charts_dir is None
    assert config.sandbox_globals is None


def test_code_execution_config_custom():
    """Test CodeExecutionConfig accepts custom values."""
    config = CodeExecutionConfig(
        timeout=60,
        max_stdout=50000,
    )

    assert config.timeout == 60
    assert config.max_stdout == 50000


def test_code_execution_service_init_valid():
    """Test CodeExecutionService initializes with valid config."""
    config = CodeExecutionConfig(timeout=30)
    service = CodeExecutionService(config)

    assert service.config == config


def test_code_execution_service_init_invalid_timeout():
    """Test CodeExecutionService raises on invalid timeout."""
    config = CodeExecutionConfig(timeout=0)

    with pytest.raises(ValueError, match="timeout must be positive"):
        CodeExecutionService(config)

    config2 = CodeExecutionConfig(timeout=-5)
    with pytest.raises(ValueError, match="timeout must be positive"):
        CodeExecutionService(config2)


def test_code_execution_service_init_invalid_max_stdout():
    """Test CodeExecutionService raises on invalid max_stdout."""
    config = CodeExecutionConfig(max_stdout=0)

    with pytest.raises(ValueError, match="max_stdout must be positive"):
        CodeExecutionService(config)


@pytest.mark.asyncio
async def test_execute_simple_code():
    """Test execute_async with simple valid code."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("print('hello')")

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.stderr == ""
    assert result.timeout_occurred is False


@pytest.mark.asyncio
async def test_execute_code_with_error():
    """Test execute_async captures runtime errors."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("raise ValueError('test error')")

    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert "test error" in result.stderr


@pytest.mark.asyncio
async def test_execute_code_with_math():
    """Test execute_async has access to math module."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("print(math.sqrt(16))")

    assert result.exit_code == 0
    assert "4.0" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_with_numpy():
    """Test execute_async has access to numpy."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("print(np.array([1, 2, 3]).sum())")

    assert result.exit_code == 0
    assert "6" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_with_pandas():
    """Test execute_async has access to pandas."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async(
        "df = pd.DataFrame({'a': [1, 2]}); print(df.shape[0])"
    )

    assert result.exit_code == 0
    assert "2" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_with_dto(sample_dto):
    """Test execute_async injects DTO into sandbox."""
    config = CodeExecutionConfig(dto_payload=sample_dto)
    service = CodeExecutionService(config)

    result = await service.execute_async(
        "print(len(df)); print(df['x'].sum())"
    )

    assert result.exit_code == 0
    assert "3" in result.stdout
    assert "9" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_dto_in_namespace(sample_dto):
    """Test execute_async provides dto dict in namespace."""
    config = CodeExecutionConfig(dto_payload=sample_dto)
    service = CodeExecutionService(config)

    result = await service.execute_async("print(dto['summary_text'])")

    assert result.exit_code == 0
    assert "Test data" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_silent(sample_dto):
    """Test execute_async with code that produces no output."""
    config = CodeExecutionConfig(dto_payload=sample_dto)
    service = CodeExecutionService(config)

    result = await service.execute_async("x = 1 + 1")

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_execute_code_multiline():
    """Test execute_async with multiline code."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    code = """
for i in range(3):
    print(i)
"""

    result = await service.execute_async(code)

    assert result.exit_code == 0
    assert "0" in result.stdout
    assert "1" in result.stdout
    assert "2" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_safe_builtins():
    """Test execute_async provides safe builtins."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("print(sum([1, 2, 3, 4]))")

    assert result.exit_code == 0
    assert "10" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_restricted_import():
    """Test execute_async code cannot import arbitrary modules."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("import os; print(os.system('ls'))")

    # Should fail because os is not imported; sandbox prevents __import__
    assert result.exit_code != 0
    assert "ImportError" in result.stderr or "NameError" in result.stderr


@pytest.mark.asyncio
async def test_execute_code_timeout():
    """Test execute_async times out on infinite loop."""
    config = CodeExecutionConfig(timeout=1)
    service = CodeExecutionService(config)

    result = await service.execute_async("while True: pass")

    assert result.timeout_occurred is True
    assert "timeout" in result.stderr.lower()
    assert result.stdout == ""


@pytest.mark.asyncio
async def test_execute_code_max_stdout_limit(sample_dto):
    """Test execute_async respects max_stdout limit."""
    config = CodeExecutionConfig(max_stdout=50)
    service = CodeExecutionService(config)

    result = await service.execute_async("print('x' * 1000)")

    assert len(result.stdout) <= 50


@pytest.mark.asyncio
async def test_execute_code_max_stderr_limit(sample_dto):
    """Test execute_async respects max_stdout limit for stderr too."""
    config = CodeExecutionConfig(max_stdout=50)
    service = CodeExecutionService(config)

    code = "raise Exception('error message that is very very very very very long')"
    result = await service.execute_async(code)

    assert len(result.stderr) <= 50


@pytest.mark.asyncio
async def test_execute_code_dto_serialization_error():
    """Test execute_async handles DTO serialization errors gracefully."""
    # Create DTO with a non-serializable field
    dto = DtoPayload(
        summary_text="Test",
        columns=[],
        num_rows=0,
        sample=[],
        rows=[],
    )

    # Mock model_dump to raise error (at class level to avoid Pydantic validation issues)
    with patch.object(DtoPayload, "model_dump", side_effect=ValueError("Serialization error")):
        config = CodeExecutionConfig(dto_payload=dto)
        service = CodeExecutionService(config)

        result = await service.execute_async("print('test')")

        assert result.exit_code == 1
        assert "Failed to serialize DTO" in result.stderr


@pytest.mark.asyncio
async def test_execute_sync_process_cleanup():
    """Test _execute_sync cleans up temporary script file."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("print('test')")

    assert result.exit_code == 0
    # Temp file should be cleaned up (we can't directly test this,
    # but it should not raise an error)


@pytest.mark.asyncio
async def test_execute_code_json_access():
    """Test execute_async provides json module."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async(
        "data = json.loads('{\"a\": 1}'); print(data['a'])"
    )

    assert result.exit_code == 0
    assert "1" in result.stdout


@pytest.mark.asyncio
async def test_execute_code_stats_access():
    """Test execute_async provides statistics module as stats."""
    config = CodeExecutionConfig()
    service = CodeExecutionService(config)

    result = await service.execute_async("print(stats.mean([1, 2, 3]))")

    assert result.exit_code == 0
    assert "2" in result.stdout
