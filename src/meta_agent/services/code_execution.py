"""Code execution service with subprocess isolation and reliable timeout handling.

Provides safe Python code execution in isolated subprocess with timeout management,
sandbox environment setup, and output capture.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.meta_agent.dto import DtoPayload


@dataclass
class ExecutionResult:
    """Result of code execution in subprocess.

    Attributes:
        stdout: Captured standard output from code execution.
        stderr: Captured standard error (includes traceback on exception).
        exit_code: Process exit code (0 for success).
        timeout_occurred: True if execution exceeded timeout limit.
    """

    stdout: str
    stderr: str
    exit_code: int
    timeout_occurred: bool


@dataclass
class CodeExecutionConfig:
    """Configuration for code execution service.

    Attributes:
        timeout: Execution timeout in seconds (default 30).
        max_stdout: Maximum stdout/stderr capture size in bytes (default 100KB).
        dto_payload: Optional DTO payload to inject into sandbox.
        charts_dir: Directory for saving matplotlib figures.
        sandbox_globals: Optional override of sandbox global namespace.
    """

    timeout: int = 30
    max_stdout: int = 102400
    dto_payload: Optional[DtoPayload] = None
    charts_dir: Optional[Path] = None
    sandbox_globals: Optional[dict[str, Any]] = None


class CodeExecutionService:
    """Service for securely executing Python code in isolated subprocess.

    Handles:
    - Subprocess creation with resource isolation
    - Reliable timeout handling with process cleanup
    - Sandbox environment with safe builtins
    - DTO data injection via environment variables
    - Output capture with size limits
    - Chart file management
    """

    DTO_ENV_VAR = "DTO_DATA_JSON"

    def __init__(self, config: CodeExecutionConfig):
        """Initialize CodeExecutionService with configuration.

        Args:
            config: CodeExecutionConfig with timeout, DTO, and other settings.

        Raises:
            ValueError: If configuration is invalid.
        """
        if config.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {config.timeout}")
        if config.max_stdout <= 0:
            raise ValueError(f"max_stdout must be positive, got {config.max_stdout}")

        self.config = config

    def _make_sandbox_script(self) -> str:
        """Generate Python script to run in subprocess with sandbox environment.

        Returns:
            Python code that sets up the sandbox and executes user code.
        """
        # Prepare DTO data if provided
        dto_import = ""
        dto_setup = ""
        if self.config.dto_payload is not None:
            dto_import = "import json"
            dto_setup = """
import os
dto_json = os.environ.get('DTO_DATA_JSON', '{}')
try:
    dto = json.loads(dto_json)
except json.JSONDecodeError:
    dto = {}
"""

        sandbox_script = f"""
import sys
import io
import json
import math
import statistics as stats
import traceback
{dto_import}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Capture stdout
_stdout_buf = io.StringIO()

# Safe builtins (minimal set for performance)
_safe_builtins = {{
    'print': lambda *a, **kw: print(*a, file=_stdout_buf, **kw),
    'range': range,
    'len': len,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'sorted': sorted,
    'enumerate': enumerate,
    'zip': zip,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'type': type,
    'isinstance': isinstance,
    'hasattr': hasattr,
    'getattr': getattr,
    'repr': repr,
    'format': format,
    'map': map,
    'filter': filter,
    'any': any,
    'all': all,
    'iter': iter,
    'next': next,
    'reversed': reversed,
    'vars': vars,
    'Exception': Exception,
    'ValueError': ValueError,
    'KeyError': KeyError,
    'TypeError': TypeError,
}}

# Setup DTO if provided
{dto_setup}

# Create DataFrame if DTO exists
try:
    if 'dto' in locals() and isinstance(dto, dict) and 'rows' in dto:
        df = pd.DataFrame(dto['rows'])
    else:
        df = pd.DataFrame()
except Exception:
    df = pd.DataFrame()

# Sandbox namespace
_namespace = {{
    '__builtins__': _safe_builtins,
    'np': np,
    'pd': pd,
    'plt': plt,
    'math': math,
    'json': json,
    'stats': stats,
}}

if 'dto' in locals():
    _namespace['dto'] = dto
if 'df' in locals():
    _namespace['df'] = df

# Execute user code
try:
    exec(sys.argv[1], _namespace)
    # Write captured output to stdout
    print(_stdout_buf.getvalue(), end='')
except Exception:
    # On exception: flush stdout and write error to stderr
    sys.stdout.write(_stdout_buf.getvalue())
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""

        return sandbox_script

    async def execute_async(self, code: str) -> ExecutionResult:
        """Execute Python code asynchronously in subprocess with timeout.

        Args:
            code: Python code string to execute.

        Returns:
            ExecutionResult with stdout, stderr, exit_code, and timeout flag.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_sync, code)

    def _execute_sync(self, code: str) -> ExecutionResult:
        """Execute Python code synchronously in subprocess.

        Manages process lifecycle, timeout handling, and output capture.
        User code stdout/stderr are captured directly without JSON wrapping.

        Args:
            code: Python code string to execute.

        Returns:
            ExecutionResult with execution results. Exit code 0 indicates success,
            non-zero indicates exception occurred (see stderr for traceback).
        """
        # Prepare environment with DTO data if provided
        env = os.environ.copy()
        if self.config.dto_payload is not None:
            try:
                dto_json = json.dumps(
                    self.config.dto_payload.model_dump(),
                    ensure_ascii=False,
                    default=str,
                )
                env[self.DTO_ENV_VAR] = dto_json
            except (json.JSONDecodeError, ValueError) as e:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Failed to serialize DTO payload: {e}",
                    exit_code=1,
                    timeout_occurred=False,
                )

        # Create script with sandbox
        sandbox_script = self._make_sandbox_script()

        # Create temporary file for the script to avoid command-line length limits
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as tmp_script:
            tmp_script.write(sandbox_script)
            script_path = tmp_script.name

        try:
            # Start subprocess
            process = subprocess.Popen(
                [sys.executable, script_path, code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )

            # Wait for process with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout)
            except subprocess.TimeoutExpired:
                # Terminate gracefully, then kill if needed
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timeout ({self.config.timeout} seconds exceeded)",
                    exit_code=process.returncode or -1,
                    timeout_occurred=True,
                )

            # Limit output size and strip whitespace
            exit_code = process.returncode or 0
            stdout = stdout[: self.config.max_stdout].strip()
            stderr = stderr[: self.config.max_stdout].strip()

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timeout_occurred=False,
            )

        finally:
            # Clean up temporary script
            try:
                Path(script_path).unlink()
            except OSError:
                pass

