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
        dto_payloads: Optional dict of DTO name -> payload to inject into sandbox.
        charts_dir: Directory for saving matplotlib figures.
        sandbox_globals: Optional override of sandbox global namespace.
    """

    timeout: int = 30
    max_stdout: int = 102400
    dto_payloads: Optional[dict[str, DtoPayload]] = None
    charts_dir: Optional[Path] = None
    sandbox_globals: Optional[dict[str, Any]] = None


class CodeExecutionService:
    """Service for securely executing Python code in isolated subprocess.

    Handles:
    - Subprocess creation with resource isolation
    - Reliable timeout handling with process cleanup
    - Sandbox environment with safe builtins
    - Multiple DTOs injection via DTOS_DATA_JSON env var
    - Output capture with size limits
    - Chart file management
    """

    DTOS_ENV_VAR = "DTOS_DATA_JSON"

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
        # Prepare DTO data if provided (multiple DTOs)
        dto_import = ""
        dto_setup = ""
        if self.config.dto_payloads is not None and len(self.config.dto_payloads) > 0:
            dto_import = "import json"
            dto_setup = """
import os
dtos_json = os.environ.get('DTOS_DATA_JSON', '{}')
try:
    dtos = json.loads(dtos_json)
except json.JSONDecodeError:
    dtos = {}
"""

        # Prepare save_chart + raw data save functions (always available, with fallback to temp dir if needed)
        save_chart_func = f"""
import os as _os
from pathlib import Path as _Path
from uuid import uuid4 as _uuid4
from datetime import datetime as _datetime
import tempfile as _tempfile
import json as _json

# Use provided charts_dir or fallback to temp directory
_charts_dir = _Path(r'{str(self.config.charts_dir)}') if {self.config.charts_dir is not None} else _Path(_tempfile.gettempdir()) / 'agent_charts'
_charts_dir.mkdir(parents=True, exist_ok=True)

def save_chart(filename=None):
    \"\"\"Save current matplotlib figure to disk in the charts directory.

    Args:
        filename: Optional filename. If None, auto-generates timestamp-based name.

    Returns:
        Path to the saved chart file.
    \"\"\"
    if filename is None:
        filename = f"chart_{{_datetime.now().strftime('%Y%m%d_%H%M%S_%f')}}.png"

    # Sanitize filename to prevent path traversal
    import re as _re
    safe_name = _re.sub(r'[^\\w\\.-]', '_', filename.strip())
    safe_name = _re.sub(r'_+', '_', safe_name)
    if '..' in safe_name or '/' in safe_name or '\\\\' in safe_name:
        safe_name = f"chart_{{_datetime.now().strftime('%Y%m%d_%H%M%S')}}.png"
    if not safe_name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        safe_name += '.png'

    # Append short unique suffix to every saved chart file (prevents collisions on repeated calls)
    _unique = _uuid4().hex[:8]
    if '.' in safe_name:
        _base, _ext = safe_name.rsplit('.', 1)
        safe_name = f"{{_base}}_{{_unique}}.{{_ext}}"
    else:
        safe_name = f"{{safe_name}}_{{_unique}}"

    target_path = _charts_dir / safe_name
    plt.savefig(str(target_path), bbox_inches='tight', dpi=150)
    plt.close()
    return str(target_path)


def save_json(data, filename=None):
    \"\"\"Save raw data (list, dict or DataFrame) as JSON artifact. Only raw source data - no aggregations.

    Args:
        data: Raw data to persist (list of records, dict, or pandas DataFrame).
        filename: Optional filename. If None, auto-generates timestamp-based name.

    Returns:
        Path to the saved JSON file.
    \"\"\"
    if filename is None:
        filename = f"data_{{_datetime.now().strftime('%Y%m%d_%H%M%S_%f')}}.json"

    # Sanitize filename
    import re as _re
    safe_name = _re.sub(r'[^\\w\\.-]', '_', filename.strip())
    safe_name = _re.sub(r'_+', '_', safe_name)
    if '..' in safe_name or '/' in safe_name or '\\\\' in safe_name:
        safe_name = f"data_{{_datetime.now().strftime('%Y%m%d_%H%M%S')}}.json"
    if not safe_name.lower().endswith('.json'):
        safe_name += '.json'

    # Append short unique suffix to every saved JSON file (prevents collisions)
    _unique = _uuid4().hex[:8]
    if '.' in safe_name:
        _base, _ext = safe_name.rsplit('.', 1)
        safe_name = f"{{_base}}_{{_unique}}.{{_ext}}"
    else:
        safe_name = f"{{safe_name}}_{{_unique}}"

    # Convert DataFrame to records if needed, keep raw
    if hasattr(data, 'to_dict'):
        records = data.to_dict(orient='records')
    else:
        records = data

    target_path = _charts_dir / safe_name
    with open(target_path, 'w', encoding='utf-8') as f:
        _json.dump(records, f, ensure_ascii=False, indent=2)
    return str(target_path)


def save_csv(data, filename=None):
    \"\"\"Save raw data (DataFrame or list of dicts) as CSV artifact. Only raw source data - no aggregations or stats.

    Args:
        data: Raw data (pandas DataFrame or list of records).
        filename: Optional filename. If None, auto-generates timestamp-based name.

    Returns:
        Path to the saved CSV file.
    \"\"\"
    if filename is None:
        filename = f"data_{{_datetime.now().strftime('%Y%m%d_%H%M%S_%f')}}.csv"

    # Sanitize filename
    import re as _re
    safe_name = _re.sub(r'[^\\w\\.-]', '_', filename.strip())
    safe_name = _re.sub(r'_+', '_', safe_name)
    if '..' in safe_name or '/' in safe_name or '\\\\' in safe_name:
        safe_name = f"data_{{_datetime.now().strftime('%Y%m%d_%H%M%S')}}.csv"
    if not safe_name.lower().endswith('.csv'):
        safe_name += '.csv'

    # Append short unique suffix to every saved CSV file (prevents collisions)
    _unique = _uuid4().hex[:8]
    if '.' in safe_name:
        _base, _ext = safe_name.rsplit('.', 1)
        safe_name = f"{{_base}}_{{_unique}}.{{_ext}}"
    else:
        safe_name = f"{{safe_name}}_{{_unique}}"

    target_path = _charts_dir / safe_name
    if hasattr(data, 'to_csv'):
        data.to_csv(target_path, index=False)
    else:
        import pandas as _pd
        _pd.DataFrame(data).to_csv(target_path, index=False)
    return str(target_path)
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

# Restricted import hook to prevent access to dangerous modules
_SAFE_MODULES = {{'matplotlib', 'numpy', 'pandas', 'json', 'math', 'statistics', 're'}}
_builtin_import = __import__

def _restricted_import(name, *args, **kwargs):
    # Allow standard library modules that are already imported at the top level
    # and explicitly whitelisted safe modules
    if name not in _SAFE_MODULES and not name.startswith(('src.', '__')):
        # Check if it's a submodule of a safe module
        base_module = name.split('.')[0]
        if base_module not in _SAFE_MODULES:
            raise ImportError(f"Import of {{name}} is not allowed in this sandbox")
    return _builtin_import(name, *args, **kwargs)

# Safe builtins (minimal set for performance)
_safe_builtins = {{
    '__import__': _restricted_import,
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

# Create DataFrames from DTOs
try:
    if 'dtos' in locals() and isinstance(dtos, dict):
        dfs = {{}}
        for name, d in dtos.items():
            if isinstance(d, dict) and 'rows' in d:
                dfs[name] = pd.DataFrame(d['rows'])
    else:
        dfs = {{}}
except Exception:
    dfs = {{}}

# Setup save_chart function if charts directory is available
{save_chart_func}

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

if 'save_chart' in locals():
    _namespace['save_chart'] = save_chart
if 'save_json' in locals():
    _namespace['save_json'] = save_json
if 'save_csv' in locals():
    _namespace['save_csv'] = save_csv

if 'dtos' in locals():
    _namespace['dtos'] = dtos
if 'dfs' in locals():
    _namespace['dfs'] = dfs

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
        if self.config.dto_payloads is not None and len(self.config.dto_payloads) > 0:
            try:
                dtos_dict = {
                    name: payload.model_dump() for name, payload in self.config.dto_payloads.items()
                }
                dtos_json = json.dumps(dtos_dict, ensure_ascii=False, default=str)
                env[self.DTOS_ENV_VAR] = dtos_json
            except Exception as e:
                return ExecutionResult(
                    stdout="",
                    stderr=f"Failed to serialize DTO payloads: {e}",
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
