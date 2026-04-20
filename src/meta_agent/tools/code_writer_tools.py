"""Инструменты агента code_writer: валидация и безопасное выполнение Python-кода."""

from __future__ import annotations

import ast
import asyncio
import io
import json
import math
import os
import re
import statistics as _stats
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from src.meta_agent.config import CHARTS_DIR, CODE_TIMEOUT
from src.meta_agent.tools.dto_tools import dto_to_dataframe, resolve_dto_or_error

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext
    from sgr_agent_core.agent_definition import AgentConfig

DTO_ENV_VAR = "DTO_DATA_JSON"

_SAFE_BUILTINS: dict = {
    "print": print,
    "range": range,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "repr": repr,
    "format": format,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "iter": iter,
    "next": next,
    "reversed": reversed,
    "vars": vars,
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
}


def _sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent path traversal and injection attacks.
    Removes dangerous characters, prevents .. / \\, falls back to safe default.
    Part of security hardening for code execution sandbox.
    """
    if not name:
        name = "chart.png"
    # Remove or replace dangerous chars
    sanitized = re.sub(r"[^\w\.-]", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized)
    # Prevent path traversal
    if ".." in sanitized or "/" in sanitized or "\\" in sanitized or sanitized.startswith("."):
        sanitized = f"safe_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    # Ensure extension
    if not sanitized.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
        sanitized += ".png"
    return sanitized


def _make_sandbox(stdout_buf: io.StringIO, saved_charts: list, dto_payload: dict[str, Any] | None = None) -> dict:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save_chart(filename: str | None = None) -> str:
        name = filename or f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        safe_name = _sanitize_filename(name)
        path = str(CHARTS_DIR / safe_name)
        # Additional resolve check for security
        resolved_path = Path(path).resolve()
        if not str(CHARTS_DIR.resolve()) in str(resolved_path):
            safe_name = _sanitize_filename("fallback.png")
            path = str(CHARTS_DIR / safe_name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        saved_charts.append(path)
        return path

    builtins_patched = {
        **_SAFE_BUILTINS,
        "print": lambda *a, **kw: print(*a, **kw, file=stdout_buf),
    }

    namespace = {
        "__builtins__": builtins_patched,
        "np": np,
        "pd": pd,
        "plt": plt,
        "math": math,
        "json": json,
        "stats": _stats,
        "save_chart": _save_chart,
    }

    if dto_payload is not None:
        namespace["dto"] = dto_payload
        namespace["df"] = dto_to_dataframe(dto_payload)

    return namespace


def _run_code(code: str, dto_payload: dict[str, Any] | None = None) -> tuple[str, str]:
    stdout_buf = io.StringIO()
    saved_charts: list = []
    namespace = _make_sandbox(stdout_buf, saved_charts, dto_payload=dto_payload)
    try:
        exec(compile(code, "<code_writer>", "exec"), namespace)  # noqa: S102
        output = stdout_buf.getvalue()
        if saved_charts:
            output += f"\nСохранённые графики: {', '.join(saved_charts)}"
        return output.strip(), ""
    except Exception:
        return stdout_buf.getvalue().strip(), traceback.format_exc()


async def _execute_safely(code: str, dto_payload: dict[str, Any] | None = None) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, _run_code, code, dto_payload),
            timeout=CODE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return "", f"Превышено время выполнения ({CODE_TIMEOUT} с)"
    finally:
        executor.shutdown(wait=False)


class ExecuteCodeTool(BaseTool):
    """Безопасное выполнение Python-кода для code_writer."""

    tool_name = "execute_code"
    description = (
        "Написать и безопасно выполнить код на Python для анализа. "
        "В песочнице доступны: np (numpy), pd (pandas), plt (matplotlib), math, json, stats, save_chart(). "
        "Нельзя читать/писать файлы и подключать внешние модули."
    )

    reasoning: str = Field(description="Что вычисляет код и зачем")
    dto_name: str = Field(description="Имя DTO, с которым нужно работать в коде")
    code: str = Field(
        description=(
            "Корректный код на Python. Уже импортированы: np, pd, plt, math, json, stats. "
            "Также доступны dto (dict DTO) и df (DataFrame из DTO rows). "
            "Для сохранения графика используй save_chart('file.png'). Вывод — через print()."
        )
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        _, dto_payload, error = resolve_dto_or_error(context, self.dto_name)
        if error:
            return error
        assert dto_payload is not None

        previous_env = os.environ.get(DTO_ENV_VAR)
        os.environ[DTO_ENV_VAR] = json.dumps(dto_payload, ensure_ascii=False, default=str)
        try:
            stdout, error_text = await _execute_safely(self.code, dto_payload=dto_payload)
        finally:
            if previous_env is None:
                os.environ.pop(DTO_ENV_VAR, None)
            else:
                os.environ[DTO_ENV_VAR] = previous_env

        result: dict[str, Any] = {"dto_name": self.dto_name}
        if stdout:
            result["output"] = stdout
        if error_text:
            result["error"] = error_text
        if "output" not in result and "error" not in result:
            result["output"] = "(нет вывода)"
        return json.dumps(result, ensure_ascii=False)


class ValidateCodeTool(BaseTool):
    """Проверить, что Python-код компилируется и проходит статические проверки."""

    tool_name = "validate_code"
    description = (
        "Проверить корректность Python-кода без выполнения в песочнице. "
        "Инструмент выполняет только синтаксические и базовые статические проверки. "
        "Возвращает результат с is_runnable, ошибками и предупреждениями."
    )

    reasoning: str = Field(description="Зачем нужна проверка и на какие риски смотреть")
    dto_name: str = Field(description="Имя DTO, с которым валидируется код")
    code: str = Field(description="Код Python только для статической проверки")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        _, dto_payload, error = resolve_dto_or_error(context, self.dto_name)
        if error:
            return error
        assert dto_payload is not None

        diagnostics: dict[str, Any] = {
            "dto_name": self.dto_name,
            "is_runnable": False,
            "errors": [],
            "warnings": [],
        }

        stripped = self.code.strip()
        if not stripped:
            diagnostics["errors"].append("Код пустой")
            return json.dumps(diagnostics, ensure_ascii=False)

        try:
            tree = ast.parse(stripped, mode="exec")
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    diagnostics["warnings"].append("Обнаружен импорт. В песочнице запрещены произвольные импорты.")
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in {"open", "exec", "eval", "__import__"}:
                        diagnostics["warnings"].append(f"Обнаружен вызов {node.func.id}(). Доступ к этим функциям в песочнице запрещен.")
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in {"open", "exec", "eval", "__import__"}:
                        diagnostics["warnings"].append(f"Обнаружен вызов атрибута {node.func.attr}(). Доступ к этим функциям в песочнице запрещен.")

            compile(tree, "<code_writer_validate>", "exec")
            diagnostics["syntax_ok"] = True
        except SyntaxError as exc:
            diagnostics["errors"].append(
                f"Синтаксическая ошибка: {exc.msg} (line={exc.lineno}, offset={exc.offset})"
            )
            diagnostics["syntax_ok"] = False
            return json.dumps(diagnostics, ensure_ascii=False)
        except Exception as exc:
            diagnostics["errors"].append(f"Ошибка компиляции: {exc}")
            diagnostics["syntax_ok"] = False
            return json.dumps(diagnostics, ensure_ascii=False)

        diagnostics["is_runnable"] = True
        return json.dumps(diagnostics, ensure_ascii=False)
