"""Инструменты агента code_writer: валидация и безопасное выполнение Python-кода.

Tools for code validation and execution. Delegates execution logic to CodeExecutionService
and chart management to ChartService for separation of concerns.
"""

from __future__ import annotations

import ast
from typing import Any, TYPE_CHECKING

from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from src.meta_agent.configs import CHARTS_DIR, CODE_TIMEOUT
from src.meta_agent.services import CodeExecutionConfig, CodeExecutionService
from src.meta_agent.tools.dto_tools import resolve_dto_or_error
from src.meta_agent.utils.json_responses import serialize_tool_result

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext
    from sgr_agent_core.agent_definition import AgentConfig


class ExecuteCodeTool(BaseTool):
    """Безопасное выполнение Python-кода для code_writer.

    Delegates code execution to CodeExecutionService which handles subprocess
    isolation, timeout management, and output capture.
    """

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

        # Create executor service with configuration
        executor = CodeExecutionService(
            config=CodeExecutionConfig(
                timeout=CODE_TIMEOUT,
                dto_payload=dto_payload,
                charts_dir=CHARTS_DIR,
            )
        )

        # Execute code asynchronously
        result = await executor.execute_async(self.code)

        # Format response
        output = result.stdout or "(нет вывода)"
        response: dict[str, Any] = {
            "dto_name": self.dto_name,
            "output": output,
        }

        if result.stderr:
            response["error"] = result.stderr

        return serialize_tool_result(response)


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
            return serialize_tool_result(diagnostics)

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
            return serialize_tool_result(diagnostics)
        except Exception as exc:
            diagnostics["errors"].append(f"Ошибка компиляции: {exc}")
            diagnostics["syntax_ok"] = False
            return serialize_tool_result(diagnostics)

        diagnostics["is_runnable"] = True
        return serialize_tool_result(diagnostics)
