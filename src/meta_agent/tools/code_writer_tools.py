"""Инструменты агента code_writer: статическая проверка и выполнение Python-кода.

Выполнение делегируется CodeExecutionService, который запускает код в отдельном
процессе, ограничивает время работы и собирает stdout/stderr.
"""

from __future__ import annotations

import ast
import logging
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

logger = logging.getLogger("meta_agent.code_writer")


class ExecuteCodeTool(BaseTool):
    """Выполнить Python-код для code_writer в сервисе исполнения.

    CodeExecutionService отвечает за отдельный процесс, timeout, передачу DTO
    и сбор вывода. PNG-графики, созданные через save_chart(), регистрируются
    как артефакты по недавно изменённым файлам в CHARTS_DIR.
    """

    tool_name = "execute_code"
    description = (
        "Выполнить Python-код для анализа выбранного DTO через CodeExecutionService. "
        "В коде доступны dto, df, np, pd, plt, math, json, stats и save_chart(). "
        "stdout возвращается в output; stderr возвращается в error. PNG, созданные через save_chart(), "
        "регистрируются как артефакты."
    )

    reasoning: str = Field(description="Что вычисляет код и зачем")
    dto_name: str = Field(description="Имя DTO, с которым нужно работать в коде")
    code: str = Field(
        description=(
            "Код Python для выполнения. Уже импортированы: np, pd, plt, math, json, stats. "
            "Также доступны dto (dict DTO) и df (DataFrame из DTO rows). "
            "Для сохранения графика используй save_chart('file.png'). Вывод — через print()."
        )
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        _, dto_payload, error = resolve_dto_or_error(context, self.dto_name)
        if error:
            return error
        assert dto_payload is not None

        executor = CodeExecutionService(
            config=CodeExecutionConfig(
                timeout=CODE_TIMEOUT,
                dto_payload=dto_payload,
                charts_dir=CHARTS_DIR,
            )
        )

        result = await executor.execute_async(self.code)

        output = result.stdout or "(нет вывода)"
        response: dict[str, Any] = {
            "dto_name": self.dto_name,
            "output": output,
        }

        if result.stderr:
            response["error"] = result.stderr

        from src.meta_agent.output_models import AgentArtifact
        from uuid import uuid4
        from pathlib import Path

        if not hasattr(context, 'custom_context'):
            context.custom_context = {}
        if 'artifacts' not in context.custom_context:
            context.custom_context['artifacts'] = []

        # save_chart() сохраняет PNG в CHARTS_DIR; регистрируем недавно созданные файлы.
        if "save_chart" in self.code and CHARTS_DIR.exists():
            try:
                import time as time_module
                current_time = time_module.time()
                for chart_file in CHARTS_DIR.glob("*.png"):
                    if current_time - chart_file.stat().st_mtime < 10:
                        artifact = AgentArtifact(
                            id=str(uuid4()),
                            kind="chart",
                            path=str(chart_file),
                            filename=chart_file.name,
                            mime_type="image/png",
                            caption=f"Chart: {chart_file.name}",
                            metadata={"source": "code_execution"},
                        )
                        context.custom_context['artifacts'].append(artifact)
            except Exception as e:
                logger.warning("Failed to register chart artifacts from code execution: %s", e)

        return serialize_tool_result(response)


class ValidateCodeTool(BaseTool):
    """Проверить, что Python-код компилируется и проходит статические проверки."""

    tool_name = "validate_code"
    description = (
        "Проверить корректность Python-кода без выполнения. "
        "Инструмент выполняет только синтаксические и базовые статические проверки. "
        "Возвращает is_runnable=True при успешной компиляции, даже если найдены предупреждения."
    )

    reasoning: str = Field(description="Зачем нужна статическая проверка и какие риски ожидаются")
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
