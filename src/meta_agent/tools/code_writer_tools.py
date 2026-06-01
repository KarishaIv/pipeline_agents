"""Инструменты агента code_writer: статическая проверка и выполнение Python-кода.

Выполнение делегируется CodeExecutionService, который запускает код в отдельном
процессе, ограничивает время работы и собирает stdout/stderr.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from src.meta_agent.configs import CHARTS_DIR, CODE_TIMEOUT
from src.meta_agent.output_models import AgentArtifact
from src.meta_agent.services import CodeExecutionConfig, CodeExecutionService
from src.meta_agent.services.artifact import ArtifactService
from src.meta_agent.tools.dto_tools import resolve_dto_or_error
from src.meta_agent.utils.json_responses import serialize_tool_result

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext
    from sgr_agent_core.agent_definition import AgentConfig

logger = logging.getLogger("meta_agent.code_writer")


class ExecuteCodeTool(BaseTool):
    """Выполнить Python-код для code_writer в сервисе исполнения.

    CodeExecutionService отвечает за отдельный процесс, timeout, передачу DTO
    и сбор вывода. Файлы (PNG/JSON/CSV), созданные через save_*(), регистрируются
    как артефакты по недавно изменённым файлам в CHARTS_DIR. JSON/CSV — только raw source data.
    """

    tool_name = "execute_code"
    description = (
        "Выполнить Python-код для анализа DTO (можно несколько или без них, если они не нужны). "
        "В коде доступны dtos (dict по именам), dfs (dataframes), np, pd, plt, math, json, stats, save_chart(), save_json() и save_csv(). "
        "stdout возвращается в output; stderr возвращается в error. Файлы (PNG/JSON/CSV), созданные через save_*(), "
        "регистрируются как артефакты (JSON/CSV содержат только raw source data)."
    )

    reasoning: str = Field(description="Что вычисляет код и зачем")
    dto_names: list[str] = Field(default_factory=list, description="Список имён DTO для использования в коде (можно несколько для кросс-DTO анализа)")
    code: str = Field(
        description=(
            "Код Python для выполнения. Уже импортированы: np, pd, plt, math, json, stats. "
            "Также доступны dtos (dict name->payload), dfs (dataframes). "
            "Для сохранения графика используй save_chart('file.png'). Вывод — через print()."
        )
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        dto_payloads: dict[str, Any] = {}
        errors = []
        for name in self.dto_names or []:
            _, payload, error = resolve_dto_or_error(context, name)
            if error:
                try:
                    errors.append(json.loads(error).get("error", error))
                except json.JSONDecodeError:
                    errors.append(error)
            elif payload is not None:
                dto_payloads[name] = payload
        if errors:
            return serialize_tool_result({
                "dto_names": self.dto_names,
                "output": "(нет вывода)",
                "error": "\n".join(errors),
            })

        artifact_service = ArtifactService(CHARTS_DIR)

        # Snapshot existing artifact files before execution so we can detect only newly created ones
        existing_files: set[Path] = {
            path for path in artifact_service.artifacts_dir.iterdir() if path.is_file()
        }

        executor = CodeExecutionService(
            config=CodeExecutionConfig(
                timeout=CODE_TIMEOUT,
                dto_payloads=dto_payloads,
                artifacts_dir=CHARTS_DIR,
            )
        )

        result = await executor.execute_async(self.code)

        output = result.stdout or "(нет вывода)"
        response: dict[str, Any] = {
            "dto_names": self.dto_names,
            "output": output,
        }

        if result.stderr:
            response["error"] = result.stderr

        if not hasattr(context, 'custom_context'):
            context.custom_context = {}
        if 'artifacts' not in context.custom_context:
            context.custom_context['artifacts'] = []

        # Register only newly created artifacts (charts, JSON/CSV raw data sources)
        try:
            for artifact_path in artifact_service.artifacts_dir.iterdir():
                if not artifact_path.is_file() or artifact_path in existing_files:
                    continue
                metadata = artifact_service.artifact_from_existing_file(
                    artifact_path,
                    metadata={"source": "code_execution"},
                )
                metadata["caption"] = metadata["caption"] or f"{metadata['kind'].upper()}: {metadata['filename']}"
                context.custom_context['artifacts'].append(AgentArtifact(**metadata))
        except Exception as e:
            logger.warning("Failed to register artifacts from code execution: %s", e)

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
    dto_names: list[str] = Field(default_factory=list, description="Список имён DTO для валидации кода")
    code: str = Field(description="Код Python только для статической проверки")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        errors = []
        for name in self.dto_names or []:
            _, payload, error = resolve_dto_or_error(context, name)
            if error:
                errors.append(error)
        if errors:
            return "\n".join(errors)

        diagnostics: dict[str, Any] = {
            "dto_names": self.dto_names,
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
