"""Классы SystemBaseTool со структурированным выводом — по одному на роль агента.

Каждый инструмент завершает цикл агента, выставляя context.state и
context.execution_result, после чего возвращает валидированный JSON-пейлоад.
"""

from __future__ import annotations

from typing import List, Literal, TYPE_CHECKING

from pydantic import Field

from sgr_agent_core.base_tool import SystemBaseTool
from sgr_agent_core.models import AgentStatesEnum

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext


class SupervisorDecisionTool(SystemBaseTool):
    """Зафиксировать решение супервайзера о маршрутизации и завершить его шаг."""

    tool_name = "supervisor_decision"
    description = (
        "Зафиксировать решение супервайзера о следующем шаге (какого агента вызвать "
        "или завершить) и завершить текущий шаг супервайзера."
    )

    reasoning: str = Field(
        default="",
        description="Анализ текущего состояния и обоснование решения"
    )
    next: Literal["data_extractor", "analyzer", "end"] = Field(
        description=(
            "Какой исполнитель следующий: "
            "data_extractor — получить или уточнить данные из Qdrant; "
            "analyzer — проанализировать уже собранное; "
            "end — вопрос пользователя полностью закрыт, можно отвечать"
        )
    )
    task: str = Field(
        default="",
        description=(
            "Краткая формулировка задачи для выбранного агента: ЧТО сделать, не КАК. "
            "Пустая строка, если next=end."
        )
    )
    final_answer: str = Field(
        default="",
        description="Полный ответ пользователю на русском — только при next=end",
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = AgentStatesEnum.COMPLETED
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload


class DataExtractionReportTool(SystemBaseTool):
    """Структурированный отчёт после извлечения данных из Qdrant; завершает шаг извлекателя."""

    tool_name = "data_extraction_report"
    description = (
        "Передать структурированные результаты извлечения данных из Qdrant "
        "и завершить шаг агента-извлекателя."
    )

    reasoning: str = Field(
        default="",
        description="Что искали, в каких коллекциях и зачем"
    )
    completed_steps: List[str] = Field(
        default_factory=list,
        description="Упорядоченный список выполненных шагов извлечения (минимум 1 при успехе)",
        min_length=0,
        max_length=20,
    )
    summary: str = Field(
        default="",
        description="Краткий итог извлечения (DTO-имена, назначение, важные поля и объёмы) в виде JSON-строки"
    )
    dto_references: str = Field(
        default="",
        description="Ссылки на DTO и служебные метаданные (без полного дампа rows) в виде JSON-строки"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        default=AgentStatesEnum.COMPLETED,
        description="Статус извлечения: успех или сбой"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload



class CodeExecutionReportTool(SystemBaseTool):
    """Структурированный отчёт code_writer после написания и проверки кода."""

    tool_name = "code_execution_report"
    description = (
        "Передать результаты code_writer: код, валидацию, выполнение и выводы; "
        "завершить шаг code_writer."
    )

    reasoning: str = Field(
        default="",
        description="Подход к решению и ключевые решения по коду"
    )
    task: str = Field(
        default="",
        description="Исходная задача для code_writer"
    )
    code: str = Field(
        default="",
        description="Итоговый исполняемый код"
    )
    validation: str = Field(
        default="",
        description="JSON-строка результата validate_code"
    )
    execution: str = Field(
        default="",
        description="JSON-строка результата execute_code"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Ключевые наблюдения из выполнения кода"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        default=AgentStatesEnum.COMPLETED,
        description="Статус выполнения code_writer: успех или сбой"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload


class AnalyzerDecisionTool(SystemBaseTool):
    """Unified decision tool for analyzer_node.
    LLM calls this single tool to decide report vs delegate.
    """

    tool_name = "analyzer_decision"
    description = (
        "Завершить шаг аналитика: сформулировать итоговые выводы (decision='report'), "
        "либо делегировать задачу code_writer (decision='delegate')."
    )

    reasoning: str = Field(
        default="",
        description="Краткий анализ текущей ситуации, данных и выбранного пути (ОБЯЗАТЕЛЬНО заполнять)"
    )
    decision: Literal["report", "delegate"] = Field(
        description="Тип решения: 'report' — завершить анализ с выводами; 'delegate' — передать code_writer"
    )
    # Fields for report
    key_findings: List[str] = Field(
        default_factory=list,
        description="Ключевые находки и закономерности (используется при decision='report')"
    )
    conclusions: str = Field(
        default="",
        description="Развёрнутые выводы на русском (используется при decision='report')"
    )
    # Fields for delegate
    task: str = Field(
        default="",
        description="Конкретная задача для code_writer (используется при decision='delegate')"
    )
    delegate_reason: str = Field(
        default="",
        description="Обоснование почему нужен code_writer (используется при decision='delegate')"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        default=AgentStatesEnum.COMPLETED,
        description="Статус шага анализа"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload
