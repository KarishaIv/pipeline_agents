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

    reasoning: str = Field(description="Что искали, в каких коллекциях и зачем")
    completed_steps: List[str] = Field(
        description="Упорядоченный список выполненных шагов извлечения",
        min_length=1,
        max_length=10,
    )
    summary: str = Field(
        description="Краткий итог извлечения (DTO-имена, назначение, важные поля и объёмы) в виде JSON-строки"
    )
    raw_data: str = Field(
        description="Ссылки на DTO и служебные метаданные (без полного дампа rows) в виде JSON-строки"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        description="Статус извлечения: успех или сбой"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload


class AnalysisReportTool(SystemBaseTool):
    """Структурированные аналитические выводы; завершает шаг аналитика."""

    tool_name = "analysis_report"
    description = (
        "Передать итоговые структурированные выводы анализа и завершить шаг агента-аналитика."
    )

    reasoning: str = Field(description="Выбранный подход и методика анализа")
    completed_steps: List[str] = Field(
        description="Упорядоченный список выполненных шагов анализа",
        min_length=1,
        max_length=10,
    )
    key_findings: List[str] = Field(
        description="Ключевые находки и закономерности по данным"
    )
    conclusions: str = Field(
        description="Развёрнутые аналитические выводы на русском языке"
    )
    status: Literal[AgentStatesEnum.COMPLETED, AgentStatesEnum.FAILED] = Field(
        description="Статус анализа: успех или сбой"
    )

    async def __call__(self, context: AgentContext, config, **_) -> str:
        context.state = self.status
        payload = self.model_dump_json()
        context.execution_result = payload
        return payload
