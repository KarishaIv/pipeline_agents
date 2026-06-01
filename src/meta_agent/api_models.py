"""Pydantic-модели запросов и ответов API meta-agent.

Модуль переэкспортирует output-модели и задаёт контракт эндпоинта POST /ask.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field

# Import and re-export output models
from src.meta_agent.output_models import (
    TextOutput,
    JsonOutput,
    FileOutput,
    ImageOutput,
    AgentOutput,
    AgentArtifact,
)


class AskRequest(BaseModel):
    """Тело запроса POST /ask."""

    question: str = Field(
        ...,
        description="Вопрос пользователя для meta-agent",
        min_length=1,
        max_length=10000,
    )
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Идентификатор thread для продолжения диалога. "
            "null или отсутствие поля — новая сессия; "
            "-1 — явно начать новую сессию; "
            "иное значение — продолжить существующую сессию"
        ),
    )


class MetaAgentApiResponse(BaseModel):
    """Структурированный ответ эндпоинта POST /ask."""

    thread_id: str = Field(
        ..., description="Идентификатор thread для следующих сообщений этой сессии"
    )
    outputs: list[AgentOutput] = Field(
        default_factory=list,
        description="Упорядоченный список результатов: текст, JSON, изображения, файлы и т.п.",
    )


class ErrorResponse(BaseModel):
    """Ответ API с ошибкой."""

    error: str = Field(..., description="Понятное пользователю сообщение об ошибке")
    error_type: str = Field(
        default="unknown_error",
        description="Классификация ошибки: validation_error, timeout_error и т.п.",
    )
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Дополнительный контекст ошибки"
    )


__all__ = [
    "AskRequest",
    "TextOutput",
    "JsonOutput",
    "FileOutput",
    "ImageOutput",
    "AgentOutput",
    "AgentArtifact",
    "MetaAgentApiResponse",
    "ErrorResponse",
]
