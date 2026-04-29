"""Состояние графа мета-агента, редьюсеры LangGraph и helpers для обновления состояния.

Pydantic-модель MetaAgentState с аннотированными редьюсерами для history и dto_store.
Включает build_turn_state_update для подготовки обновлений перед вызовом графа.
"""

import logging
from typing import Annotated, Any

from pydantic import BaseModel, Field

from src.meta_agent.dto import DtoPayload

logger = logging.getLogger(__name__)


def append_history(left: list[dict], right: list[dict] | dict | None) -> list[dict]:
    """Редьюсер для поля history: добавляет новые сообщения (поддерживает list или dict)."""
    if right is None:
        return left
    if isinstance(right, dict):
        
        # Если в right есть ключ "__replace__" и значение этого ключа является списком,
        # то возвращаем этот список вместо добавления нового сообщения
        if "__replace__" in right and isinstance(right["__replace__"], list):
            return right["__replace__"]

        right = [right]
    if isinstance(right, list):
        return (left or []) + right
    return left or []


def merge_dto_store(
    left: dict[str, DtoPayload], right: dict[str, DtoPayload] | None
) -> dict[str, DtoPayload]:
    """Редьюсер для dto_store: объединяет новые DTO (последний побеждает при конфликте ключей).
    
    Ensures that all values in the merged store are DtoPayload objects. This handles
    the case where right-side values (from node state updates) are dicts that need
    to be converted to DtoPayload instances.
    
    Raises ValueError if a dict cannot be converted to DtoPayload (indicates malformed data).
    """
    merged = dict(left) if left else {}
    if isinstance(right, dict):
        for key, value in right.items():
            if isinstance(value, dict) and not isinstance(value, DtoPayload):
                try:
                    merged[key] = DtoPayload(**value)
                except (ValueError, TypeError) as e:
                    logger.error(
                        "Failed to convert dto_store[%r] to DtoPayload: %s. "
                        "This indicates malformed DTO data in state update.",
                        key, e
                    )
                    raise ValueError(
                        f"Cannot convert dto_store[{key!r}] to DtoPayload: {e}. "
                        "All DTO values must be valid DtoPayload objects."
                    ) from e
            else:
                merged[key] = value
    return merged


class MetaAgentState(BaseModel):
    """Pydantic-модель состояния графа мета-агента.

    Использует LangGraph reducers для безопасного обновления history (append) и dto_store (merge).
    """

    question: str = Field(default="")

    # Редьюсеры обеспечивают корректное объединение частичных обновлений от узлов
    history: Annotated[list[dict[str, Any]], append_history] = Field(
        default_factory=list
    )  # [{"role": str, "content": str}]

    dto_store: Annotated[dict[str, DtoPayload], merge_dto_store] = Field(
        default_factory=dict
    )  # {dto_name: DtoPayload}

    next_worker: str = Field(default="")
    current_task: str = Field(default="")
    delegated_attempts: int = Field(default=0)
    answer: str = Field(default="")
    iterations: int = Field(default=0)

    model_config = {"arbitrary_types_allowed": True}


def state_to_dict(state: dict | Any) -> dict:
    """Преобразует состояние (Pydantic или dict) в обычный словарь.

    Мост между Pydantic-состоянием LangGraph и кодом узлов, которые ожидают dict.
    """
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state) if not isinstance(state, dict) else state


def build_turn_state_update(question: str, snapshot_values: dict) -> dict:
    """Формирует обновление состояния для очередного хода графа.

    Сбрасывает управляющие поля, сохраняет dto_store и добавляет вопрос в историю.
    """
    dto_store = snapshot_values.get("dto_store", {})

    state_update = {
        "question": question,
        "iterations": 0,
        "delegated_attempts": 0,
        "next_worker": "",
        "current_task": "",
        "answer": "",
        "dto_store": dict(dto_store),
        # history использует append reducer: в update передаём только дельту текущего хода
        "history": [{"role": "user", "content": question}],
    }
    return state_update
