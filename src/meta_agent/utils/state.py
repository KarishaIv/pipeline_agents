"""Состояние графа мета-агента, редьюсеры LangGraph и helpers для обновления состояния.

Pydantic-модель MetaAgentState с аннотированными редьюсерами для history и dto_store.
Включает build_turn_state_update для подготовки обновлений перед вызовом графа.
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field


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


def merge_dto_store(left: dict[str, dict], right: dict[str, dict] | None) -> dict[str, dict]:
    """Редьюсер для dto_store: объединяет новые DTO (последний побеждает при конфликте ключей)."""
    merged = dict(left) if left else {}
    if isinstance(right, dict):
        merged.update(right)
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

    dto_store: Annotated[dict[str, dict[str, Any]], merge_dto_store] = Field(
        default_factory=dict
    )  # {dto_name: dto_payload}

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
