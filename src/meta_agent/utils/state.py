"""Состояние графа мета-агента, редьюсеры LangGraph и helpers для обновления состояния.

Pydantic-модель MetaAgentState с аннотированными редьюсерами для history, dto_store, outputs, и artifacts.
Включает build_turn_state_update для подготовки обновлений перед вызовом графа.
"""

import logging
from typing import Annotated, Any

from pydantic import BaseModel, Field

from src.meta_agent.dto import DtoPayload

logger = logging.getLogger(__name__)


def append_list(left: list[Any], right: list[Any] | Any | None) -> list[Any]:
    """Generic reducer for appending items to lists.

    Supports:
    - None: returns unchanged left list
    - Single item (dict or other): converts to list and appends
    - List: concatenates with left list
    - __replace__ directive: replaces entire list with specified value
    """
    if right is None:
        return left
    if isinstance(right, dict):
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

    Использует LangGraph reducers для безопасного обновления history (append),
    dto_store (merge), outputs (append), и artifacts (append).
    """

    question: str = Field(default="")

    # Редьюсеры обеспечивают корректное объединение частичных обновлений от узлов
    history: Annotated[list[dict[str, Any]], append_list] = Field(
        default_factory=list
    )  # [{"role": str, "content": str}]

    dto_store: Annotated[dict[str, DtoPayload], merge_dto_store] = Field(
        default_factory=dict
    )  # {dto_name: DtoPayload}

    # New fields for graph-native outputs and artifacts
    outputs: Annotated[list[Any], append_list] = Field(
        default_factory=list
    )  # [AgentOutput] — ordered user-facing outputs (text, JSON, images, files)

    artifacts: Annotated[list[Any], append_list] = Field(
        default_factory=list
    )  # [AgentArtifact] — internal artifact metadata

    next_worker: str = Field(default="")
    current_task: str = Field(default="")
    delegated_attempts: int = Field(default=0)
    iterations: int = Field(default=0)
    force_bypass_ood: bool = Field(default=False)

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

    Сбрасывает управляющие поля, текущие outputs/artifacts, копирует dto_store
    для merge-редьюсера, и добавляет вопрос в историю.

    Поддерживает префикс /force для обхода OOD-проверки.
    """
    dto_store = snapshot_values.get("dto_store", {})

    force_bypass = False
    q = question.strip()
    lower_q = q.lower()
    if "/force" in lower_q:
        force_bypass = True
        # delete /force and optional following whitespace
        q = q.replace("/force", "").lstrip()

    state_update = {
        "question": q,
        "iterations": 0,
        "delegated_attempts": 0,
        "next_worker": "",
        "current_task": "",
        "dto_store": dict(dto_store),
        "force_bypass_ood": force_bypass,
        # outputs/artifacts используют append reducer; replace нужен, чтобы новый
        # ответ не переотправлял результаты прошлых ходов той же Telegram-сессии.
        "outputs": {"__replace__": []},
        "artifacts": {"__replace__": []},
        # history использует append reducer: в update передаём только дельту текущего хода
        "history": [{"role": "user", "content": question}],
    }
    return state_update
