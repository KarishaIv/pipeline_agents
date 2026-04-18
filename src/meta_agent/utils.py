"""Вспомогательные функции для работы с историей сессии мета-агента."""

from __future__ import annotations

import json
import uuid
from typing import Any, Annotated

from src.meta_agent.config import MAX_HISTORY_CHARS


def append_history(left: list[dict], right: list[dict] | dict | None) -> list[dict]:
    """LangGraph reducer for history field: appends new entries (supports list or single dict update)."""
    if right is None:
        return left
    if isinstance(right, dict):
        right = [right]
    if isinstance(right, list):
        return (left or []) + right
    return left or []


def merge_dto_store(left: dict[str, dict], right: dict[str, dict] | None) -> dict[str, dict]:
    """LangGraph reducer for dto_store: merges new DTO payloads (last writer wins on key collision)."""
    merged = dict(left) if left else {}
    if isinstance(right, dict):
        merged.update(right)
    return merged


def state_to_dict(state: dict | Any) -> dict:
    """Convert LangGraph state (Pydantic MetaAgentState or dict) to plain dict.

    This is the bridge when using Pydantic state schema with reducers.
    Nodes can continue to use dict interface while benefiting from reducers.
    """
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state) if not isinstance(state, dict) else state

_TRUNCATION_MARKER = "…(история обрезана)…\n\n"


def truncate_output_value(value: Any, max_len: int) -> Any:
    """Обрезать значение для компактного ответа инструмента.

    - str: обрезается напрямую;
    - list/dict: сначала сериализуется целиком в JSON-строку, затем обрезается;
    - остальные типы возвращаются без изменений.
    """
    if isinstance(value, str):
        if len(value) <= max_len:
            return value
        return value[:max_len] + "..."
    if isinstance(value, (list, dict)):
        rendered = json.dumps(value, ensure_ascii=False, default=str)
        if len(rendered) <= max_len:
            return rendered
        return rendered[:max_len] + "..."
    return value


def _history_as_text(history: list) -> str:
    """Сериализовать историю в формате, который использует супервайзер в промпте."""
    serialized_parts = [f"[{message['role'].upper()}]: {message['content']}" for message in history]
    return "\n\n".join(serialized_parts)


def truncate_history(history: list) -> str:
    """Преобразовать историю в текст, сохранив только хвост до MAX_HISTORY_CHARS."""
    serialized_history = _history_as_text(history)
    if len(serialized_history) > MAX_HISTORY_CHARS:
        return _TRUNCATION_MARKER + serialized_history[-MAX_HISTORY_CHARS:]
    return serialized_history


def _trim_single_message_to_limit(message: dict, char_limit: int) -> dict:
    """Обрезать одно сообщение так, чтобы его сериализованный вид влезал в char_limit."""
    role = str(message.get("role", "unknown"))
    prefix = f"[{role.upper()}]: "
    content_budget = char_limit - len(prefix)
    if content_budget <= 0:
        return {"role": role, "content": "…"}

    content = str(message.get("content", ""))
    if len(content) <= content_budget:
        return {"role": role, "content": content}

    marker_budget = content_budget - len(_TRUNCATION_MARKER)
    if marker_budget <= 0:
        trimmed_content = content[-content_budget:]
    else:
        trimmed_content = _TRUNCATION_MARKER + content[-marker_budget:]
    return {"role": role, "content": trimmed_content}


def truncate_history_list(history: list, max_chars: int | None = None) -> list:
    """Удалять самые старые сообщения, пока сериализованная история не уложится в лимит."""
    if not history:
        return []

    char_limit = max_chars if max_chars is not None else MAX_HISTORY_CHARS
    # Копируем каждую запись, чтобы обрезка не меняла исходный список/словари у вызывающего кода.
    history_copy = [dict(message) for message in history]

    while len(history_copy) > 1 and len(_history_as_text(history_copy)) > char_limit:
        history_copy.pop(0)

    if len(_history_as_text(history_copy)) <= char_limit:
        return history_copy

    trimmed_latest = _trim_single_message_to_limit(history_copy[-1], char_limit)
    return [trimmed_latest]


def resolve_thread_id(thread_id: str | None) -> str:
    """Определить идентификатор сессии по правилам API.

    - thread_id == "-1": всегда создать новую сессию;
    - thread_id is None: создать новую сессию;
    - иначе: использовать переданный thread_id.
    """
    if thread_id == "-1" or thread_id is None:
        return str(uuid.uuid7())
    return thread_id


def build_turn_state_update(question: str, snapshot_values: dict) -> dict:
    """Собрать state update для очередного хода с добавлением вопроса в историю.

    Explicitly resets control fields while preserving dto_store and history.
    The reducers defined on MetaAgentState (append_history, merge_dto_store)
    will handle how these updates are applied by LangGraph.
    """
    existing_history = list(snapshot_values.get("history", []))
    dto_store = snapshot_values.get("dto_store", {})

    state_update = {
        "question": question,
        "iterations": 0,
        "delegated_attempts": 0,
        "next_worker": "",
        "current_task": "",
        "answer": "",
        "dto_store": dict(dto_store),  # explicit copy for reducer
        "history": existing_history + [{"role": "user", "content": question}],
    }
    return state_update


def build_persisted_history(result: dict, question: str) -> list:
    """Собрать историю для сохранения после выполнения графа и ограничить её размер."""
    answer = result.get("answer", "")
    result_history = list(result.get("history", []))
    result_history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    return truncate_history_list(result_history)


def route_supervisor(state: dict | Any) -> str:
    """Вернуть имя следующего узла по решению супервайзера.
    Uses state_to_dict to support both dict and Pydantic MetaAgentState (from reducers).
    """
    state = state_to_dict(state)
    return state.get("next_worker", "end")


def route_analyzer(state: dict | Any) -> str:
    """Вернуть следующий узел после analyzer: code_writer или supervisor.
    Uses state_to_dict to support both dict and Pydantic MetaAgentState (from reducers).
    """
    state = state_to_dict(state)
    next_worker = state.get("next_worker", "supervisor")
    if next_worker == "code_writer":
        return "code_writer"
    return "supervisor"
