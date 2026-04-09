"""Вспомогательные функции для работы с историей сессии мета-агента."""

from __future__ import annotations

import uuid

from src.meta_agent.prompts import MAX_HISTORY_CHARS

_TRUNCATION_MARKER = "…(история обрезана)…\n\n"


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


def resolve_thread_id(thread_id: str | None, last_thread_id: str | None) -> str:
    """Определить идентификатор сессии по правилам API.

    - thread_id == "-1": всегда создать новую сессию;
    - thread_id is None: использовать предыдущую сессию, если есть;
    - иначе: использовать переданный thread_id.
    """
    if thread_id == "-1":
        return str(uuid.uuid7())
    if thread_id is None:
        return last_thread_id or str(uuid.uuid7())
    return thread_id


def build_turn_state_update(question: str, snapshot_values: dict) -> dict:
    """Собрать state update для очередного хода с добавлением вопроса в историю."""
    existing_history = list(snapshot_values.get("history", []))
    if snapshot_values:
        state_update = {"question": question, "iterations": 0}
    else:
        state_update = {
            "question": question,
            "history": [],
            "next_worker": "",
            "current_task": "",
            "answer": "",
            "iterations": 0,
        }
    state_update["history"] = existing_history + [{"role": "user", "content": question}]
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


def route_supervisor(state: dict) -> str:
    """Вернуть имя следующего узла по решению супервайзера."""
    return state.get("next_worker", "end")
