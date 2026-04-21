"""Утилиты для работы с историей: обрезка для промптов и формирование persisted history.
"""

import json
from typing import Any

from src.meta_agent.config import MAX_HISTORY_CHARS


_TRUNCATION_MARKER = "…(история обрезана)…\n\n"


def truncate_output_value(value: Any, max_len: int) -> Any:
    """Обрезает значение для компактного ответа инструмента.

    - str: обрезается напрямую;
    - list/dict: сериализуется в JSON, затем обрезается;
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
    """Сериализует историю в текстовый формат для промпта супервайзера."""
    serialized_parts = [f"[{message['role'].upper()}]: {message['content']}" for message in history]
    return "\n\n".join(serialized_parts)


def truncate_history(history: list) -> str:
    """Преобразует историю в текст, сохраняя только хвост до MAX_HISTORY_CHARS."""
    serialized_history = _history_as_text(history)
    if len(serialized_history) > MAX_HISTORY_CHARS:
        return _TRUNCATION_MARKER + serialized_history[-MAX_HISTORY_CHARS:]
    return serialized_history


def _trim_single_message_to_limit(message: dict, char_limit: int) -> dict:
    """Обрезает одно сообщение, чтобы оно уложилось в лимит символов."""
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
    """Удаляет старые сообщения, пока сериализованная история не уложится в лимит.

    Если после удаления всё равно слишком длинная — обрезает последнее сообщение.
    """
    if not history:
        return []

    char_limit = max_chars if max_chars is not None else MAX_HISTORY_CHARS
    # Копируем, чтобы не модифицировать оригинал
    history_copy = [dict(message) for message in history]

    while len(history_copy) > 1 and len(_history_as_text(history_copy)) > char_limit:
        history_copy.pop(0)

    if len(_history_as_text(history_copy)) <= char_limit:
        return history_copy

    trimmed_latest = _trim_single_message_to_limit(history_copy[-1], char_limit)
    return [trimmed_latest]


def build_role_history_text(
    history: list,
    roles: tuple[str, ...],
    max_chars: int | None = None,
) -> str:
    """Формирует ограниченный по размеру текст истории только для указанных ролей."""
    if not history:
        return ""

    selected_roles = set(roles)
    filtered_history = [dict(message) for message in history if message.get("role") in selected_roles]
    if not filtered_history:
        return ""

    truncated = truncate_history_list(filtered_history, max_chars=max_chars)
    return _history_as_text(truncated)


def build_persisted_history(result: dict) -> list:
    """Формирует историю для сохранения после выполнения графа с обрезкой."""
    answer = result.get("answer", "")
    result_history = list(result.get("history", []))
    if answer:
        result_history.append({"role": "assistant", "content": answer})
    return truncate_history_list(result_history)
