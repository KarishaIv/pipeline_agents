"""Utilities for compacting meta-agent history for prompts/checkpointing."""

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from langsmith import traceable

from config import get_model_uri
from src.meta_agent.agent_factory import make_openai_client
from src.meta_agent.config import (
    HISTORY_SUMMARY_MAX_TOKENS,
    HISTORY_SUMMARY_MODEL,
    MAX_HISTORY_CHARS,
    SUMMARY_RECENT_MESSAGES,
)
from src.meta_agent.prompts import HISTORY_SUMMARIZER_SYSTEM


Message = dict[str, Any]
SummaryCallable = Callable[[str], Awaitable[str]]

TRUNCATION_MARKER = "…(история обрезана)…\n\n"
HISTORY_SUMMARY_ROLE = "history_summary"
LOGGER = logging.getLogger("meta_agent")


def truncate_output_value(value: Any, max_len: int) -> Any:
    """Truncate tool output payloads for compact history entries."""
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


def _normalize_message(message: Message) -> Message:
    return {
        "role": str(message.get("role", "unknown")),
        "content": str(message.get("content", "")),
    }


def _normalize_history(history: list[Message]) -> list[Message]:
    return [_normalize_message(message) for message in history]


def _history_as_text(history: list[Message]) -> str:
    parts = [f"[{message['role'].upper()}]: {message['content']}" for message in history]
    return "\n\n".join(parts)


def _history_len(history: list[Message]) -> int:
    return len(_history_as_text(history))


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            elif hasattr(item, "type") and getattr(item, "type", None) == "text":
                chunks.append(str(getattr(item, "text", "")))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content or "")


@traceable(name="history.default_summarizer", run_type="llm")
async def _default_history_summarizer(history_text: str, max_tokens: int | None = None) -> str:
    """Summarize history via direct chat completion (no tools)."""
    if not os.getenv("YANDEX_API_KEY") or not os.getenv("YANDEX_FOLDER_ID"):
        LOGGER.debug("History summarizer skipped: missing Yandex credentials in environment")
        return ""

    client = make_openai_client()
    completion_kwargs = {
        "model": get_model_uri(HISTORY_SUMMARY_MODEL),
        "messages": [
            {"role": "system", "content": HISTORY_SUMMARIZER_SYSTEM},
            {"role": "user", "content": history_text},
        ],
        "temperature": 0.0,
    }
    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens

    response = await client.chat.completions.create(**completion_kwargs)
    if not response.choices:
        return ""
    message = response.choices[0].message
    return _extract_text_content(message.content).strip()


async def _summarize_messages(
    messages: list[Message],
    summarizer: SummaryCallable | None = None,
    summary_max_tokens: int | None = HISTORY_SUMMARY_MAX_TOKENS,
) -> str | None:
    if not messages:
        return None

    try:
        if summarizer is not None:
            summary = await summarizer(_history_as_text(messages))
        else:
            summary = await _default_history_summarizer(
                _history_as_text(messages),
                max_tokens=summary_max_tokens,
            )
    except Exception as exc:
        LOGGER.warning("LLM history summarization failed: %s", exc)
        return None

    normalized = summary.strip()
    return normalized if normalized else None


def truncate_history(history: list[Message]) -> str:
    """Deterministic textual fallback: keep the tail within MAX_HISTORY_CHARS."""
    serialized_history = _history_as_text(history)
    if len(serialized_history) > MAX_HISTORY_CHARS:
        return TRUNCATION_MARKER + serialized_history[-MAX_HISTORY_CHARS:]
    return serialized_history


def _trim_single_message_to_limit(message: Message, char_limit: int) -> Message:
    role = str(message.get("role", "unknown"))
    prefix = f"[{role.upper()}]: "
    content_budget = char_limit - len(prefix)
    if content_budget <= 0:
        return {"role": role, "content": "…"}

    content = str(message.get("content", ""))
    if len(content) <= content_budget:
        return {"role": role, "content": content}

    marker_budget = content_budget - len(TRUNCATION_MARKER)
    if marker_budget <= 0:
        trimmed_content = content[-content_budget:]
    else:
        trimmed_content = TRUNCATION_MARKER + content[-marker_budget:]
    return {"role": role, "content": trimmed_content}


def truncate_history_list(history: list[Message], max_chars: int | None = None) -> list[Message]:
    """Deterministic list fallback: drop oldest entries, then trim latest if needed."""
    if not history:
        return []

    char_limit = max_chars if max_chars is not None else MAX_HISTORY_CHARS
    history_copy = _normalize_history(history)

    while len(history_copy) > 1 and _history_len(history_copy) > char_limit:
        history_copy.pop(0)

    if _history_len(history_copy) <= char_limit:
        return history_copy

    trimmed_latest = _trim_single_message_to_limit(history_copy[-1], char_limit)
    return [trimmed_latest]


def _filter_history_by_roles(history: list[Message], roles: tuple[str, ...]) -> list[Message]:
    allowed = set(roles)
    allowed.add(HISTORY_SUMMARY_ROLE) # Keep previously generated summary context even if caller asks for worker-only roles.

    return [msg for msg in _normalize_history(history) if msg.get("role") in allowed]


@traceable(name="history.summarize_list", run_type="chain")
async def summarize_history_list(
    history: list[Message],
    max_chars: int | None = None,
    *,
    preserve_recent_messages: int = SUMMARY_RECENT_MESSAGES,
    summarizer: SummaryCallable | None = None,
    summary_max_tokens: int | None = HISTORY_SUMMARY_MAX_TOKENS,
) -> list[Message]:
    """Compress history with LLM summary; fallback to deterministic truncation."""
    if not history:
        return []

    char_limit = max_chars if max_chars is not None else MAX_HISTORY_CHARS
    history_copy = _normalize_history(history)
    if _history_len(history_copy) <= char_limit:
        return history_copy

    recent_count = max(1, preserve_recent_messages)
    while len(history_copy) > recent_count:
        older_messages = history_copy[:-recent_count]
        recent_messages = history_copy[-recent_count:]
        summary = await _summarize_messages(
            older_messages,
            summarizer=summarizer,
            summary_max_tokens=summary_max_tokens,
        )
        if summary:
            candidate = [{"role": HISTORY_SUMMARY_ROLE, "content": summary}] + recent_messages
            if _history_len(candidate) <= char_limit:
                return candidate
        if recent_count == 1:
            break
        recent_count = max(1, recent_count // 2)

    full_summary = await _summarize_messages(
        history_copy,
        summarizer=summarizer,
        summary_max_tokens=summary_max_tokens,
    )
    if full_summary:
        summary_only = [{"role": HISTORY_SUMMARY_ROLE, "content": full_summary}]
        if _history_len(summary_only) <= char_limit:
            return summary_only

    return truncate_history_list(history_copy, max_chars=char_limit)


async def summarize_history_text(
    history: list[Message],
    max_chars: int | None = None,
    *,
    summarizer: SummaryCallable | None = None,
    summary_max_tokens: int | None = HISTORY_SUMMARY_MAX_TOKENS,
) -> str:
    summarized = await summarize_history_list(
        history,
        max_chars=max_chars,
        summarizer=summarizer,
        summary_max_tokens=summary_max_tokens,
    )
    return _history_as_text(summarized)


async def build_role_history_text_async(
    history: list[Message],
    roles: tuple[str, ...],
    max_chars: int | None = None,
    *,
    summarizer: SummaryCallable | None = None,
    summary_max_tokens: int | None = HISTORY_SUMMARY_MAX_TOKENS,
) -> str:
    filtered_history = _filter_history_by_roles(history, roles)
    if not filtered_history:
        return ""

    summarized = await summarize_history_list(
        filtered_history,
        max_chars=max_chars,
        summarizer=summarizer,
        summary_max_tokens=summary_max_tokens,
    )
    return _history_as_text(summarized)


async def build_persisted_history(
    result: dict[str, Any],
    *,
    summarizer: SummaryCallable | None = None,
    summary_max_tokens: int | None = HISTORY_SUMMARY_MAX_TOKENS,
) -> list[Message]:
    """Build final history snapshot for checkpointing."""
    answer = result.get("answer", "")
    result_history = list(result.get("history", []))
    if answer:
        result_history.append({"role": "assistant", "content": answer})
    return await summarize_history_list(
        result_history,
        summarizer=summarizer,
        summary_max_tokens=summary_max_tokens,
    )
