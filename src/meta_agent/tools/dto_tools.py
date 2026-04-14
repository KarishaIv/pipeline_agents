"""Утилиты для хранения и чтения DTO в AgentContext.custom_context."""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from src.meta_agent.utils import truncate_output_value

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

DTO_STORE_KEY = "dto_store"
_DEFAULT_SAMPLE_SIZE = 5


def _ensure_context_dict(context: "AgentContext") -> dict:
    if context.custom_context is None:
        context.custom_context = {}
    elif not isinstance(context.custom_context, dict):
        context.custom_context = {"legacy_context": context.custom_context}
    return context.custom_context


def get_dto_store(context: "AgentContext") -> dict[str, dict[str, Any]]:
    custom_context = _ensure_context_dict(context)
    dto_store = custom_context.get(DTO_STORE_KEY)
    if not isinstance(dto_store, dict):
        dto_store = {}
        custom_context[DTO_STORE_KEY] = dto_store
    return dto_store


def set_dto_store(context: "AgentContext", dto_store: dict[str, dict[str, Any]] | None) -> None:
    custom_context = _ensure_context_dict(context)
    custom_context[DTO_STORE_KEY] = dto_store or {}


def _normalize_rows(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, dict):
        if isinstance(data.get("points"), list):
            return [item if isinstance(item, dict) else {"value": item} for item in data["points"]]
        return [data]
    if isinstance(data, list):
        if not data:
            return []
        if all(isinstance(item, dict) for item in data):
            return data
        return [{"value": item} for item in data]
    return [{"value": data}]


def _infer_columns(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def _sanitize_source(source: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", source.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "dto"


def _next_dto_name(context: "AgentContext", source: str) -> str:
    store = get_dto_store(context)
    base = _sanitize_source(source)
    idx = 1
    candidate = f"{base}_{idx}"
    while candidate in store:
        idx += 1
        candidate = f"{base}_{idx}"
    return candidate


def dto_summary_view(dto_name: str, dto_payload: dict[str, Any], max_len: int = 100) -> dict[str, Any]:
    """Вернуть краткое представление DTO."""
    return {
        "dto_name": dto_name,
        "summary_text": dto_payload.get("summary_text", ""),
        "columns": dto_payload.get("columns", []),
        "num_rows": dto_payload.get("num_rows", 0),
        "sample": truncate_output_value(dto_payload.get("sample", []), max_len),
    }


def register_dto(
    context: "AgentContext",
    *,
    source: str,
    data: Any,
    summary_text: str | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    rows = _normalize_rows(data)
    columns = _infer_columns(rows)
    dto_name = _next_dto_name(context, source)
    payload = {
        "summary_text": summary_text or f"{source}: {len(rows)} rows, {len(columns)} columns",
        "columns": columns,
        "num_rows": len(rows),
        "sample": rows[:_DEFAULT_SAMPLE_SIZE],
        "rows": rows,
        "meta": meta or {},
    }
    get_dto_store(context)[dto_name] = payload
    return dto_name, payload


def get_dto(context: "AgentContext", dto_name: str) -> dict[str, Any]:
    store = get_dto_store(context)
    if dto_name not in store:
        raise KeyError(dto_name)
    dto = store[dto_name]
    if not isinstance(dto, dict):
        raise KeyError(dto_name)
    return dto


class ListDtoNamesTool(BaseTool):
    """Вернуть список всех DTO, доступных в custom_context."""

    tool_name = "list_dtos"
    description = "Показать имена всех доступных DTO и их краткие сводки (без полного набора данных)."

    reasoning: str = Field(description="Зачем нужен просмотр доступных DTO")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        store = get_dto_store(context)
        dto_items = [
            dto_summary_view(name, payload, 50)
            for name, payload in sorted(store.items())
        ]
        return json.dumps(
            {"dto_count": len(dto_items), "dtos": dto_items},
            ensure_ascii=False,
            default=str,
        )


class SampleDtoTool(BaseTool):
    """Вернуть дополнительную выборку строк из DTO по имени."""

    tool_name = "sample_dto"
    description = "Получить sample строк из DTO по его имени (без возврата полного набора rows)."

    reasoning: str = Field(description="Зачем нужна дополнительная выборка из DTO")
    dto_name: str = Field(description="Имя DTO, полученное из list_dtos или extractor-инструментов")
    sample_size: int = Field(default=5, ge=1, le=100, description="Сколько строк вернуть")
    start: int = Field(default=0, ge=0, description="Смещение начала выборки")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            dto = get_dto(context, self.dto_name)
        except KeyError:
            available = sorted(get_dto_store(context).keys())
            return json.dumps(
                {"error": f"DTO '{self.dto_name}' не найден", "available_dto_names": available},
                ensure_ascii=False,
            )

        rows = dto.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        sample = rows[self.start : self.start + self.sample_size]
        return json.dumps(
            {
                "dto_name": self.dto_name,
                "columns": dto.get("columns", []),
                "num_rows": dto.get("num_rows", len(rows)),
                "start": self.start,
                "sample_size": len(sample),
                "sample": sample,
            },
            ensure_ascii=False,
            default=str,
        )
