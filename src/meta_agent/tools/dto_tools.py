"""Утилиты для хранения и чтения DTO в AgentContext.custom_context."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import pandas as pd
from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from src.meta_agent.dto import DtoPayload, DtoSummary
from src.meta_agent.utils.json_responses import json_error, serialize_tool_result

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

DTO_STORE_KEY = "dto_store"
_DEFAULT_SAMPLE_SIZE = 5
_DTO_PAYLOAD_REQUIRED_FIELDS = {"summary_text", "columns", "num_rows", "sample", "rows"}


def _ensure_context_dict(context: "AgentContext") -> dict:
    if context.custom_context is None:
        context.custom_context = {}
    elif not isinstance(context.custom_context, dict):
        context.custom_context = {"legacy_context": context.custom_context}
    return context.custom_context


def _is_dto_payload_dict(value: Any) -> bool:
    return isinstance(value, dict) and _DTO_PAYLOAD_REQUIRED_FIELDS.issubset(value.keys())


def _coerce_dto_payload(value: Any) -> DtoPayload | Any:
    if isinstance(value, DtoPayload):
        return value
    if _is_dto_payload_dict(value):
        return DtoPayload.model_validate(value)
    return value


def get_dto_store(context: "AgentContext") -> dict[str, DtoPayload]:
    custom_context = _ensure_context_dict(context)
    dto_store = custom_context.get(DTO_STORE_KEY)
    if not isinstance(dto_store, dict):
        dto_store = {}
        custom_context[DTO_STORE_KEY] = dto_store
    else:
        for name, value in list(dto_store.items()):
            dto_store[name] = _coerce_dto_payload(value)
    return dto_store


def set_dto_store(context: "AgentContext", dto_store: dict[str, DtoPayload] | None) -> None:
    custom_context = _ensure_context_dict(context)
    custom_context[DTO_STORE_KEY] = {
        name: _coerce_dto_payload(value)
        for name, value in (dto_store or {}).items()
    }


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
        return [item if isinstance(item, dict) else {"value": item} for item in data]
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


def _dto_to_dataframe(dto_payload: dict[str, Any]) -> pd.DataFrame:
    """Helper to convert DTO payload (with 'rows' or 'columns') to pandas DataFrame.
    """
    rows = dto_payload.get("rows", [])
    columns = dto_payload.get("columns", [])
    if isinstance(rows, list) and rows:
        return pd.DataFrame(rows)
    if isinstance(columns, list):
        return pd.DataFrame(columns=columns)
    return pd.DataFrame()


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


def dto_summary_view(dto_name: str, dto_payload: DtoPayload, max_len: int = 100) -> DtoSummary:
    """Вернуть краткое представление DTO."""
    return dto_payload.get_summary(dto_name, max_len)


def register_dto(
    context: "AgentContext",
    *,
    source: str,
    data: Any,
    summary_text: str | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[str, DtoPayload]:
    rows = _normalize_rows(data)
    columns = _infer_columns(rows)
    dto_name = _next_dto_name(context, source)
    payload = DtoPayload(
        summary_text=summary_text or f"{source}: {len(rows)} rows, {len(columns)} columns",
        columns=columns,
        num_rows=len(rows),
        sample=rows[:_DEFAULT_SAMPLE_SIZE],
        rows=rows,
        meta=meta or {},
    )
    get_dto_store(context)[dto_name] = payload
    return dto_name, payload


def get_dto(context: "AgentContext", dto_name: str) -> DtoPayload:
    store = get_dto_store(context)
    if dto_name not in store:
        raise KeyError(dto_name)
    dto = store[dto_name]
    if not isinstance(dto, DtoPayload):
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
        return serialize_tool_result({
            "dto_count": len(dto_items),
            "dtos": dto_items
        })


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
            return json_error(
                f"DTO '{self.dto_name}' не найден",
                error_type="not_found",
                available_dto_names=available,
            )

        sample = dto.rows[self.start : self.start + self.sample_size]
        return serialize_tool_result({
            "dto_name": self.dto_name,
            "columns": dto.columns,
            "num_rows": dto.num_rows,
            "start": self.start,
            "sample_size": len(sample),
            "sample": sample,
        })


def resolve_dto_or_error(
    context: "AgentContext", dto_name: str
) -> tuple[pd.DataFrame | None, DtoPayload | None, str | None]:
    """Унифицированный resolver DTO для инструментов анализатора и code_writer.

    Возвращает (df, payload, error_json_str). Если DTO не найден — возвращает error_json
    """
    try:
        dto_payload = get_dto(context, dto_name)
    except KeyError:
        return None, None, json_error(
            f"DTO '{dto_name}' не найден. Сначала вызови list_dtos.",
            error_type="not_found",
        )

    df = dto_payload.to_dataframe()
    return df, dto_payload, None


def dto_to_dataframe(dto_payload: DtoPayload) -> pd.DataFrame:
    """Преобразует payload DTO в pandas DataFrame."""
    return dto_payload.to_dataframe()
