"""Pydantic-модели DTO payload и кратких сводок.

Модуль задаёт типизированные структуры для хранения полных данных DTO
и коротких представлений, которые можно безопасно показывать агентам.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class DtoPayload(BaseModel):
    """Полный payload DTO со строками данных, колонками и метаданными.

    Поля:
        summary_text: человекочитаемое описание набора данных;
        columns: имена колонок в порядке появления;
        num_rows: общее число строк;
        sample: первые строки для предварительного просмотра;
        rows: полный список строк данных;
        meta: служебные метаданные источника, например вектор, limit или фильтр.
    """

    summary_text: str
    columns: list[str]
    num_rows: int
    sample: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("num_rows")
    @classmethod
    def validate_num_rows(cls, v: int, info) -> int:
        """Проверить, что num_rows совпадает с фактическим числом rows."""
        if v < 0:
            raise ValueError("num_rows must be non-negative")
        if "rows" in info.data and len(info.data["rows"]) != v:
            raise ValueError(f"num_rows={v} does not match len(rows)={len(info.data['rows'])}")
        return v

    @field_validator("sample")
    @classmethod
    def validate_sample(cls, v: list[dict], info) -> list[dict]:
        """Проверить, что sample не длиннее rows."""
        if "rows" in info.data and v and info.data["rows"]:
            sample_len = len(v)
            rows_len = len(info.data["rows"])
            if sample_len > rows_len:
                raise ValueError(f"sample length {sample_len} exceeds rows length {rows_len}")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Преобразовать DTO payload в pandas DataFrame по rows и columns."""
        if self.rows:
            return pd.DataFrame(self.rows)
        if self.columns:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame()

    def get_summary(self, dto_name: str, max_len: int = 100) -> DtoSummary:
        """Создать DtoSummary и при необходимости обрезать sample."""
        truncated_sample = self.sample
        if isinstance(truncated_sample, list):
            import json
            sample_str = json.dumps(truncated_sample)
            if len(sample_str) > max_len:
                truncated_sample = str(sample_str)[:max_len]
        return DtoSummary(
            dto_name=dto_name,
            summary_text=self.summary_text,
            columns=self.columns,
            num_rows=self.num_rows,
            sample=truncated_sample,
        )


class DtoSummary(BaseModel):
    """Краткая сводка DTO для списков и быстрого просмотра.

    Используется list_dtos и похожими инструментами, чтобы показать доступные
    данные без передачи полного rows. Sample может быть обрезан по длине.
    """

    dto_name: str
    summary_text: str
    columns: list[str]
    num_rows: int
    sample: list[dict[str, Any]] | str = Field(
        default_factory=list,
        description="Пример строк списком или обрезанная строковая версия sample"
    )

    model_config = {"arbitrary_types_allowed": True}
