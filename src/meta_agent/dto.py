"""Pydantic-модели DTO payload и кратких сводок.

Модуль задаёт типизированные структуры для хранения полных данных DTO
и коротких представлений, которые можно безопасно показывать агентам.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class DtoPayload(BaseModel):
    """Полный payload DTO со строками данных, колонками и метаданными.

    Поля:
        summary_text: человекочитаемое описание набора данных;
        columns: имена колонок в порядке появления;
        rows: полный список строк данных;
        meta: служебные метаданные источника, например вектор, limit или фильтр.
    """

    summary_text: str
    columns: list[str]
    rows: list[dict[str, Any]]
    meta: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def num_rows(self) -> int:
        """Вернуть число строк, вычисленное из rows."""
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        """Преобразовать DTO payload в pandas DataFrame по rows и columns."""
        if self.rows:
            return pd.DataFrame(self.rows)
        if self.columns:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame()

    def get_summary(
        self, dto_name: str, max_len: int = 100, sample_size: int = 5
    ) -> DtoSummary:
        """Создать DtoSummary с preview-sample, вычисленным из rows."""
        truncated_sample: list[dict[str, Any]] | str = self.rows[:sample_size]
        if isinstance(truncated_sample, list):
            import json
            sample_str = json.dumps(truncated_sample, ensure_ascii=False)
            if len(sample_str) > max_len:
                truncated_sample = sample_str[:max_len]
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
