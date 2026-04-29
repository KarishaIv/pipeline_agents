"""Pydantic models for DTO payloads and summaries.

Provides typed, validated data structures to replace ad-hoc dict-based DTOs.
Includes DtoPayload for complete data and DtoSummary for quick previews.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class DtoPayload(BaseModel):
    """Complete DTO payload with all data, metadata, and derived information.
    
    Fields:
        summary_text: Human-readable description of the dataset
        columns: List of column names in order
        num_rows: Total number of rows in the dataset
        sample: First N rows (for preview)
        rows: Complete list of all data rows
        meta: Source-specific metadata (vector name, limit, filter conditions, etc.)
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
        """Ensure num_rows matches actual rows count."""
        if v < 0:
            raise ValueError("num_rows must be non-negative")
        if "rows" in info.data and len(info.data["rows"]) != v:
            raise ValueError(f"num_rows={v} does not match len(rows)={len(info.data['rows'])}")
        return v

    @field_validator("sample")
    @classmethod
    def validate_sample(cls, v: list[dict], info) -> list[dict]:
        """Ensure sample is subset of rows."""
        if "rows" in info.data and v and info.data["rows"]:
            sample_len = len(v)
            rows_len = len(info.data["rows"])
            if sample_len > rows_len:
                raise ValueError(f"sample length {sample_len} exceeds rows length {rows_len}")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """Convert DTO payload to pandas DataFrame using rows and columns."""
        if self.rows:
            return pd.DataFrame(self.rows)
        if self.columns:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame()

    def get_summary(self, dto_name: str, max_len: int = 100) -> DtoSummary:
        """Create a DtoSummary view of this payload (truncates sample if needed)."""
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
    """Brief summary of a DTO for listing and quick preview.
    
    Used by list_dtos and similar tools to show available data without
    transferring full row data. Sample may be truncated for size.
    """

    dto_name: str
    summary_text: str
    columns: list[str]
    num_rows: int
    sample: list[dict[str, Any]] | str = Field(
        default_factory=list,
        description="Sample rows (list) or truncated preview (str)"
    )

    model_config = {"arbitrary_types_allowed": True}
