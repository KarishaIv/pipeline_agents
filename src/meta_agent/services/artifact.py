"""Artifact service for secure chart, JSON, and CSV file management."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd


class ArtifactSaveError(Exception):
    """Raised when an artifact save operation fails."""


class ArtifactService:
    """Securely save and describe generated artifacts."""

    _EXTENSION_META = {
        ".png": ("chart", "image/png"),
        ".jpg": ("chart", "image/jpeg"),
        ".jpeg": ("chart", "image/jpeg"),
        ".pdf": ("pdf", "application/pdf"),
        ".json": ("data", "application/json"),
        ".csv": ("csv", "text/csv"),
    }

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir).resolve()
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as exc:
            raise ArtifactSaveError(
                f"Cannot create artifacts directory {self.artifacts_dir}: {exc}"
            ) from exc

    def sanitize_filename(self, filename: str | None, *, default_extension: str) -> str:
        """Return a safe basename with the requested default extension."""
        default_extension = self._normalize_extension(default_extension)
        fallback = f"artifact_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{default_extension}"
        if not filename or not filename.strip():
            return fallback

        sanitized = re.sub(r"[^\w\.-]", "_", filename.strip())
        sanitized = re.sub(r"_+", "_", sanitized)

        if (
            not sanitized
            or ".." in sanitized
            or "/" in sanitized
            or "\\" in sanitized
            or sanitized.startswith(".")
        ):
            sanitized = fallback

        suffix = Path(sanitized).suffix.lower()
        if suffix and suffix != default_extension:
            sanitized = f"{Path(sanitized).stem}{default_extension}"
        elif not suffix:
            sanitized += default_extension

        return sanitized

    def save_chart(
        self,
        filename: str | None = None,
        *,
        caption: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Save the active matplotlib figure and return artifact metadata."""
        safe_name = self._available_filename(
            self.sanitize_filename(filename, default_extension=".png")
        )
        target_path = self._target_path(safe_name)

        try:
            plt.savefig(str(target_path), bbox_inches="tight", dpi=150)
            plt.close()
        except (OSError, PermissionError, ValueError) as exc:
            raise ArtifactSaveError(f"Failed to save chart to {target_path}: {exc}") from exc

        return str(target_path), self._metadata(safe_name, caption=caption, metadata=metadata)

    def save_json(
        self,
        data: Any,
        filename: str | None = None,
        *,
        caption: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Save raw JSON-serializable data and return artifact metadata."""
        safe_name = self._available_filename(
            self.sanitize_filename(filename, default_extension=".json")
        )
        target_path = self._target_path(safe_name)
        payload = self._records_from_data(data)

        try:
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except (OSError, PermissionError, TypeError, ValueError) as exc:
            raise ArtifactSaveError(f"Failed to save JSON artifact to {target_path}: {exc}") from exc

        return str(target_path), self._metadata(safe_name, caption=caption, metadata=metadata)

    def save_csv(
        self,
        data: Any,
        filename: str | None = None,
        *,
        caption: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Save raw tabular data as CSV and return artifact metadata."""
        safe_name = self._available_filename(
            self.sanitize_filename(filename, default_extension=".csv")
        )
        target_path = self._target_path(safe_name)

        try:
            self._dataframe_from_data(data).to_csv(target_path, index=False)
        except (OSError, PermissionError, ValueError) as exc:
            raise ArtifactSaveError(f"Failed to save CSV artifact to {target_path}: {exc}") from exc

        return str(target_path), self._metadata(safe_name, caption=caption, metadata=metadata)

    def artifact_from_existing_file(
        self,
        path: Path | str,
        *,
        caption: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build artifact metadata for an already-created file in the artifact directory."""
        artifact_path = Path(path).resolve()
        self._ensure_within_artifacts_dir(artifact_path)
        if not artifact_path.exists():
            raise ArtifactSaveError(f"Artifact file does not exist: {artifact_path}")
        return self._metadata(artifact_path.name, caption=caption, metadata=metadata)

    def _available_filename(self, filename: str) -> str:
        path = self.artifacts_dir / filename
        if not path.exists():
            return filename

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        return f"{stem}_{uuid4().hex[:8]}{suffix}"

    def _target_path(self, filename: str) -> Path:
        target_path = (self.artifacts_dir / filename).resolve()
        self._ensure_within_artifacts_dir(target_path)
        return target_path

    def _ensure_within_artifacts_dir(self, path: Path) -> None:
        try:
            path.relative_to(self.artifacts_dir)
        except ValueError as exc:
            raise ArtifactSaveError(f"Artifact path escapes artifacts directory: {path}") from exc

    def _metadata(
        self,
        filename: str,
        *,
        caption: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        kind, mime_type = self._EXTENSION_META.get(suffix, ("file", "application/octet-stream"))
        return {
            "id": str(uuid4()),
            "kind": kind,
            "path": str(self.artifacts_dir / filename),
            "filename": filename,
            "mime_type": mime_type,
            "caption": caption,
            "metadata": dict(metadata or {}),
        }

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        return extension if extension.startswith(".") else f".{extension}"

    @staticmethod
    def _records_from_data(data: Any) -> Any:
        if hasattr(data, "to_dict"):
            return data.to_dict(orient="records")
        return data

    @classmethod
    def _dataframe_from_data(cls, data: Any) -> pd.DataFrame:
        if hasattr(data, "to_csv"):
            return data
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame(data)
