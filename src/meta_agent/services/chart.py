"""Chart service for secure matplotlib figure management.

Provides chart saving functionality with path security checks,
sanitization of filenames, and proper directory management.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


class ChartSaveError(Exception):
    """Raised when chart save operation fails due to security or I/O issues."""

    pass


class ChartService:
    """Service for securely saving matplotlib figures to disk.

    Handles:
    - Filename sanitization to prevent path traversal attacks
    - Chart directory creation and management
    - Security checks on resolved paths
    - Automatic fallback to safe filenames on security violations
    """

    def __init__(self, charts_dir: Path):
        """Initialize ChartService with target directory for chart storage.

        Args:
            charts_dir: Path to directory where charts will be saved.
                       Will be created if it doesn't exist.

        Raises:
            ChartSaveError: If charts_dir path is invalid or cannot be created.
        """
        self.charts_dir = Path(charts_dir).resolve()
        try:
            self.charts_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ChartSaveError(f"Cannot create charts directory {self.charts_dir}: {e}") from e

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename to prevent path traversal and injection attacks.

        Removes dangerous characters, prevents .. / \\ paths, falls back to
        safe default name if suspicious patterns detected.

        Args:
            name: Proposed filename.

        Returns:
            Safe sanitized filename with safe extension appended if needed.
        """
        if not name:
            name = "chart.png"

        # Remove or replace dangerous chars
        sanitized = re.sub(r"[^\w\.-]", "_", name.strip())
        sanitized = re.sub(r"_+", "_", sanitized)

        # Check if sanitized result is effectively empty after stripping whitespace
        if not sanitized:
            sanitized = "chart.png"

        # Prevent path traversal
        if ".." in sanitized or "/" in sanitized or "\\" in sanitized or sanitized.startswith("."):
            sanitized = f"safe_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # Ensure safe extension
        if not sanitized.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
            sanitized += ".png"

        return sanitized

    def save_chart(self, filename: Optional[str] = None) -> tuple[str, dict]:
        """Save current matplotlib figure to disk and return path with metadata.

        Handles filename sanitization, path security validation, and
        automatic fallback to safe filenames if security checks fail.

        Args:
            filename: Optional custom filename. If None, auto-generates timestamp-based name.

        Returns:
            Tuple of (absolute_path, artifact_metadata_dict) where metadata includes:
            - id: unique identifier
            - kind: "chart"
            - path: full filesystem path
            - filename: sanitized filename
            - mime_type: "image/png"

        Raises:
            ChartSaveError: If figure cannot be saved due to I/O or security errors.
        """
        from uuid import uuid4

        # Generate or sanitize filename
        if filename is None:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        safe_name = self._sanitize_filename(filename)

        # Build target path
        target_path = self.charts_dir / safe_name

        # Security check: ensure resolved path is within charts directory
        try:
            resolved_path = target_path.resolve()
            # Use is_relative_to for proper path containment check
            try:
                resolved_path.relative_to(self.charts_dir.resolve())
            except ValueError:
                # Path is not relative to charts_dir - traversal attempt detected
                safe_name = self._sanitize_filename("fallback.png")
                target_path = self.charts_dir / safe_name
                resolved_path = target_path.resolve()
        except (OSError, RuntimeError) as e:
            raise ChartSaveError(f"Failed to resolve chart path: {e}") from e

        # Save figure
        try:
            plt.savefig(str(target_path), bbox_inches="tight", dpi=150)
            plt.close()
        except (OSError, PermissionError, ValueError) as e:
            raise ChartSaveError(f"Failed to save chart to {target_path}: {e}") from e

        # Return path and artifact metadata
        artifact_metadata = {
            "id": str(uuid4()),
            "kind": "chart",
            "path": str(target_path),
            "filename": safe_name,
            "mime_type": "image/png",
            "metadata": {},
        }
        return str(target_path), artifact_metadata
