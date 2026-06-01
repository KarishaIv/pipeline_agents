"""Tests for ArtifactService: artifact saving, filename sanitization, and metadata."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.meta_agent.services.artifact import ArtifactSaveError, ArtifactService


@pytest.fixture
def artifact_service(tmp_path):
    """Create ArtifactService with a temporary directory."""
    return ArtifactService(tmp_path)


def test_artifact_service_init_creates_directory(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    assert not artifacts_dir.exists()

    service = ArtifactService(artifacts_dir)

    assert artifacts_dir.exists()
    assert service.artifacts_dir == artifacts_dir.resolve()


def test_artifact_service_init_invalid_path(tmp_path):
    invalid_path = tmp_path / "nonexistent" / "nested" / "artifacts"

    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
        with pytest.raises(ArtifactSaveError, match="Cannot create artifacts directory"):
            ArtifactService(invalid_path)


def test_sanitize_filename_by_artifact_type(artifact_service):
    assert artifact_service.sanitize_filename("normal.png", default_extension=".png") == "normal.png"
    assert artifact_service.sanitize_filename("report", default_extension=".json").endswith(".json")
    assert artifact_service.sanitize_filename("table", default_extension=".csv").endswith(".csv")
    assert artifact_service.sanitize_filename("report.csv", default_extension=".json").endswith(".json")
    assert artifact_service.sanitize_filename("table.json", default_extension=".csv").endswith(".csv")


def test_sanitize_filename_blocks_traversal_and_hidden_files(artifact_service):
    for unsafe_name in ("../../../etc/passwd.json", "..\\windows\\system32.csv", ".hidden.png"):
        result = artifact_service.sanitize_filename(unsafe_name, default_extension=".json")
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result
        assert not result.startswith(".")


def test_save_chart_creates_chart_artifact(artifact_service):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    fig_num = fig.number

    path, metadata = artifact_service.save_chart("test_chart.png", metadata={"chart_type": "line"})

    assert Path(path).exists()
    assert fig_num not in plt.get_fignums()
    assert metadata["kind"] == "chart"
    assert metadata["mime_type"] == "image/png"
    assert metadata["metadata"]["chart_type"] == "line"


def test_save_chart_raises_on_io_error(artifact_service):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    with patch("matplotlib.pyplot.savefig", side_effect=OSError("Disk full")):
        with pytest.raises(ArtifactSaveError, match="Failed to save chart"):
            artifact_service.save_chart("test.png")


def test_save_json_creates_data_artifact(artifact_service):
    path, metadata = artifact_service.save_json(
        [{"name": "Alice", "score": 10}],
        "scores.json",
        caption="Raw scores",
        metadata={"source": "unit_test"},
    )

    assert Path(path).exists()
    assert json.loads(Path(path).read_text(encoding="utf-8")) == [{"name": "Alice", "score": 10}]
    assert metadata["kind"] == "data"
    assert metadata["mime_type"] == "application/json"
    assert metadata["caption"] == "Raw scores"
    assert metadata["metadata"]["source"] == "unit_test"


def test_save_json_overrides_wrong_extension(artifact_service):
    path, metadata = artifact_service.save_json({"ok": True}, "raw.csv")

    assert Path(path).suffix == ".json"
    assert metadata["kind"] == "data"
    assert metadata["mime_type"] == "application/json"


def test_save_json_accepts_dataframe(artifact_service):
    path, metadata = artifact_service.save_json(pd.DataFrame([{"x": 1}, {"x": 2}]), "frame.json")

    assert json.loads(Path(path).read_text(encoding="utf-8")) == [{"x": 1}, {"x": 2}]
    assert metadata["kind"] == "data"


def test_save_csv_creates_csv_artifact(artifact_service):
    path, metadata = artifact_service.save_csv(
        [{"name": "Alice", "score": 10}],
        "scores.csv",
        metadata={"source": "unit_test"},
    )

    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows == [{"name": "Alice", "score": "10"}]
    assert metadata["kind"] == "csv"
    assert metadata["mime_type"] == "text/csv"
    assert metadata["metadata"]["source"] == "unit_test"


def test_save_csv_overrides_wrong_extension(artifact_service):
    path, metadata = artifact_service.save_csv([{"ok": True}], "raw.json")

    assert Path(path).suffix == ".csv"
    assert metadata["kind"] == "csv"
    assert metadata["mime_type"] == "text/csv"


def test_artifact_from_existing_file_infers_kind_and_mime(artifact_service):
    existing_file = artifact_service.artifacts_dir / "raw.json"
    existing_file.write_text('{"ok": true}', encoding="utf-8")

    metadata = artifact_service.artifact_from_existing_file(
        existing_file,
        caption="Existing JSON",
        metadata={"source": "code_execution"},
    )

    assert metadata["kind"] == "data"
    assert metadata["mime_type"] == "application/json"
    assert metadata["filename"] == "raw.json"
    assert metadata["metadata"]["source"] == "code_execution"
