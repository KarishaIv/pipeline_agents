"""Tests for ChartService - chart saving, filename sanitization, and security."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest

from src.meta_agent.services.chart import ChartSaveError, ChartService


@pytest.fixture
def chart_service(tmp_path):
    """Create ChartService with temporary directory."""
    return ChartService(tmp_path)


def test_chart_service_init_creates_directory(tmp_path):
    """Test ChartService creates charts directory on initialization."""
    charts_dir = tmp_path / "charts"
    assert not charts_dir.exists()

    service = ChartService(charts_dir)

    assert charts_dir.exists()
    assert service.charts_dir == charts_dir.resolve()


def test_chart_service_init_existing_directory(tmp_path):
    """Test ChartService works with existing directory."""
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()

    service = ChartService(charts_dir)

    assert service.charts_dir == charts_dir.resolve()


def test_chart_service_init_invalid_path(tmp_path):
    """Test ChartService raises error on invalid path."""
    invalid_path = tmp_path / "nonexistent" / "nested" / "bad" / "charts"

    # Mock mkdir to raise permission error
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
        with pytest.raises(ChartSaveError, match="Cannot create charts directory"):
            ChartService(invalid_path)


def test_sanitize_filename_normal(chart_service):
    """Test _sanitize_filename preserves normal filenames."""
    assert chart_service._sanitize_filename("normal.png") == "normal.png"
    assert chart_service._sanitize_filename("my_chart.jpg") == "my_chart.jpg"
    assert chart_service._sanitize_filename("result.pdf") == "result.pdf"


def test_sanitize_filename_adds_extension(chart_service):
    """Test _sanitize_filename adds .png extension if missing."""
    assert chart_service._sanitize_filename("myfile").endswith(".png")
    assert chart_service._sanitize_filename("image").endswith(".png")


def test_sanitize_filename_path_traversal(chart_service):
    """Test _sanitize_filename prevents path traversal attacks."""
    result = chart_service._sanitize_filename("../../../etc/passwd.png")
    assert ".." not in result
    assert "/" not in result
    assert result.endswith(".png")

    result2 = chart_service._sanitize_filename("..\\windows\\system32.png")
    assert ".." not in result2
    assert "\\" not in result2


def test_sanitize_filename_dot_start(chart_service):
    """Test _sanitize_filename prevents hidden file creation."""
    result = chart_service._sanitize_filename(".hidden.png")
    assert not result.startswith(".")
    assert result.endswith(".png")


def test_sanitize_filename_special_chars(chart_service):
    """Test _sanitize_filename removes dangerous characters."""
    result = chart_service._sanitize_filename("chart@#$%^&().png")
    assert "@" not in result
    assert "#" not in result
    assert "$" not in result
    assert result.endswith(".png")


def test_sanitize_filename_multiple_underscores(chart_service):
    """Test _sanitize_filename collapses multiple underscores."""
    result = chart_service._sanitize_filename("bad____name.png")
    # Multiple consecutive underscores should be collapsed
    assert "____" not in result
    assert result.endswith(".png")


def test_sanitize_filename_empty_returns_default(chart_service):
    """Test _sanitize_filename returns default for empty/whitespace input."""
    assert chart_service._sanitize_filename("") == "chart.png"
    assert chart_service._sanitize_filename("   ") == "chart.png"


def test_save_chart_creates_file(chart_service):
    """Test save_chart creates file and returns path."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    path = chart_service.save_chart("test_chart.png")

    assert Path(path).exists()
    assert "test_chart.png" in path


def test_save_chart_auto_filename(chart_service):
    """Test save_chart generates filename when None provided."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    path = chart_service.save_chart(None)

    assert Path(path).exists()
    assert path.endswith(".png")


def test_save_chart_with_timestamp_default(chart_service):
    """Test save_chart generates timestamp-based name by default."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    path1 = chart_service.save_chart()
    path2 = chart_service.save_chart()

    # Should create different files with different timestamps
    assert Path(path1).exists()
    assert Path(path2).exists()
    assert path1 != path2


def test_save_chart_closes_figure(chart_service):
    """Test save_chart closes matplotlib figure after saving."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    fig_num = fig.number

    chart_service.save_chart("test.png")

    # Figure should be closed
    assert fig_num not in plt.get_fignums()


def test_save_chart_within_charts_dir(chart_service):
    """Test save_chart file is saved within charts directory."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    path = chart_service.save_chart("chart.png")

    resolved_chart_path = Path(path).resolve()
    resolved_charts_dir = chart_service.charts_dir.resolve()

    assert str(resolved_charts_dir) in str(resolved_chart_path)


def test_save_chart_fallback_on_path_traversal_attempt(chart_service):
    """Test save_chart uses fallback name when path traversal detected."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    # Attempt path traversal (should be sanitized to safe name)
    path = chart_service.save_chart("../../../evil.png")

    # Should save with fallback safe name instead
    assert "evil" not in path or "fallback" in path
    assert Path(path).exists()


def test_save_chart_raises_on_io_error(chart_service):
    """Test save_chart raises ChartSaveError on I/O failure."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    # Mock savefig to raise error
    with patch("matplotlib.pyplot.savefig", side_effect=OSError("Disk full")):
        with pytest.raises(ChartSaveError, match="Failed to save chart"):
            chart_service.save_chart("test.png")


def test_save_chart_permission_denied(chart_service):
    """Test save_chart handles permission errors gracefully."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])

    # Mock savefig to raise permission error
    with patch("matplotlib.pyplot.savefig", side_effect=PermissionError("Access denied")):
        with pytest.raises(ChartSaveError, match="Failed to save chart"):
            chart_service.save_chart("test.png")


def test_save_chart_multiple_formats(chart_service):
    """Test save_chart works with multiple file formats."""
    for ext in ["png", "jpg", "pdf"]:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        path = chart_service.save_chart(f"chart.{ext}")

        assert Path(path).exists()
        assert path.endswith(ext)
