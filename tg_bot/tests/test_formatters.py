import io
import zipfile

from formatters import _format_news_block, _make_archive, format_news_message, format_reasoning_message, format_result_simple

from conftest import make_state


def test_format_result_simple_agree():
    state = make_state(True, 0.9)
    text = format_result_simple(state)
    assert "ДА" in text
    assert "возьмут кредит?" in text


def test_format_result_simple_disagree():
    state = make_state(False, 0.3)
    text = format_result_simple(state)
    assert "НЕТ" in text


def test_format_result_simple_no_results():
    state = {
        "audiences": ["врачи"],
        "counts": [5],
        "question": "вопрос",
        "result": {"results": []}
    }
    text = format_result_simple(state)
    assert "не дала результатов" in text


def test_format_news_block_full():
    ctx = {
        "overall_reaction": "настороженный",
        "impact_horizon": "short_term",
        "factors": ["рост ставки", "инфляция"],
        "summary_text": "Рынок ожидает повышения ключевой ставки."
    }
    result = _format_news_block(ctx)
    assert "настороженный" in result
    assert "рост ставки" in result
    assert "<i>" in result


def test_format_news_block_empty_dict():
    assert _format_news_block({}) == ""


def test_format_news_block_none():
    assert _format_news_block(None) == ""


def test_make_archive_creates_valid_zip(tmp_path):
    (tmp_path / "result.json").write_text('{"ok": true}')
    (tmp_path / "summary.csv").write_text("a,b\n1,2")

    zip_bytes = _make_archive(str(tmp_path))
    assert len(zip_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
    assert "result.json" in names
    assert "summary.csv" in names


def test_make_archive_empty_dir_returns_valid_zip(tmp_path):
    zip_bytes = _make_archive(str(tmp_path))
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert zf.namelist() == []
