"""
Юнит-тесты для Telegram-бота.
Запуск: pytest tests/test_bot.py -v
"""

import io
import json
import zipfile
from pathlib import Path

import pytest
from aiogram.types import InlineKeyboardMarkup

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot import (
    distribute_personas,
    _extract_json,
    format_result_simple,
    format_reasoning_message,
    format_news_message,
    keyboard_confirm_ta,
    keyboard_select_count,
    keyboard_select_ratio,
    keyboard_result,
    _make_archive,
    _format_news_block,
)

# Тесты для распределения людей по аудиториям

def test_distribute_personas_equal_ratio():
    counts = distribute_personas(10, [1, 1])
    assert sum(counts) == 10
    assert counts[0] == counts[1]


def test_distribute_personas_3_to_1():
    counts = distribute_personas(20, [3, 1])
    assert sum(counts) == 20
    assert counts[0] > counts[1]


def test_distribute_personas_single_audience():
    counts = distribute_personas(15, [1])
    assert counts == [15]


def test_distribute_personas_minimum_one_per_audience():
    counts = distribute_personas(3, [10, 10, 10])
    assert all(c >= 1 for c in counts)
    assert sum(counts) == 3


def test_distribute_personas_three_audiences():
    counts = distribute_personas(20, [2, 1, 1])
    assert sum(counts) == 20
    assert counts[0] >= counts[1]
    assert counts[0] >= counts[2]


# Тесты для проверки формата json

def test_extract_json_plain():
    text = '{"audiences": ["пенсионеры"], "question": "возьмут кредит?", "ratios": [1]}'
    result = _extract_json(text)
    assert result["audiences"] == ["пенсионеры"]
    assert result["ratios"] == [1]


def test_extract_json_with_markdown_wrapper():
    text = '```json\n{"audiences": ["студенты"], "question": "откроют вклад?", "ratios": [1]}\n```'
    result = _extract_json(text)
    assert result["audiences"] == ["студенты"]


def test_extract_json_with_extra_text():
    text = 'Вот результат:\n{"audiences": ["врачи"], "question": "возьмут ипотеку?", "ratios": [1]}'
    result = _extract_json(text)
    assert result["audiences"] == ["врачи"]


def test_extract_json_invalid_returns_empty():
    result = _extract_json("не валидный json вообще")
    assert result == {}


def test_extract_json_empty_string():
    result = _extract_json("")
    assert result == {}


# Тесты на работу клавиатур

def test_keyboard_confirm_ta_structure():
    kb = keyboard_confirm_ta()
    assert isinstance(kb, InlineKeyboardMarkup)
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    cb_data = [b.callback_data for b in buttons]
    assert "confirm_ta" in cb_data
    assert "new_query" in cb_data


def test_keyboard_select_count_has_four_options():
    kb = keyboard_select_count()
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    cb_data = [b.callback_data for b in buttons]
    assert "count_5" in cb_data
    assert "count_10" in cb_data
    assert "count_15" in cb_data
    assert "count_20" in cb_data


def test_keyboard_select_ratio_two_audiences():
    kb = keyboard_select_ratio(2)
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    cb_data = [b.callback_data for b in buttons]
    assert "ratio_equal" in cb_data
    assert "ratio_2_1_first" in cb_data
    assert "ratio_3_1_first" in cb_data


def test_keyboard_select_ratio_three_audiences():
    kb = keyboard_select_ratio(3)
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    cb_data = [b.callback_data for b in buttons]
    assert "ratio_equal" in cb_data
    assert "ratio_2_1_first" in cb_data
    assert "ratio_3_1_first" in cb_data


def test_keyboard_result_has_all_buttons():
    kb = keyboard_result()
    buttons = [btn for row in kb.inline_keyboard for btn in row]
    cb_data = [b.callback_data for b in buttons]
    assert "show_reasoning" in cb_data
    assert "show_news" in cb_data
    assert "download_archive" in cb_data
    assert "new_query" in cb_data


# Тесты на формат результата

def _make_state(decision: bool, confidence: float, ta: str = "пенсионеры"):
    return {
        "audiences": [ta],
        "counts": [5],
        "question": "возьмут кредит?",
        "result": {
            "results": [
                {
                    "profile": {"target_audience_name": ta},
                    "survey_responses": [{
                        "full_state": {
                            "final_decision": {
                                "decision": decision,
                                "confidence": confidence,
                                "reasoning": "потому что так"
                            }
                        }
                    }]
                }
            ]
        }
    }


def test_format_result_simple_agree():
    state = _make_state(True, 0.9)
    text = format_result_simple(state)
    assert "ДА" in text
    assert "возьмут кредит?" in text


def test_format_result_simple_disagree():
    state = _make_state(False, 0.3)
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


# Тесты на создание архива

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
