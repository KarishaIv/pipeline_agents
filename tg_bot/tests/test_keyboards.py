from aiogram.types import InlineKeyboardMarkup

from keyboards import keyboard_confirm_ta, keyboard_result, keyboard_select_count, keyboard_select_ratio


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
