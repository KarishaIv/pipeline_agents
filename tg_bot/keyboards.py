from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def keyboard_confirm_ta() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_ta"),
        InlineKeyboardButton(text="✏️ Изменить",  callback_data="new_query"),
    ]])


def keyboard_select_count() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="5 персон",  callback_data="count_5"),
            InlineKeyboardButton(text="10 персон", callback_data="count_10"),
        ],
        [
            InlineKeyboardButton(text="15 персон", callback_data="count_15"),
            InlineKeyboardButton(text="20 персон", callback_data="count_20"),
        ],
    ])


def keyboard_select_ratio(n: int) -> InlineKeyboardMarkup:
    equal_label = ":".join(["1"] * n)
    rows = [
        [InlineKeyboardButton(text=f"Поровну ({equal_label})", callback_data="ratio_equal")],
        [InlineKeyboardButton(text="2:1 (первая больше)",       callback_data="ratio_2_1_first")],
        [InlineKeyboardButton(text="3:1 (первая больше)",       callback_data="ratio_3_1_first")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def keyboard_result() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="📝 Ход рассуждений",    callback_data="show_reasoning"),
            InlineKeyboardButton(text="📰 Новостной контекст", callback_data="show_news"),
        ],
        [
            InlineKeyboardButton(text="📦 Скачать архив", callback_data="download_archive"),
            InlineKeyboardButton(text="🔁 Новый запрос",  callback_data="new_query"),
        ],
    ])
