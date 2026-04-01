import asyncio
import html
import logging
from pathlib import Path
from typing import Dict, List

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from formatters import _make_archive, format_news_message, format_reasoning_message, format_result_simple
from keyboards import keyboard_confirm_ta, keyboard_result, keyboard_select_count, keyboard_select_ratio
from llm import infer_ta_from_query, parse_query_with_llm
from pipeline import distribute_personas, run_full_analysis
from config import ANALYSIS_TIMEOUT

log = logging.getLogger(__name__)
router = Router()

_chat_state: Dict[int, dict] = {}


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Отправь запрос — я выделю целевые аудитории и вопрос, "
        "затем запущу анализ через новостной агент и мультиагентную симуляцию.\n\n"
        "Можно указать несколько ЦА и соотношение:\n"
        "<i>как поведут себя программисты и визажистки в соотношении 3 к 1 при росте ставки?</i>",
        parse_mode="HTML",
    )


@router.message(F.text)
async def handle_user_query(message: Message) -> None:
    user_text = message.text.strip()
    if not user_text:
        return

    wait_msg = await message.answer("Анализирую запрос…")
    parsed = await parse_query_with_llm(user_text)

    if "error" in parsed:
        await wait_msg.edit_text(f"Ошибка парсинга: {parsed['error']}")
        return

    audiences = parsed.get("audiences", [])
    ratios = parsed.get("ratios", [1] * max(len(audiences), 1))
    question = parsed.get("question", "не определён")

    ta_inferred = False
    if not audiences:
        await wait_msg.edit_text("Уточняю целевую аудиторию…")
        fallback = await infer_ta_from_query(user_text)
        inferred = fallback.get("target_audience", "не удалось")
        if inferred and inferred not in ("не удалось", "не указана", ""):
            audiences = [inferred]
            ratios = [1]
            ta_inferred = True
        else:
            await wait_msg.edit_text(
                "Не удалось определить целевую аудиторию.\n\n"
                "Укажи конкретную группу людей.\n\n"
                "<b>Примеры:</b>\n"
                "• <i>будут ли пенсионеры брать кредит в этом году?</i>\n"
                "• <i>возьмёт ли ипотеку мужчина 35 лет с зарплатой 150к?</i>\n"
                "• <i>снимут ли семьи с детьми деньги со вкладов?</i>",
                parse_mode="HTML",
            )
            return

    _chat_state[message.chat.id] = {
        "step":      "confirm",
        "raw_query": user_text,
        "audiences": audiences,
        "ratios":    ratios,
        "question":  question,
        "total":     None,
        "counts":    [],
        "result":    None,
    }

    inferred_note = "\n<i>⚠️ ЦА выведена автоматически из контекста</i>" if ta_inferred else ""
    ta_lines = "\n".join(f"  {i+1}. {html.escape(ta)}" for i, ta in enumerate(audiences))
    ratio_line = " : ".join(str(r) for r in ratios)

    await wait_msg.edit_text(
        f"<b>Запрос разобран</b>{inferred_note}\n\n"
        f"<b>Целевые аудитории:</b>\n{ta_lines}\n"
        f"<b>Соотношение:</b> {ratio_line}\n"
        f"<b>Вопрос:</b> {html.escape(question)}\n\n"
        "Всё верно?",
        parse_mode="HTML",
        reply_markup=keyboard_confirm_ta(),
    )


@router.callback_query(F.data == "confirm_ta")
async def on_confirm_ta(callback: CallbackQuery) -> None:
    state = _chat_state.get(callback.message.chat.id)
    if not state:
        await callback.answer("Контекст потерян, отправь запрос заново.", show_alert=True)
        return
    await callback.answer()
    state["step"] = "set_count"
    await callback.message.answer(
        "Сколько синтетических персон сгенерировать? (максимум 20)",
        reply_markup=keyboard_select_count(),
    )


async def _handle_count_selected(callback: CallbackQuery, total: int) -> None:
    state = _chat_state.get(callback.message.chat.id)
    if not state:
        await callback.answer("Контекст потерян.", show_alert=True)
        return
    await callback.answer()
    state["total"] = total

    n = len(state["audiences"])
    ratios = state["ratios"]
    ratio_already_set = n <= 1 or len(set(ratios)) > 1

    if not ratio_already_set:
        state["step"] = "set_ratio"
        ta_line = " / ".join(html.escape(ta) for ta in state["audiences"])
        await callback.message.answer(
            f"Выбери соотношение аудиторий:\n<i>{ta_line}</i>",
            parse_mode="HTML",
            reply_markup=keyboard_select_ratio(n),
        )
    else:
        await _start_analysis(callback)


async def _apply_ratio(callback: CallbackQuery, ratios: List[int]) -> None:
    state = _chat_state.get(callback.message.chat.id)
    if not state:
        await callback.answer("Контекст потерян.", show_alert=True)
        return
    await callback.answer()
    state["ratios"] = ratios
    await _start_analysis(callback)


async def _start_analysis(callback: CallbackQuery) -> None:
    chat_id = callback.message.chat.id
    state   = _chat_state.get(chat_id)
    if not state:
        return

    state["step"] = "running"
    audiences = state["audiences"]
    ratios    = state["ratios"]
    total     = state["total"]
    question  = state["question"]
    counts    = distribute_personas(total, ratios)
    state["counts"] = counts

    status_msg = await callback.message.answer(
        "⏳ <b>Шаг 1/3</b> — получаю новостной контекст…",
        parse_mode="HTML",
    )

    try:
        async def _run():
            await status_msg.edit_text(
                "⏳ <b>Шаг 2/3</b> — генерирую персоны (PGM → GMM → OCEAN)…\n"
                "<i>(занимает 2–5 минут)</i>",
                parse_mode="HTML",
            )
            result = await run_full_analysis(audiences, counts, question)
            await status_msg.edit_text(
                "⏳ <b>Шаг 3/3</b> — симуляция завершена, формирую ответ…",
                parse_mode="HTML",
            )
            return result

        result = await asyncio.wait_for(_run(), timeout=ANALYSIS_TIMEOUT)

    except asyncio.TimeoutError:
        await status_msg.edit_text(
            f"⏰ Анализ превысил {ANALYSIS_TIMEOUT} сек и был прерван. Попробуй снова."
        )
        return
    except Exception as exc:
        log.exception("Ошибка анализа")
        await status_msg.edit_text(f"❌ Ошибка анализа: {exc}")
        return

    await status_msg.delete()
    state["result"] = result
    state["step"]   = "done"

    answer = format_result_simple(state)
    if len(answer) > 4000:
        answer = answer[:4000] + "\n\n<i>… текст обрезан</i>"

    await callback.message.answer(answer, parse_mode="HTML", reply_markup=keyboard_result())


@router.callback_query(F.data == "count_5")
async def on_count_5(cb: CallbackQuery):  await _handle_count_selected(cb, 5)

@router.callback_query(F.data == "count_10")
async def on_count_10(cb: CallbackQuery): await _handle_count_selected(cb, 10)

@router.callback_query(F.data == "count_15")
async def on_count_15(cb: CallbackQuery): await _handle_count_selected(cb, 15)

@router.callback_query(F.data == "count_20")
async def on_count_20(cb: CallbackQuery): await _handle_count_selected(cb, 20)


@router.callback_query(F.data == "ratio_equal")
async def on_ratio_equal(cb: CallbackQuery):
    n = len(_chat_state.get(cb.message.chat.id, {}).get("audiences", [1]))
    await _apply_ratio(cb, [1] * n)

@router.callback_query(F.data == "ratio_2_1_first")
async def on_ratio_2_1_first(cb: CallbackQuery):
    n = len(_chat_state.get(cb.message.chat.id, {}).get("audiences", [1]))
    await _apply_ratio(cb, [2] + [1] * (n - 1))

@router.callback_query(F.data == "ratio_3_1_first")
async def on_ratio_3_1_first(cb: CallbackQuery):
    n = len(_chat_state.get(cb.message.chat.id, {}).get("audiences", [1]))
    await _apply_ratio(cb, [3] + [1] * (n - 1))


@router.callback_query(F.data == "show_reasoning")
async def on_show_reasoning(callback: CallbackQuery) -> None:
    state = _chat_state.get(callback.message.chat.id)
    await callback.answer()
    if not state or not state.get("result"):
        await callback.message.answer("Результаты недоступны.")
        return
    text = format_reasoning_message(state)
    if len(text) > 4000:
        text = text[:4000] + "\n\n<i>… текст обрезан</i>"
    await callback.message.answer(text, parse_mode="HTML")


@router.callback_query(F.data == "show_news")
async def on_show_news(callback: CallbackQuery) -> None:
    state = _chat_state.get(callback.message.chat.id)
    await callback.answer()
    if not state or not state.get("result"):
        await callback.message.answer("Новостной контекст недоступен.")
        return
    text = format_news_message(state)
    if len(text) > 4000:
        text = text[:4000] + "\n\n<i>… текст обрезан</i>"
    await callback.message.answer(text, parse_mode="HTML")


@router.callback_query(F.data == "download_archive")
async def on_download_archive(callback: CallbackQuery) -> None:
    state = _chat_state.get(callback.message.chat.id)
    await callback.answer()
    if not state or not state.get("result"):
        await callback.message.answer("Архив недоступен.")
        return
    out_dir = state["result"].get("out_dir", "")
    if not out_dir or not Path(out_dir).exists():
        await callback.message.answer("Директория результатов не найдена.")
        return
    try:
        zip_bytes = _make_archive(out_dir)
        file = BufferedInputFile(zip_bytes, filename="results.zip")
        await callback.message.answer_document(file, caption="📦 Архив результатов симуляции")
    except Exception as exc:
        log.exception("Ошибка создания архива")
        await callback.message.answer(f"Ошибка создания архива: {exc}")


@router.callback_query(F.data == "new_query")
async def on_new_query(callback: CallbackQuery) -> None:
    await callback.answer()
    _chat_state.pop(callback.message.chat.id, None)
    await callback.message.answer("Отправь новый запрос:")
