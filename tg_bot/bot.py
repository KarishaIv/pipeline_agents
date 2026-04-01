"""
Telegram-бот: произвольный запрос - парсинг через YandexGPT - новостной контекст - мультиагентная симуляция - ответ пользователю.

Переменные окружения =:
  TG_BOT_TOKEN        — токен бота
  YANDEX_FOLDER_ID    — ID каталога в Yandex Cloud
  YANDEX_API_KEY      — API-ключ YandexGPT
  PIPELINE_PATH       — путь к папке с базовым пайплайном
  NEWS_SYSTEM_PATH    — путь к папке с агентом новостного контекста 
"""

import asyncio
import html
import io
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)


PIPELINE_PATH = os.getenv("PIPELINE_PATH",    "../pipeline 4")
NEWS_SYSTEM_PATH = os.getenv("NEWS_SYSTEM_PATH", "../pipeline_agents-multiagent_system_for_context/multi_agent_rag")

_abs_pipeline = os.path.abspath(PIPELINE_PATH)
if _abs_pipeline not in sys.path:
    sys.path.insert(0, _abs_pipeline)


TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN",    "YOUR_BOT_TOKEN")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID","YOUR_FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY",  "YOUR_API_KEY")
YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"
ANALYSIS_TIMEOUT = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

router = Router()


_chat_state: Dict[int, dict] = {}

_news_enricher: Optional[object] = None
_pipeline_initialized = False


def _init_pipeline():
    global _pipeline_initialized
    if _pipeline_initialized:
        return
    from config import set_yandex_config
    set_yandex_config(YANDEX_FOLDER_ID, YANDEX_API_KEY)
    _pipeline_initialized = True


async def _get_news_enricher():
    global _news_enricher
    if _news_enricher is None:
        from src.news_enricher import NewsContextEnricher
        _news_enricher = NewsContextEnricher(
            news_system_path=os.path.abspath(NEWS_SYSTEM_PATH),
            iam_token=YANDEX_API_KEY,
        )
        await _news_enricher.initialize()
    return _news_enricher



SYSTEM_PROMPT = dedent("""\
    Ты — ассистент-аналитик. Пользователь присылает запрос о финансовом поведении людей.
    Извлеки целевые аудитории, вопрос и соотношение. Верни ТОЛЬКО валидный JSON без пояснений:
    {
      "audiences": ["<ЦА 1>", "<ЦА 2>"],
      "question": "<суть финансового вопроса кратко>",
      "ratios": [<число>, <число>]
    }

    Правила для audiences:
    - Список реальных групп людей (любое количество)
    - Если одна группа — список из одного элемента
    - Если субъект вымышленный — пустой список []
    - Если субъект бессмысленный — пустой список []
    - Если ЦА не упомянута вообще — пустой список []

    Правила для ratios:
    - Если упомянуто соотношение (3 к 1, 2:1 и т.д.) — отрази в ratios
    - Если соотношение не указано — все значения равны 1

    Если вопрос не про финансы — question: "не финансовый вопрос"
""")

FALLBACK_TA_PROMPT = dedent("""\
    Пользователь задал вопрос о финансовом поведении, но целевая аудитория не была явно указана.
    Попробуй предположить наиболее вероятную реальную группу людей для этого вопроса.

    Верни ТОЛЬКО валидный JSON:
    {
      "target_audience": "<реальная группа людей>",
      "inferred": true
    }

    Правила:
    - Если субъект вымышленный — верни {"target_audience": "не удалось", "inferred": false}
    - Если субъект бессмысленный — верни {"target_audience": "не удалось", "inferred": false}
    - Если вопрос совсем не про финансы — верни {"target_audience": "не удалось", "inferred": false}
    - Иначе — предложи разумное обобщение: "работающие граждане России", "жители России старше 18 лет" и т.д.
""")

DEMOGRAPHICS_PROMPT = dedent("""\
    Из описания целевой аудитории извлеки демографические параметры типичного представителя.
    Верни ТОЛЬКО валидный JSON без пояснений:
    {
      "age_group": <число — типичный возраст>,
      "gender": "<Мужской|Женский>",
      "marital_status": "<Женат|Замужем|Не женат|Не замужем|Разведен|Разведена|Вдовец|Вдова>",
      "education": "<среднего профессионального образования|бакалавриат|специалитет|магистратура|аспирантура>"
    }
    Если параметр неизвестен — используй наиболее вероятное значение для России.
""")



async def _yandex_call(messages: list, temperature: float = 0.1, max_tokens: int = 500) -> str:
    payload = {
        "modelUri": YANDEX_MODEL_URI,
        "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": max_tokens},
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(YANDEX_GPT_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"YandexGPT {resp.status}: {body}")
            data = await resp.json()
            return data["result"]["alternatives"][0]["message"]["text"]


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1:
        return {}
    try:
        return json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return {}


async def parse_query_with_llm(user_text: str) -> dict:
    try:
        raw = await _yandex_call([
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user",   "text": user_text},
        ])
        return _extract_json(raw)
    except Exception as exc:
        log.error("parse_query_with_llm: %s", exc)
        return {"error": str(exc)}


async def infer_ta_from_query(user_text: str) -> dict:
    try:
        raw = await _yandex_call([
            {"role": "system", "text": FALLBACK_TA_PROMPT},
            {"role": "user",   "text": user_text},
        ], temperature=0.2)
        return _extract_json(raw)
    except Exception as exc:
        log.warning("infer_ta_from_query: %s", exc)
    return {"target_audience": "не удалось", "inferred": False}


async def extract_demographics_with_llm(target_audience: str) -> dict:
    try:
        raw = await _yandex_call([
            {"role": "system", "text": DEMOGRAPHICS_PROMPT},
            {"role": "user",   "text": target_audience},
        ], max_tokens=200)
        return _extract_json(raw)
    except Exception as exc:
        log.warning("extract_demographics_with_llm: %s", exc)
    return {}



def distribute_personas(total: int, ratios: List[int]) -> List[int]:
    """Распределяет total персон пропорционально ratios."""
    total_ratio = sum(ratios)
    counts = [max(1, round(total * r / total_ratio)) for r in ratios]
    diff = total - sum(counts)
    counts[0] += diff
    return counts


async def run_full_analysis(
    audiences: List[str],
    counts: List[int],
    question: str,
) -> dict:
    _init_pipeline()
    from src.orchestration import PipelineRunner

    news_contexts: Dict[str, dict] = {}
    evidence_list = []

    enricher = None
    try:
        enricher = await _get_news_enricher()
    except Exception as exc:
        log.warning("news enricher недоступен: %s", exc)

    for ta, count in zip(audiences, counts):
        news_ctx = {}
        if enricher:
            try:
                news_ctx = await enricher.get_news_context(question, ta)
                log.info("Новостной контекст для '%s' получен", ta)
            except Exception as exc:
                log.warning("Новостной контекст для '%s': %s", ta, exc)
        news_contexts[ta] = news_ctx

        demographics = await extract_demographics_with_llm(ta)
        log.info("Демография '%s': %s", ta, demographics)

        evidence_list.append({
            "target_audience_name": ta,
            "synthetic_size":       count,
            "news_question":        question,
            **demographics,
        })

    out_dir = os.path.join(_abs_pipeline, "outputs", "bot_runs")
    pipeline_config = {
        "evidence":         evidence_list,
        "survey_questions": [question],
        "nemo_size":        5000,
        "output":           out_dir,
        "simulation_steps": 1,
        "concurrency":      5,
        "timeout":          90.0,
        "ta_concurrency":   1,
        "agent_mode":       "survey",
        "use_pgm":          True,
        "ocean_flag":       True,
    }

    runner = PipelineRunner(pipeline_config, news_enricher=_news_enricher)
    results = await runner.run()

    return {
        "results":       results,
        "news_contexts": news_contexts,
        "question":      question,
        "audiences":     audiences,
        "counts":        counts,
        "out_dir":       out_dir,
    }



def _format_news_block(news_ctx: dict) -> str:
    if not isinstance(news_ctx, dict) or not news_ctx:
        return ""
    lines = []
    if news_ctx.get("overall_reaction"):
        lines.append(f"Реакция: {news_ctx['overall_reaction']} (горизонт: {news_ctx.get('impact_horizon', '?')})")
    if news_ctx.get("factors"):
        lines.append("Факторы: " + "; ".join(news_ctx["factors"]))
    if news_ctx.get("summary_text"):
        lines.append(news_ctx["summary_text"])
    if not lines:
        return ""
    return "<i>" + html.escape("\n".join(lines)) + "</i>"


def format_result_simple(state: dict) -> str:
    """Краткий итог — только счётчики. Детали — по кнопкам."""
    results_list = state.get("result", {}).get("results", [])
    audiences = state.get("audiences", [])
    counts = state.get("counts", [])
    question = state.get("question", "")

    agreed, total, confidences = 0, 0, []

    for r in results_list:
        if "error" in r:
            continue
        responses = r.get("survey_responses", [])
        if not responses:
            continue
        full_state = responses[0].get("full_state", {})
        final = full_state.get("final_decision") or {}
        if not isinstance(final, dict):
            continue
        decision   = final.get("decision")
        confidence = final.get("confidence", 0.0)
        total += 1
        if decision is True:
            agreed += 1
        if confidence:
            confidences.append(confidence)

    if total == 0:
        return (
            f"<b>Вопрос:</b> {html.escape(question)}\n"
            f"<b>ЦА:</b> {html.escape(', '.join(audiences))}\n\n"
            "❌ Симуляция не дала результатов."
        )

    agree_pct = int(agreed / total * 100)
    avg_conf_pct = int(sum(confidences) / len(confidences) * 100) if confidences else 0

    ta_parts = []
    for ta, cnt in zip(audiences, counts):
        ta_parts.append(f"{html.escape(ta)} ({cnt} пер.)")
    ta_line = " + ".join(ta_parts)

    return (
        f"<b>Вопрос:</b> {html.escape(question)}\n"
        f"<b>ЦА:</b> {ta_line}\n\n"
        f"<b>Результат ({total} персон):</b>\n"
        f"✅ ДА: {agreed} ({agree_pct}%)\n"
        f"❌ НЕТ: {total - agreed} ({100 - agree_pct}%)\n"
        f"Средняя уверенность: {avg_conf_pct}%"
    )


def format_reasoning_message(state: dict) -> str:
    """Примеры рассуждений — один YES и один NO на каждую ЦА."""
    results_list = state.get("result", {}).get("results", [])
    audiences = state.get("audiences", [])
    samples: Dict[str, dict] = {ta: {"yes": None, "no": None} for ta in audiences}

    for r in results_list:
        if "error" in r:
            continue
        profile = r.get("profile", {})
        ta = profile.get("target_audience_name", audiences[0] if audiences else "")
        responses = r.get("survey_responses", [])
        if not responses:
            continue
        full_state = responses[0].get("full_state", {})
        final = full_state.get("final_decision") or {}
        if not isinstance(final, dict):
            continue

        decision = final.get("decision")
        reasoning = final.get("reasoning", "")
        if not reasoning:
            continue

        age = profile.get("age_group", "?")
        region = profile.get("region", "")
        income = profile.get("income_level", "")
        edu = profile.get("education", "")
        parts = [p for p in [str(age) + " л.", region, income, edu] if p and p != "не указано"]
        label = "<i>" + html.escape(", ".join(parts)) + "</i>"
        text = f"{label}\n{html.escape(reasoning)}"

        if ta in samples:
            if decision is True and samples[ta]["yes"] is None:
                samples[ta]["yes"] = "✅ " + text
            elif decision is False and samples[ta]["no"] is None:
                samples[ta]["no"] = "❌ " + text

    lines = ["<b>📝 Примеры рассуждений агентов:</b>"]
    for ta, s in samples.items():
        if s["yes"] or s["no"]:
            lines.append(f"\n<b>{html.escape(ta)}:</b>")
            if s["yes"]:
                lines.append(s["yes"])
            if s["no"]:
                lines.append(s["no"])

    if len(lines) == 1:
        return "Рассуждения недоступны."
    return "\n\n".join(lines)


def format_news_message(state: dict) -> str:
    """Новостной контекст по каждой ЦА."""
    news_contexts = state.get("result", {}).get("news_contexts", {})
    audiences = state.get("audiences", [])

    if not news_contexts:
        return "Новостной контекст недоступен."

    lines = ["<b>📰 Новостной фон:</b>"]
    for ta in audiences:
        ctx = news_contexts.get(ta, {})
        block = _format_news_block(ctx)
        if block:
            lines.append(f"\n<b>{html.escape(ta)}:</b>\n{block}")

    if len(lines) == 1:
        return "Новостной контекст пуст."
    return "\n".join(lines)


def _make_archive(out_dir: str) -> bytes:
    """Создаёт zip-архив из директории результатов."""
    buf = io.BytesIO()
    path = Path(out_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in path.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(path))
    buf.seek(0)
    return buf.read()



def keyboard_confirm_ta() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Всё верно", callback_data="confirm_ta"),
        InlineKeyboardButton(text="✏️ Изменить",              callback_data="new_query"),
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
            InlineKeyboardButton(text="📦 Скачать архив",  callback_data="download_archive"),
            InlineKeyboardButton(text="🔁 Новый запрос",   callback_data="new_query"),
        ],
    ])



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

    # Если ЦА не найдена — пробуем fallback
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



async def main() -> None:
    bot = Bot(token=TG_BOT_TOKEN)
    dp  = Dispatcher()
    dp.include_router(router)
    log.info("Bot started — polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
