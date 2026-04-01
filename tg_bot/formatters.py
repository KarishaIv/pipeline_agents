import html
import io
import zipfile
from pathlib import Path
from typing import Dict


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
    """Краткий итог - только счётчики. Детали - по кнопкам."""
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
