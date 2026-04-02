import html
import io
import zipfile
from pathlib import Path
from typing import Dict


def _format_news_block(news_ctx: dict) -> str:
    if not isinstance(news_ctx, dict) or not news_ctx:
        return ""
    lines = []
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

        text = html.escape(reasoning)

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


def _make_archive(out_dir: str, news_contexts: dict = None) -> bytes:
    """Создаёт zip-архив из директории результатов + папка news/ с топ-3 новостями по темам."""
    buf = io.BytesIO()
    path = Path(out_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in path.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(path))

        if news_contexts:
            for ta, ctx in news_contexts.items():
                if not isinstance(ctx, dict):
                    continue
                evidence = ctx.get("evidence") or []
                if not evidence:
                    continue
                by_topic: Dict[str, list] = {}
                for item in evidence:
                    if not isinstance(item, dict):
                        continue
                    topic = item.get("topic") or "Прочее"
                    by_topic.setdefault(topic, []).append(item)

                safe_ta = "".join(c if c.isalnum() or c in " _-" else "_" for c in ta)[:80].strip()
                for topic, items in by_topic.items():
                    top_items = sorted(items, key=lambda x: x.get("rank", 999))[:10]
                    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:60].strip()
                    lines = [f"ЦА: {ta}", f"Тема: {topic}", "=" * 60, ""]
                    for i, it in enumerate(top_items, 1):
                        date = (it.get("source_datetime") or "")[:10]
                        summary = (it.get("summary") or "").strip()
                        src = it.get("source_type") or ""
                        lines.append(f"#{i}  [{date}]  {src}")
                        lines.append(summary)
                        lines.append("")
                        lines.append("-" * 60)
                        lines.append("")
                    content = "\n".join(lines)
                    zf.writestr(f"news/{safe_ta}/{safe_topic}.txt", content)
    buf.seek(0)
    return buf.read()
