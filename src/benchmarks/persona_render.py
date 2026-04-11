from __future__ import annotations

from typing import Any, Dict

from src.benchmarks.common import translate_persona_to_en


def _clean_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    # Keep stable set for prompt readability.
    keys = [
        "target_audience_name",
        "age_group",
        "gender",
        "marital_status",
        "children_group",
        "education",
        "occupation",
        "income_level",
        "region",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    return {k: profile.get(k) for k in keys if k in profile and profile.get(k) is not None}


def render_persona_context(profile: Dict[str, Any], locale: str) -> str:
    locale_norm = (locale or "en").lower()
    if locale_norm not in {"en", "ru"}:
        locale_norm = "en"

    if locale_norm == "en":
        p = _clean_profile(translate_persona_to_en(profile))
        return (
            "You are role-playing a synthetic US banking customer.\n"
            "Stay consistent with this persona profile:\n"
            f"{p}\n"
            "Respond as the customer, not as an assistant."
        )

    p = _clean_profile(profile)
    return (
        "Вы играете роль синтетического банковского клиента в российском контексте.\n"
        "Сохраняйте поведение и стиль, соответствующие профилю:\n"
        f"{p}\n"
        "Отвечайте как клиент, а не как ассистент."
    )


def render_answer_style_hint(locale: str) -> str:
    if (locale or "").lower() == "ru":
        return "Дайте короткий, но содержательный ответ (1-3 предложения)."
    return "Give a concise but meaningful answer (1-3 sentences)."

