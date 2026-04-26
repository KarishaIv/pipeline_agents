from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import LLM_MODEL
from src.agents.survey_news_adapter import SurveyNewsAdapter
from src.agents.survey_persona_adapter import SurveyPersonaAdapter
from src.schemas.survey_deliberation_schema import (
    SurveyConflictResolution,
    SurveyDecisionOutput,
    SurveyPersonaModel,
    SurveyVoiceOpinion,
)
from src.utils import robust_llm_call


logger = logging.getLogger(__name__)


VOICE_ORDER = ["emotional", "rational", "social", "ideological"]
VOICE_WEIGHTS = {
    "emotional": 0.24,
    "rational": 0.30,
    "social": 0.22,
    "ideological": 0.24,
}
TOPIC_VOICE_WEIGHTS = {
    "financial_risk": {"emotional": 0.22, "rational": 0.37, "social": 0.15, "ideological": 0.26},
    "financial_self_view": {"emotional": 0.25, "rational": 0.31, "social": 0.21, "ideological": 0.23},
    "trust_media": {"emotional": 0.17, "rational": 0.33, "social": 0.22, "ideological": 0.28},
    "advertising_attitudes": {"emotional": 0.23, "rational": 0.29, "social": 0.20, "ideological": 0.28},
    "values_norms": {"emotional": 0.18, "rational": 0.16, "social": 0.27, "ideological": 0.39},
    "openness_social": {"emotional": 0.20, "rational": 0.16, "social": 0.28, "ideological": 0.36},
    "general": VOICE_WEIGHTS,
}


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coerce_text_list(value: Any, fallback: str) -> List[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        items = []
    else:
        items = [str(value).strip()]
    items = items[:3]
    return items or [fallback]


def _compact_persona(persona_context: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "age_group",
        "gender",
        "education",
        "income_level",
        "marital_status",
        "region",
        "children_group",
        "occupation",
        "target_audience_name",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "openness_readable",
        "conscientiousness_readable",
        "extraversion_readable",
        "agreeableness_readable",
        "neuroticism_readable",
    ]
    return {key: persona_context.get(key) for key in keys if persona_context.get(key) is not None}


def _extract_statement(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    match = re.search(r"[«\"']([^\"'»]+)[»\"']\s*$", text)
    if match:
        return match.group(1).strip()
    if ":" in text:
        return text.split(":")[-1].strip()
    return text


def _infer_question_topic(statement: str) -> str:
    lowered = statement.lower()
    if any(token in lowered for token in ["акци", "ценны", "инвест", "облигац", "слишком рискован", "рискованн", "риск для меня"]):
        return "financial_risk"
    if any(token in lowered for token in ["финансово обеспеч", "из-за денег", "денег", "зарплат", "заработ", "материальн", "долг", "кредит", "финанс"]):
        return "financial_self_view"
    if any(token in lowered for token in ["реклама", "бренд", "маркетинг"]):
        return "advertising_attitudes"
    if any(token in lowered for token in ["доверя", "газет", "новост", "сми", "медиа"]):
        return "trust_media"
    if any(token in lowered for token in ["женщин", "дом", "морал", "правильн", "ценност"]):
        return "values_norms"
    if any(token in lowered for token in ["люди", "культур", "обыча"]):
        return "openness_social"
    return "general"


def _serialize_block(title: str, payload: Any) -> str:
    return f"{title}:\n{payload}"


def _voice_label(name: str) -> str:
    return {
        "emotional": "эмоциональный голос",
        "rational": "рациональный голос",
        "social": "социальный голос",
        "ideological": "ценностный голос",
    }.get(name, name)


def _format_name_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} и {items[1]}"
    return ", ".join(items[:-1]) + f" и {items[-1]}"


def _describe_signal(value: float, low: str, mid: str, high: str) -> str:
    if value >= 0.66:
        return high
    if value <= 0.34:
        return low
    return mid


def _persona_anchor_text(persona: Dict[str, Any], persona_model: SurveyPersonaModel, summary: List[str]) -> str:
    traits: List[str] = []
    occupation = str(persona.get("occupation") or "").strip()
    income = str(persona.get("income_level") or "").strip()
    children = str(persona.get("children_group") or "").strip()
    region = str(persona.get("region") or "").strip()

    if occupation:
        traits.append(f"профессия — {occupation}")
    if income:
        traits.append(f"доход — {income}")
    if children and children not in {"0", "0.0"}:
        traits.append(f"дети — {children}")
    if region:
        traits.append(f"регион — {region}")

    profile_line = "; ".join(traits[:4]) if traits else "социально-демографический профиль без ярко выделенного фактора"
    summary_line = summary[0] if summary else "Профиль выглядит умеренным и зависит от темы вопроса."
    pressure_line = _describe_signal(
        persona_model.economic_pressure,
        low="финансовое давление выражено слабо",
        mid="финансовая устойчивость неоднозначна",
        high="финансовое давление заметно влияет на поведение",
    )
    return f"{profile_line}; {summary_line} Также {pressure_line}."


def _topic_reasoning_frame(topic: str, statement: str, persona_model: SurveyPersonaModel) -> str:
    if topic == "financial_risk":
        caution_text = _describe_signal(
            persona_model.financial_caution,
            low="склонность к финансовому риску у персоны не выглядит высокой",
            mid="отношение к финансовому риску у персоны смешанное",
            high="персона заметно осторожна в вопросах денег и риска",
        )
        return (
            f"В утверждении «{statement}» ключевой вопрос — воспринимает ли персона акции и ценные бумаги как риск для себя; "
            f"здесь особенно важны финансовая осторожность, экономическое давление и ощущение личной безопасности. {caution_text}"
        )
    if topic == "financial_self_view":
        pressure_text = _describe_signal(
            persona_model.economic_pressure,
            low="субъективное чувство обеспеченности не обязано быть низким",
            mid="объективные ресурсы и внутреннее чувство устойчивости расходятся неявно",
            high="субъективное ощущение финансовой устойчивости легко подтачивается расходами и обязательствами",
        )
        return (
            f"В утверждении «{statement}» система различает объективные деньги и субъективное чувство финансовой обеспеченности или мотивации работать ради денег. "
            f"{pressure_text}"
        )
    if topic == "trust_media":
        trust_text = _describe_signal(
            persona_model.media_skepticism,
            low="персона не выглядит резко недоверчивой к привычным источникам",
            mid="доверие к газетам и медиа у персоны выборочное",
            high="персона склонна перепроверять медиа-источники и не доверять им автоматически",
        )
        return (
            f"В утверждении «{statement}» вопрос не только о доверии вообще, но именно о доверии к газетам и медиа как источнику информации. "
            f"{trust_text}"
        )
    if topic == "advertising_attitudes":
        prof_text = _describe_signal(
            persona_model.professional_identity_salience,
            low="профессиональная идентичность здесь влияет слабо",
            mid="профессиональная роль может частично менять взгляд на рекламу",
            high="профессиональная роль сильно влияет на отношение к рекламе и ее полезности",
        )
        return (
            f"В утверждении «{statement}» важно различать личное раздражение от рекламы и понимание того, зачем реклама нужна. "
            f"{prof_text}"
        )
    if topic == "openness_social":
        openness_text = _describe_signal(
            persona_model.openness_to_change,
            low="персона тяготеет к знакомому и осторожна к новому",
            mid="интерес к новому у персоны умеренный",
            high="персона открыта к новому опыту и легче принимает другие культуры",
        )
        return f"В утверждении «{statement}» ключевую роль играет openness_to_change; {openness_text}"
    if topic == "values_norms":
        traditional_text = _describe_signal(
            persona_model.traditionalism,
            low="жесткая привязка к традиционным ролям выражена слабо",
            mid="ценности и социальные нормы у персоны смешанные",
            high="персона заметно тяготеет к традиционным нормам и устойчивым ролям",
        )
        return f"В утверждении «{statement}» решение связано с ценностями и нормами; {traditional_text}"
    return f"В утверждении «{statement}» итог зависит от бытовой практичности, привычек и общего самоощущения персоны."


def _voice_factor_summary(
    voices: Dict[str, SurveyVoiceOpinion],
    topic_weights: Dict[str, float],
) -> Tuple[str, str]:
    ranking = sorted(
        voices.items(),
        key=lambda item: topic_weights.get(item[0], 0.0) * (0.55 + 0.45 * item[1].salience),
        reverse=True,
    )
    dominant_names = [_voice_label(name) for name, _ in ranking[:2]]

    support_factors: List[str] = []
    reservations: List[str] = []
    for _, voice in ranking:
        for factor in voice.key_factors:
            if factor and factor not in support_factors:
                support_factors.append(factor)
        if voice.blocking_factor and voice.blocking_factor not in reservations:
            reservations.append(voice.blocking_factor)
    support_text = ", ".join(support_factors[:3]) if support_factors else "обобщенные внутренние соображения"
    reservation_text = ", ".join(reservations[:3]) if reservations else ""
    return _format_name_list(dominant_names), support_text + ("|RES|" + reservation_text if reservation_text else "")


def _derive_stance(support_score: float, conflict_pressure: float) -> str:
    if support_score >= 0.66 and conflict_pressure <= 0.80:
        return "agree"
    if support_score <= 0.34 and conflict_pressure <= 0.80:
        return "disagree"
    if support_score >= 0.58 and conflict_pressure <= 0.68:
        return "agree"
    if support_score <= 0.42 and conflict_pressure <= 0.72:
        return "disagree"
    return "uncertain"


def _topic_neutral_band(topic: str) -> Tuple[float, float]:
    if topic == "financial_risk":
        return 0.47, 0.53
    if topic == "trust_media":
        return 0.46, 0.54
    if topic == "financial_self_view":
        return 0.45, 0.55
    if topic == "advertising_attitudes":
        return 0.46, 0.54
    return 0.46, 0.54


def _topic_directional_lean(topic: str, persona_model: SurveyPersonaModel, voice_name: str) -> float:
    if topic == "financial_risk":
        lean = 0.10 * (persona_model.financial_caution - 0.5) + 0.05 * (persona_model.economic_pressure - 0.5)
        if voice_name == "rational":
            lean += 0.03 * (persona_model.consumer_pragmatism - 0.5)
        return lean
    if topic == "financial_self_view":
        lean = 0.12 * (persona_model.economic_pressure - 0.5) - 0.04 * (persona_model.professional_identity_salience - 0.5)
        if voice_name == "social":
            lean += 0.03 * (persona_model.social_conformity - 0.5)
        return lean
    if topic == "trust_media":
        return (
            0.07 * (persona_model.institutional_trust - 0.5)
            - 0.13 * (persona_model.media_skepticism - 0.5)
        )
    if topic == "advertising_attitudes":
        lean = 0.08 * (persona_model.media_skepticism - 0.5) - 0.09 * (persona_model.professional_identity_salience - 0.5)
        if voice_name == "rational":
            lean -= 0.03 * (persona_model.professional_identity_salience - 0.5)
        return lean
    if topic == "values_norms":
        return 0.10 * (persona_model.traditionalism - 0.5)
    if topic == "openness_social":
        return 0.10 * (persona_model.openness_to_change - 0.5)
    return 0.05 * (persona_model.consumer_pragmatism - 0.5)


def _normalize_voice(
    voice_name: str,
    voice: SurveyVoiceOpinion,
    topic: str,
    persona_model: SurveyPersonaModel,
) -> SurveyVoiceOpinion:
    payload = _dump_model(voice)
    support_score = _clamp(payload.get("support_score"), default=0.5)
    confidence = _clamp(payload.get("confidence"), default=0.55)
    conflict_pressure = _clamp(payload.get("conflict_pressure"), default=0.40)
    salience = _clamp(payload.get("salience"), default=0.55)
    stance = str(payload.get("stance") or "").strip()
    if stance not in {"agree", "disagree", "uncertain"}:
        stance = _derive_stance(support_score, conflict_pressure)

    lo, hi = _topic_neutral_band(topic)
    lean = _topic_directional_lean(topic, persona_model, voice_name)

    # Avoid dead-center 0.5 plateaus when persona signals already imply a weak direction.
    if lo <= support_score <= hi and abs(lean) >= 0.015:
        support_score = _clamp(support_score + lean, default=support_score)

    suggested_stance = _derive_stance(support_score, conflict_pressure)
    if stance == "uncertain":
        if suggested_stance != "uncertain":
            stance = suggested_stance
        elif abs(lean) >= 0.035 and conflict_pressure <= 0.62:
            stance = "agree" if lean > 0 else "disagree"
            support_score = _clamp(support_score + 0.03 * (1 if lean > 0 else -1), default=support_score)
        elif topic == "financial_risk" and support_score >= 0.56 and conflict_pressure <= 0.62:
            stance = "agree"
        elif topic == "trust_media" and support_score <= 0.48 and conflict_pressure <= 0.68:
            stance = "disagree"
    else:
        # If the explicit stance from the model contradicts the numeric score strongly, trust the score.
        if stance == "agree" and support_score < 0.50:
            stance = suggested_stance
        elif stance == "disagree" and support_score > 0.50:
            stance = suggested_stance

    payload.update(
        {
            "stance": stance,
            "support_score": support_score,
            "confidence": confidence,
            "conflict_pressure": conflict_pressure,
            "salience": salience,
        }
    )
    if hasattr(SurveyVoiceOpinion, "model_validate"):
        return SurveyVoiceOpinion.model_validate(payload)
    return SurveyVoiceOpinion(**payload)


def _parse_voice(raw: Any) -> SurveyVoiceOpinion:
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif hasattr(raw, "dict"):
        data = raw.dict()
    else:
        data = dict(raw)

    support_score = _clamp(data.get("support_score"), default=0.5)
    confidence = _clamp(data.get("confidence"), default=0.55)
    salience = _clamp(data.get("salience"), default=0.55)
    conflict_pressure = _clamp(data.get("conflict_pressure"), default=0.40)
    stance = str(data.get("stance") or "").strip()
    if stance not in {"agree", "disagree", "uncertain"}:
        stance = _derive_stance(support_score, conflict_pressure)

    payload = {
        "stance": stance,
        "support_score": support_score,
        "confidence": confidence,
        "salience": salience,
        "conflict_pressure": conflict_pressure,
        "reasoning": str(data.get("reasoning") or "Позиция сформулирована слишком общо.").strip(),
        "reaction": str(data.get("reaction") or "Позиция выражена неявно.").strip(),
        "key_factors": _coerce_text_list(data.get("key_factors"), fallback="ключевые факторы обозначены слишком общо"),
        "blocking_factor": str(data.get("blocking_factor") or "").strip() or None,
    }
    if hasattr(SurveyVoiceOpinion, "model_validate"):
        return SurveyVoiceOpinion.model_validate(payload)
    return SurveyVoiceOpinion(**payload)


def _voice_snapshot(voice: SurveyVoiceOpinion) -> Dict[str, Any]:
    return {
        "stance": voice.stance,
        "support_score": round(voice.support_score, 3),
        "confidence": round(voice.confidence, 3),
        "salience": round(voice.salience, 3),
        "conflict_pressure": round(voice.conflict_pressure, 3),
        "key_factors": voice.key_factors,
        "blocking_factor": voice.blocking_factor,
    }


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _topic_weights(topic: str, persona_model: SurveyPersonaModel) -> Dict[str, float]:
    base = dict(TOPIC_VOICE_WEIGHTS.get(topic, VOICE_WEIGHTS))
    if topic == "financial_risk":
        base["rational"] += 0.05 * persona_model.financial_caution
        base["emotional"] += 0.03 * persona_model.financial_caution
        base["social"] -= 0.03 * persona_model.financial_caution
        base["ideological"] -= 0.02 * persona_model.financial_caution
    elif topic == "financial_self_view":
        base["rational"] += 0.05 * persona_model.economic_pressure
        base["emotional"] += 0.03 * persona_model.economic_pressure
        base["social"] += 0.02 * persona_model.social_conformity
        base["ideological"] -= 0.02 * persona_model.economic_pressure
    elif topic == "trust_media":
        skepticism = persona_model.media_skepticism
        base["rational"] += 0.05 * skepticism
        base["emotional"] += 0.02 * skepticism
        base["social"] -= 0.03 * skepticism
        base["ideological"] -= 0.02 * skepticism
    elif topic == "advertising_attitudes":
        base["rational"] += 0.04 * persona_model.professional_identity_salience
        base["ideological"] += 0.03 * persona_model.professional_identity_salience
        base["emotional"] += 0.04 * persona_model.media_skepticism
        base["social"] -= 0.02 * persona_model.professional_identity_salience
    elif topic == "values_norms":
        base["ideological"] += 0.06 * persona_model.traditionalism
        base["social"] += 0.04 * persona_model.social_conformity
        base["rational"] -= 0.03 * persona_model.traditionalism
    elif topic == "openness_social":
        base["ideological"] += 0.05 * persona_model.openness_to_change
        base["social"] += 0.03 * persona_model.social_conformity
        base["rational"] -= 0.02
    else:
        base["rational"] += 0.03 * persona_model.consumer_pragmatism
        base["emotional"] += 0.02 * (1.0 - persona_model.consumer_pragmatism)

    total = sum(max(0.01, value) for value in base.values())
    return {key: round(max(0.01, value) / total, 3) for key, value in base.items()}


class StructuredSurveyVoice:
    def __init__(self, name: str, role_prompt: str, model: str = LLM_MODEL, temperature: float = 0.2):
        self.name = name
        self.role_prompt = role_prompt
        self.model = model
        self.temperature = temperature

    def _focus_hint(self, persona_model: SurveyPersonaModel) -> str:
        if self.name == "emotional":
            return (
                "Особенно учитывай внутреннее чувство безопасности, тревогу и эмоциональный комфорт. "
                f"Ключевые persona-сигналы: financial_caution={persona_model.financial_caution}, "
                f"economic_pressure={persona_model.economic_pressure}, "
                f"openness_to_change={persona_model.openness_to_change}."
            )
        if self.name == "rational":
            return (
                "Особенно учитывай практичность, внутреннюю непротиворечивость и реалистичность позиции. "
                f"Ключевые persona-сигналы: consumer_pragmatism={persona_model.consumer_pragmatism}, "
                f"financial_caution={persona_model.financial_caution}, "
                f"professional_identity_salience={persona_model.professional_identity_salience}."
            )
        if self.name == "social":
            return (
                "Особенно учитывай семейные ожидания, нормы окружения и репутацию. "
                f"Ключевые persona-сигналы: social_conformity={persona_model.social_conformity}, "
                f"traditionalism={persona_model.traditionalism}."
            )
        return (
            "Особенно учитывай ценности, идентичность и допустимость позиции для самой персоны. "
            f"Ключевые persona-сигналы: traditionalism={persona_model.traditionalism}, "
            f"openness_to_change={persona_model.openness_to_change}."
        )

    def build_prompt(self, context: Dict[str, Any]) -> str:
        lines = [
            self.role_prompt,
            "Верни JSON со следующими полями: stance, support_score, confidence, salience, conflict_pressure, reasoning, reaction, key_factors, blocking_factor.",
            "support_score означает склонность согласиться с утверждением: 0 = скорее не согласен, 1 = скорее согласен.",
            "Если позиция неоднозначна, используй stance=uncertain и повышай conflict_pressure.",
            "Не используй support_score=0.5 по умолчанию. Если есть хотя бы слабый перевес в любую сторону, отрази его числом.",
            _serialize_block("Утверждение", context["statement"]),
            _serialize_block("Исходный вопрос", context["scenario"]),
            _serialize_block("Профиль персоны", context["persona"]),
            _serialize_block("Survey persona model", context["persona_model"]),
            _serialize_block("Краткое summary персоны для survey", context["persona_summary"]),
            _serialize_block("Тема вопроса", context["world_signals"]["question_topic"]),
            self._focus_hint(context["persona_model_obj"]),
        ]
        if context["world_signals"]["question_topic"] == "financial_risk":
            lines.append(
                "Для вопросов о риске инвестиций не путай нехватку опыта с несогласием. "
                "Если персона не разбирается в инструменте и поэтому настораживается, это чаще сдвигает к agree или uncertain-with-positive-lean, а не к disagree."
            )
        if context["world_signals"]["question_topic"] == "financial_self_view":
            lines.append(
                "Для вопросов о личной финансовой обеспеченности и мотивации работать из-за денег различай объективный доход и субъективное чувство устойчивости. "
                "Наличие дохода выше среднего не всегда означает ощущение полной обеспеченности."
            )
        if context["world_signals"]["question_topic"] == "trust_media":
            lines.append(
                "Для вопросов о газетах и медиа не приравнивай автоматически доверие к институтам к доверию к конкретному медиаканалу. "
                "Учитывай привычку к критической проверке источников, устаревание канала и возможную выборочную доверчивость."
            )
        if context["world_signals"]["question_topic"] == "advertising_attitudes":
            lines.append(
                "Для вопросов о рекламе различай профессиональное понимание пользы рекламы и личное раздражение от навязчивых форматов. "
                "Для специалиста по маркетингу или смежной роли это часто реальный внутренний конфликт, а не простое agree/disagree."
            )
        if context["world_context"]:
            lines.extend(
                [
                    _serialize_block("Внешний контекст среды", context["world_context"]),
                    _serialize_block("Сигналы внешней среды", context["world_signals"]),
                    "Внешний контекст — это окружение персоны, а не часть ее профиля. Учитывай его только если он действительно релевантен этому утверждению.",
                ]
            )
        return "\n\n".join(lines)

    async def run(self, context: Dict[str, Any]) -> Tuple[SurveyVoiceOpinion, int]:
        prompt = self.build_prompt(context)
        response = await robust_llm_call(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            structured_output=SurveyVoiceOpinion,
        )
        parsed = _parse_voice(response)
        parsed = _normalize_voice(
            voice_name=self.name,
            voice=parsed,
            topic=context["world_signals"]["question_topic"],
            persona_model=context["persona_model_obj"],
        )
        return parsed, len(prompt)


class StructuredSurveyConflictResolver:
    def __init__(self, model: str = LLM_MODEL, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def build_prompt(
        self,
        scenario: str,
        statement: str,
        persona: Dict[str, Any],
        persona_model: SurveyPersonaModel,
        world_signals: Dict[str, Any],
        voices: Dict[str, SurveyVoiceOpinion],
    ) -> str:
        compact_voices = {
            name: {
                "stance": voice.stance,
                "support_score": round(voice.support_score, 3),
                "confidence": round(voice.confidence, 3),
                "salience": round(voice.salience, 3),
                "conflict_pressure": round(voice.conflict_pressure, 3),
                "key_factors": voice.key_factors,
                "blocking_factor": voice.blocking_factor,
                "reaction": voice.reaction,
            }
            for name, voice in voices.items()
        }
        return "\n\n".join(
            [
                "Ты conflict-aware synthesis layer для survey reasoning.",
                "Твоя задача — не переписать ответ с нуля, а дать небольшую корректировку к уже посчитанному решению.",
                "Верни JSON со следующими полями: support_adjustment, confidence_adjustment, conflict_summary, dominant_tension.",
                "support_adjustment должен быть в диапазоне [-0.15, 0.15].",
                "confidence_adjustment должен быть в диапазоне [-0.12, 0.12].",
                _serialize_block("Исходный вопрос", scenario),
                _serialize_block("Утверждение", statement),
                _serialize_block("Профиль персоны", persona),
                _serialize_block("Survey persona model", _dump_model(persona_model)),
                _serialize_block("Сигналы среды", world_signals),
                _serialize_block("Голоса", json.dumps(compact_voices, ensure_ascii=False, indent=2)),
                "Если голоса в основном согласны, adjustment должен быть очень маленьким. Если есть реальная коллизия между ценностями, логикой и эмоцией, можно слегка сдвинуть итог.",
            ]
        )

    async def run(
        self,
        scenario: str,
        statement: str,
        persona: Dict[str, Any],
        persona_model: SurveyPersonaModel,
        world_signals: Dict[str, Any],
        voices: Dict[str, SurveyVoiceOpinion],
    ) -> Tuple[SurveyConflictResolution, int]:
        prompt = self.build_prompt(
            scenario=scenario,
            statement=statement,
            persona=persona,
            persona_model=persona_model,
            world_signals=world_signals,
            voices=voices,
        )
        response = await robust_llm_call(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            structured_output=SurveyConflictResolution,
        )
        if hasattr(SurveyConflictResolution, "model_validate"):
            parsed = SurveyConflictResolution.model_validate(_dump_model(response))
        else:
            parsed = SurveyConflictResolution(**_dump_model(response))
        return parsed, len(prompt)


class StructuredSurveyReasoner:
    def __init__(
        self,
        persona_context: Dict[str, Any],
        world_context: Optional[Dict[str, Any]] = None,
        model: str = LLM_MODEL,
    ):
        self.persona_context = persona_context
        self.news_adapter = SurveyNewsAdapter()
        self.world_context = self.news_adapter.build(world_context)
        self.model = model
        self.persona_model = SurveyPersonaAdapter().build(persona_context)
        self.conflict_resolver = StructuredSurveyConflictResolver(model=model)
        self.voices = {
            "emotional": StructuredSurveyVoice(
                "emotional",
                "Ты эмоциональный голос персоны. Смотри на инстинктивный отклик, симпатию, тревогу, отторжение и внутренний комфорт от согласия с утверждением.",
                model=model,
                temperature=0.25,
            ),
            "rational": StructuredSurveyVoice(
                "rational",
                "Ты рациональный голос персоны. Смотри на логику, практичность, долгосрочные последствия и реалистичность согласия с утверждением.",
                model=model,
                temperature=0.15,
            ),
            "social": StructuredSurveyVoice(
                "social",
                "Ты социальный голос персоны. Смотри на давление окружения, нормы, статус, семью и репутацию.",
                model=model,
                temperature=0.2,
            ),
            "ideological": StructuredSurveyVoice(
                "ideological",
                "Ты ценностный голос персоны. Смотри на убеждения, идентичность, внутреннюю правильность и моральную совместимость с утверждением.",
                model=model,
                temperature=0.2,
            ),
        }

    def _build_context(self, scenario: str) -> Dict[str, Any]:
        statement = _extract_statement(scenario)
        question_topic = _infer_question_topic(statement)
        news_resolution = self.news_adapter.resolve(
            adapted_context=self.world_context,
            target_topic=question_topic,
            persona_context=self.persona_context,
        )
        world_signals = news_resolution["signals"]
        return {
            "scenario": scenario,
            "statement": statement,
            "persona": _compact_persona(self.persona_context),
            "persona_model": _dump_model(self.persona_model),
            "persona_model_obj": self.persona_model,
            "persona_summary": self.persona_model.summary,
            "world_context": news_resolution["selected_context"],
            "world_context_bundle": self.world_context,
            "world_signals": world_signals,
            "topic_weights": _topic_weights(world_signals["question_topic"], self.persona_model),
        }

    def _needs_conflict_resolution(self, voices: Dict[str, SurveyVoiceOpinion]) -> bool:
        agree_votes = sum(1 for voice in voices.values() if voice.stance == "agree")
        disagree_votes = sum(1 for voice in voices.values() if voice.stance == "disagree")
        uncertain_votes = sum(1 for voice in voices.values() if voice.stance == "uncertain")
        avg_conflict = _mean([voice.conflict_pressure for voice in voices.values()])
        avg_support = _mean([voice.support_score for voice in voices.values()])
        support_span = max(voice.support_score for voice in voices.values()) - min(voice.support_score for voice in voices.values())
        return (
            (agree_votes and disagree_votes)
            or (uncertain_votes >= 4 and (0.45 <= avg_support <= 0.55 or avg_conflict >= 0.52))
            or (uncertain_votes >= 3 and support_span <= 0.18)
            or avg_conflict >= 0.55
        )

    def _compute_decision(
        self,
        voices: Dict[str, SurveyVoiceOpinion],
        context: Dict[str, Any],
        resolution: Optional[SurveyConflictResolution] = None,
    ) -> SurveyDecisionOutput:
        weighted_support = 0.0
        weight_total = 0.0
        confidence_values = []
        conflict_values = []
        salience_values = []
        agree_votes = 0
        disagree_votes = 0
        uncertain_votes = 0
        soft_agree_votes = 0
        soft_disagree_votes = 0

        topic_weights = context["topic_weights"]
        for voice_name, voice in voices.items():
            influence = topic_weights[voice_name] * (0.55 + 0.45 * voice.salience)
            weighted_support += influence * voice.support_score
            weight_total += influence
            confidence_values.append(voice.confidence)
            conflict_values.append(voice.conflict_pressure)
            salience_values.append(voice.salience)
            if voice.stance == "agree":
                agree_votes += 1
            elif voice.stance == "disagree":
                disagree_votes += 1
            else:
                uncertain_votes += 1
                if voice.support_score >= 0.56:
                    soft_agree_votes += 1
                elif voice.support_score <= 0.44:
                    soft_disagree_votes += 1

        avg_support = weighted_support / weight_total if weight_total else 0.5
        avg_confidence = _mean(confidence_values)
        avg_conflict = _mean(conflict_values)
        avg_salience = _mean(salience_values)
        support_span = max(voice.support_score for voice in voices.values()) - min(voice.support_score for voice in voices.values())
        contradiction_penalty = 0.06 if agree_votes and disagree_votes else 0.0
        uncertainty_penalty = 0.045 * avg_conflict if uncertain_votes else 0.0
        if uncertain_votes == len(voices):
            uncertainty_penalty *= 0.55

        world_signals = context["world_signals"]
        headwind = world_signals["headwind"]
        opportunity_support = world_signals["opportunity_support"]
        relevance = world_signals["question_relevance"]
        persona_model = context["persona_model_obj"]
        topic = world_signals["question_topic"]

        persona_bias = 0.0
        if topic == "financial_risk":
            persona_bias += 0.06 * (0.5 - persona_model.financial_caution)
            persona_bias += 0.04 * (persona_model.consumer_pragmatism - 0.5)
            if uncertain_votes >= 3 and avg_support >= 0.52:
                persona_bias += 0.03 + 0.04 * persona_model.financial_caution
        elif topic == "financial_self_view":
            persona_bias += 0.09 * (persona_model.economic_pressure - 0.5)
            persona_bias += 0.03 * (0.5 - persona_model.financial_caution)
        elif topic == "trust_media":
            persona_bias += 0.04 * (persona_model.institutional_trust - 0.5)
            persona_bias -= 0.10 * (persona_model.media_skepticism - 0.5)
        elif topic == "advertising_attitudes":
            persona_bias += 0.07 * (persona_model.media_skepticism - 0.5)
            persona_bias -= 0.08 * (persona_model.professional_identity_salience - 0.5)
        elif topic == "values_norms":
            persona_bias += 0.10 * (persona_model.traditionalism - 0.5)
        elif topic == "openness_social":
            persona_bias += 0.10 * (persona_model.openness_to_change - 0.5)
        else:
            persona_bias += 0.05 * (persona_model.consumer_pragmatism - 0.5)

        uncertainty_lean = 0.0
        if uncertain_votes:
            uncertainty_lean = 0.12 * (avg_support - 0.5)
            if uncertain_votes == len(voices):
                uncertainty_lean += 0.06 * (avg_support - 0.5)

        final_support = _clamp(
            avg_support
            + 0.08 * opportunity_support
            - 0.10 * headwind
            - contradiction_penalty
            - uncertainty_penalty
            + 0.03 * (avg_salience - 0.5),
            default=0.5,
        )
        final_support = _clamp(final_support + persona_bias + uncertainty_lean, default=0.5)
        if resolution is not None:
            final_support = _clamp(final_support + resolution.support_adjustment, default=0.5)

        if agree_votes >= 3 and final_support >= 0.48:
            decision = True
        elif disagree_votes >= 3 and final_support <= 0.52:
            decision = False
        elif agree_votes + soft_agree_votes >= 3 and final_support >= 0.52:
            decision = True
        elif disagree_votes + soft_disagree_votes >= 3 and final_support <= 0.48:
            decision = False
        else:
            decision = final_support >= 0.5

        decision_confidence = _clamp(
            0.44
            + 0.32 * abs(final_support - 0.5) * 2
            + 0.22 * avg_confidence
            - 0.18 * avg_conflict
            - 0.10 * support_span,
            default=0.5,
        )
        if resolution is not None:
            decision_confidence = _clamp(decision_confidence + resolution.confidence_adjustment, default=0.5)

        conflict_summary = None
        if agree_votes and disagree_votes:
            conflict_summary = (
                f"Есть реальный конфликт между голосами: agree={agree_votes}, disagree={disagree_votes}. "
                f"Самые сильные расхождения возникли по теме '{context['world_signals']['question_topic']}'."
            )
        elif any(voice.stance == "uncertain" for voice in voices.values()):
            conflict_summary = "Часть голосов не пришла к однозначной позиции, поэтому решение формировалось при наличии внутренней неоднозначности."
        if resolution is not None:
            conflict_summary = resolution.conflict_summary

        dominant_supporters = [name for name, voice in voices.items() if voice.stance == "agree"]
        dominant_blockers = [name for name, voice in voices.items() if voice.stance == "disagree"]
        supporter_text = (
            _format_name_list([_voice_label(name) for name in dominant_supporters])
            if dominant_supporters
            else "ни одна перспектива не дала сильной поддержки"
        )
        blocker_text = (
            _format_name_list([_voice_label(name) for name in dominant_blockers])
            if dominant_blockers
            else "ни одна перспектива не дала сильного возражения"
        )
        dominant_voice_text, factor_blob = _voice_factor_summary(voices, context["topic_weights"])
        if "|RES|" in factor_blob:
            support_factor_text, reservation_text = factor_blob.split("|RES|", 1)
        else:
            support_factor_text, reservation_text = factor_blob, ""
        persona_anchor = _persona_anchor_text(context["persona"], persona_model, context["persona_summary"])
        topic_frame = _topic_reasoning_frame(topic, context["statement"], persona_model)
        world_phrase = ""
        if context["world_context"] and relevance > 0.15:
            world_phrase = (
                f" Внешний контекст среды дал поправку через question_relevance={round(relevance, 2)}, "
                f"headwind={round(headwind, 2)} и opportunity_support={round(opportunity_support, 2)}."
            )
        decision_phrase = "Итог ближе к согласию с утверждением." if decision else "Итог ближе к несогласию с утверждением."
        reservation_phrase = (
            f" Главные оговорки и сдерживающие факторы: {reservation_text}."
            if reservation_text
            else ""
        )
        conflict_phrase = (
            f" {conflict_summary}"
            if conflict_summary
            else ""
        )
        reasoning = (
            f"{decision_phrase} {topic_frame} "
            f"Для этой персоны важен такой контекст: {persona_anchor} "
            f"Сильнее всего на итог повлияли {dominant_voice_text}; их основные аргументы: {support_factor_text}. "
            f"Поддерживающие голоса: {supporter_text}; сдерживающие голоса: {blocker_text}."
            f"{reservation_phrase}{conflict_phrase} "
            f"Итоговый support_score={round(final_support, 3)}, средняя уверенность={round(avg_confidence, 3)}, "
            f"средний внутренний конфликт={round(avg_conflict, 3)}, persona_bias={round(persona_bias, 3)}.{world_phrase}"
        ).strip()

        score_breakdown = {
            "avg_support": round(avg_support, 3),
            "avg_confidence": round(avg_confidence, 3),
            "avg_conflict": round(avg_conflict, 3),
            "avg_salience": round(avg_salience, 3),
            "support_span": round(support_span, 3),
            "agree_votes": agree_votes,
            "disagree_votes": disagree_votes,
            "uncertain_votes": uncertain_votes,
            "soft_agree_votes": soft_agree_votes,
            "soft_disagree_votes": soft_disagree_votes,
            "question_topic": context["world_signals"]["question_topic"],
            "question_relevance": world_signals["question_relevance"],
            "news_headwind": round(headwind, 3),
            "news_opportunity_support": round(opportunity_support, 3),
            "persona_bias": round(persona_bias, 3),
            "uncertainty_lean": round(uncertainty_lean, 3),
            "topic_weights": context["topic_weights"],
            "final_support": round(final_support, 3),
            "resolver_used": bool(resolution),
            "resolver_tension": getattr(resolution, "dominant_tension", None) if resolution else None,
        }

        payload = {
            "reasoning": reasoning,
            "decision": decision,
            "confidence": decision_confidence,
            "conflict_summary": conflict_summary,
            "voice_stances": {name: voice.stance for name, voice in voices.items()},
            "score_breakdown": score_breakdown,
            "survey_mode": "structured",
            "news_context_used": bool(context["world_context"]),
        }
        if hasattr(SurveyDecisionOutput, "model_validate"):
            return SurveyDecisionOutput.model_validate(payload)
        return SurveyDecisionOutput(**payload)

    async def run(
        self,
        scenario: str,
        max_generations: Optional[int] = None,
        persona_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self._build_context(scenario)
        logger.info(f"🚀 Starting structured survey reasoning for persona: {persona_id} on scenario: '{scenario}'")

        tasks = [self.voices[name].run(context) for name in VOICE_ORDER]
        results = await asyncio.gather(*tasks)
        voice_map = {name: voice for name, (voice, _) in zip(VOICE_ORDER, results)}
        prompt_lengths = {name: prompt_len for name, (_, prompt_len) in zip(VOICE_ORDER, results)}
        resolver_used = False
        resolver_prompt_len = 0
        resolution = None
        if self._needs_conflict_resolution(voice_map):
            resolution, resolver_prompt_len = await self.conflict_resolver.run(
                scenario=scenario,
                statement=context["statement"],
                persona=context["persona"],
                persona_model=self.persona_model,
                world_signals=context["world_signals"],
                voices=voice_map,
            )
            resolver_used = True
        final_decision = self._compute_decision(voice_map, context, resolution=resolution)

        history_payload = {
            name: [
                {
                    "reasoning": voice.reasoning,
                    "reaction": voice.reaction,
                    "stance": voice.stance,
                    "support_score": round(voice.support_score, 3),
                    "confidence": round(voice.confidence, 3),
                    "salience": round(voice.salience, 3),
                    "conflict_pressure": round(voice.conflict_pressure, 3),
                    "key_factors": voice.key_factors,
                    "blocking_factor": voice.blocking_factor,
                }
            ]
            for name, voice in voice_map.items()
        }

        return {
            "persona_id": persona_id,
            "scenario": scenario,
            "persona_context": self.persona_context,
            "world_context": context["world_context"],
            "emotional_history": history_payload["emotional"],
            "rational_history": history_payload["rational"],
            "social_history": history_payload["social"],
            "ideological_history": history_payload["ideological"],
            "generation_count": len(VOICE_ORDER),
            "max_generations": len(VOICE_ORDER),
            "final_decision": final_decision.model_dump() if hasattr(final_decision, "model_dump") else final_decision.dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "survey_mode": "structured",
            "trace": {
                "llm_calls": len(VOICE_ORDER) + (1 if resolver_used else 0),
                "prompt_char_counts": {
                    **prompt_lengths,
                    **({"conflict_resolver": resolver_prompt_len} if resolver_used else {}),
                },
                "prompt_chars": sum(prompt_lengths.values()) + resolver_prompt_len,
                "world_context_used": bool(context["world_context"]),
                "question_topic": context["world_signals"]["question_topic"],
                "resolver_used": resolver_used,
                "persona_model": _dump_model(self.persona_model),
                "topic_weights": context["topic_weights"],
                "world_signals": context["world_signals"],
            },
            "trace_voices": {name: _voice_snapshot(voice) for name, voice in voice_map.items()},
        }

    async def answer_survey_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        all_results = []
        for i, question in enumerate(questions):
            logger.info(f"[SURVEY][structured] Запуск вопроса {i + 1}/{len(questions)}: {question[:50]}...")
            question_result = await self.run(
                scenario=question,
                max_generations=len(VOICE_ORDER),
                persona_id=self.persona_context.get("name", f"persona_{i}"),
            )
            all_results.append(
                {
                    "question": question,
                    "question_index": i,
                    "scenario": question,
                    "full_state": question_result,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        return all_results
