from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.schemas.decision_schema import DecisionOutcome
from src.schemas.deliberation_schema import (
    SynthesizedCreditDecision,
    SynthesizedDecisionNarrative,
    VoiceOpinion,
)
from src.schemas.news_context_schema import NewsContextSnapshot, normalize_news_context_payload
from src.utils import robust_llm_call


def _as_clean_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if v is not None}


def _compact_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    ordered_keys = [
        "target_audience_name",
        "region",
        "age_group",
        "gender",
        "education",
        "income_level",
        "marital_status",
        "children_group",
        "occupation",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]
    compact = {}
    for key in ordered_keys:
        if key in profile and profile[key] is not None:
            compact[key] = profile[key]
    return compact or _as_clean_dict(profile)


def _compact_history(history: List[Dict[str, Any]], limit: int = 2) -> List[Dict[str, Any]]:
    if not history:
        return []
    compact_rows: List[Dict[str, Any]] = []
    for row in history[-limit:]:
        tool_result = row.get("tool_result", {}) if isinstance(row, dict) else {}
        compact_rows.append(
            {
                "step": row.get("step"),
                "action": row.get("action"),
                "tool_message": tool_result.get("message") if isinstance(tool_result, dict) else None,
                "status": tool_result.get("status") if isinstance(tool_result, dict) else None,
                "emotional_state": row.get("emotional_state"),
            }
        )
    return compact_rows


def _serialize_lines(title: str, payload: Any) -> str:
    if payload in (None, "", [], {}):
        return f"{title}: не указано"
    return f"{title}: {payload}"


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _shorten_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _keyword_score(values: List[str], keywords: List[str]) -> float:
    haystack = " ".join(str(value or "").lower() for value in values if str(value or "").strip())
    if not haystack:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return min(1.0, hits / 4.0)


def _normalize_news_context(raw: Optional[Dict[str, Any]]) -> Optional[NewsContextSnapshot]:
    if raw is None:
        return None
    if isinstance(raw, NewsContextSnapshot):
        return raw
    data = normalize_news_context_payload(raw)
    if hasattr(NewsContextSnapshot, "model_validate"):
        return NewsContextSnapshot.model_validate(data)
    return NewsContextSnapshot(**data)


def _compact_news_context(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normalized = _normalize_news_context(raw)
    if normalized is None:
        return None
    data = _dump_model(normalized)
    evidence_rows = sorted(
        data.get("evidence", []),
        key=lambda item: (
            item.get("rank") is None,
            item.get("rank") if item.get("rank") is not None else 999,
        ),
    )
    compact_evidence: List[Dict[str, Any]] = []
    for row in evidence_rows[:2]:
        compact_evidence.append(
            {
                "topic": row.get("topic"),
                "rank": row.get("rank"),
                "summary": _shorten_text(row.get("summary"), limit=120),
                "source_type": row.get("source_type"),
                "source_datetime": row.get("source_datetime"),
            }
        )

    return {
        "snapshot_id": data.get("snapshot_id"),
        "generated_at": data.get("generated_at"),
        "audience": data.get("audience"),
        "question": data.get("question"),
        "overall_reaction": data.get("overall_reaction"),
        "confidence": _clamp_score(data.get("confidence"), default=0.5),
        "impact_horizon": data.get("impact_horizon"),
        "summary_text": _shorten_text(data.get("summary_text"), limit=260),
        "factors": _coerce_text_list(data.get("factors"), fallback="внешний фон описан слишком общо"),
        "risks": _coerce_text_list(data.get("risks"), fallback="явные внешние риски не выделены"),
        "opportunities": _coerce_text_list(data.get("opportunities"), fallback="точки отклика не выделены"),
        "audience_effects": _coerce_text_list(
            data.get("audience_effects"),
            fallback="специфический эффект на аудиторию описан слабо",
        ),
        "evidence": compact_evidence,
        "fictional_warning": bool(data.get("fictional_warning", False)),
    }


def _derive_news_signals(compact_news: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not compact_news:
        return {}

    summary_text = str(compact_news.get("summary_text") or "")
    factors = [str(item) for item in compact_news.get("factors", [])]
    risks = [str(item) for item in compact_news.get("risks", [])]
    opportunities = [str(item) for item in compact_news.get("opportunities", [])]
    audience_effects = [str(item) for item in compact_news.get("audience_effects", [])]
    evidence_summaries = [str(item.get("summary") or "") for item in compact_news.get("evidence", [])]
    question = str(compact_news.get("question") or "")
    text_blocks = [summary_text, question, *factors, *risks, *opportunities, *audience_effects, *evidence_summaries]

    risk_keywords = [
        "ставк",
        "ужесточ",
        "осторож",
        "риск",
        "неопредел",
        "снижен",
        "отказ",
        "нагруз",
        "нестабиль",
        "охлажден",
    ]
    opportunity_keywords = [
        "господдерж",
        "льгот",
        "выгод",
        "поддержк",
        "программ",
        "рефинанс",
        "скидк",
    ]
    credit_keywords = [
        "кредит",
        "ипотек",
        "ставк",
        "долг",
        "займ",
        "потреб",
        "одобр",
        "заявк",
    ]
    uncertainty_keywords = ["неопредел", "нестабиль", "риск", "ужесточ", "осторож", "отказ"]

    confidence = _clamp_score(compact_news.get("confidence"), default=0.5)
    horizon_factor = {
        "short_term": 1.0,
        "medium_term": 0.75,
        "long_term": 0.55,
    }.get(str(compact_news.get("impact_horizon") or ""), 0.7)

    reaction = str(compact_news.get("overall_reaction") or "neutral")
    negative_bias = 1.0 if reaction == "negative" else 0.35 if reaction == "neutral" else 0.0
    positive_bias = 1.0 if reaction == "positive" else 0.35 if reaction == "neutral" else 0.0

    risk_density = min(
        1.0,
        0.18 * len(risks)
        + 0.10 * len(audience_effects)
        + 0.24 * _keyword_score(text_blocks, risk_keywords),
    )
    opportunity_density = min(
        1.0,
        0.20 * len(opportunities)
        + 0.22 * _keyword_score(text_blocks, opportunity_keywords),
    )
    uncertainty_density = _keyword_score(text_blocks, uncertainty_keywords)
    credit_relevance = _keyword_score(text_blocks, credit_keywords)
    if "кредит" in question.lower():
        credit_relevance = max(credit_relevance, 0.75)
    credit_relevance = max(0.25, credit_relevance)

    news_strength = confidence * horizon_factor
    credit_headwind = _clamp_score(
        (0.42 * negative_bias + 0.33 * risk_density + 0.25 * uncertainty_density)
        * news_strength
        * credit_relevance,
        default=0.0,
    )
    opportunity_support = _clamp_score(
        (0.45 * positive_bias + 0.35 * opportunity_density + 0.20 * (1.0 - min(1.0, risk_density)))
        * news_strength
        * max(0.35, credit_relevance),
        default=0.0,
    )
    stress_bias = _clamp_score(
        (0.45 * negative_bias + 0.35 * risk_density + 0.20 * uncertainty_density) * news_strength,
        default=0.0,
    )
    trust_drag = _clamp_score(
        (0.40 * negative_bias + 0.35 * uncertainty_density + 0.25 * risk_density) * news_strength,
        default=0.0,
    )

    return {
        "snapshot_id": compact_news.get("snapshot_id"),
        "overall_reaction": reaction,
        "confidence": round(confidence, 3),
        "impact_horizon": compact_news.get("impact_horizon"),
        "credit_relevance": round(credit_relevance, 3),
        "credit_headwind": round(credit_headwind, 3),
        "opportunity_support": round(opportunity_support, 3),
        "stress_bias": round(stress_bias, 3),
        "trust_drag": round(trust_drag, 3),
    }


def _decision_mode_value(decision_mode: str) -> str:
    if decision_mode not in {"direct", "compact_debate"}:
        raise ValueError(f"Unsupported decision mode: {decision_mode}")
    return decision_mode


def _decision_from_data(data: Dict[str, Any]) -> DecisionOutcome:
    if hasattr(DecisionOutcome, "model_validate"):
        return DecisionOutcome.model_validate(data)
    return DecisionOutcome(**data)


def _coerce_text_list(value: Any, fallback: str) -> List[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        items = []
    else:
        items = [str(value).strip()]
    items = items[:3]
    return items or [fallback]


def _clamp_score(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _readiness_level_from_score(score: float) -> str:
    if score < 0.34:
        return "browsing"
    if score < 0.67:
        return "considering"
    return "ready_now"


def _coerce_stance(value: Any) -> Optional[str]:
    stance = str(value or "").strip()
    if stance in {"take_credit", "reject_credit", "uncertain"}:
        return stance
    return None


def _derive_voice_stance(
    desire_for_credit: float,
    need_for_credit: float,
    readiness_now: float,
    risk_pressure: float,
    confidence: float,
) -> str:
    support = 0.34 * desire_for_credit + 0.38 * need_for_credit + 0.28 * readiness_now
    caution = 0.58 * risk_pressure + 0.12 * (1.0 - confidence)
    margin = support - caution
    if margin >= 0.12 and readiness_now >= 0.52 and need_for_credit >= 0.40:
        return "take_credit"
    if margin <= -0.05 or (need_for_credit < 0.30 and readiness_now < 0.42 and risk_pressure > 0.52):
        return "reject_credit"
    return "uncertain"


def _derive_emotional_stance(voice: VoiceOpinion) -> str:
    if (
        voice.desire_for_credit >= 0.68
        and voice.readiness_now >= 0.46
        and voice.risk_pressure <= 0.62
    ):
        return "take_credit"
    if (
        voice.desire_for_credit <= 0.32
        or (voice.readiness_now < 0.28 and voice.risk_pressure > 0.58)
    ):
        return "reject_credit"
    return "uncertain"


def _derive_rational_stance(voice: VoiceOpinion) -> str:
    if (
        voice.need_for_credit >= 0.52
        and voice.readiness_now >= 0.48
        and voice.risk_pressure <= 0.55
    ):
        return "take_credit"
    if (
        (voice.need_for_credit < 0.32 and voice.readiness_now < 0.40)
        or (voice.risk_pressure > 0.65 and voice.need_for_credit < 0.45)
    ):
        return "reject_credit"
    return "uncertain"


def _infer_goal_intent(goal_description: Any, history: List[Dict[str, Any]]) -> str:
    goal_text = str(goal_description or "").strip().lower()
    application_markers = [
        "оформ",
        "подать заяв",
        "взять кредит",
        "получить кредит",
        "оформить кредит",
        "подать заявку",
        "take credit",
        "apply",
    ]
    informational_markers = [
        "узнать",
        "ознаком",
        "изуч",
        "посмотр",
        "провер",
        "сравн",
        "услови",
        "програм",
        "лояльност",
        "информац",
        "доступн",
        "learn",
        "information",
    ]
    exploratory_markers = [
        "подобра",
        "выбра",
        "рассчит",
        "калькулятор",
        "вариант",
        "оцен",
        "explore",
        "compare",
    ]

    if any(marker in goal_text for marker in informational_markers):
        return "informational"
    if any(marker in goal_text for marker in exploratory_markers):
        return "exploratory"
    if any(marker in goal_text for marker in application_markers):
        return "application"

    history_chunks: List[str] = []
    for row in history or []:
        if not isinstance(row, dict):
            continue
        history_chunks.extend(
            [
                str(row.get("action") or ""),
                str(row.get("tool_message") or ""),
            ]
        )
    history_text = " ".join(history_chunks).lower()

    if any(marker in history_text for marker in application_markers):
        return "application"
    if any(marker in history_text for marker in informational_markers):
        return "informational"
    if any(marker in history_text for marker in exploratory_markers):
        return "exploratory"
    return "exploratory"


def _build_guardrail_note(guardrails: List[str], context: Dict[str, Any]) -> str:
    goal_intent = context.get("goal_intent")
    mapping = {
        "informational_goal_requires_need": (
            "цель клиента пока информационная, а рациональная необходимость кредита остается низкой"
        ),
        "rational_reject_blocks_positive_decision": (
            "рациональный голос видит блокирующие причины и не подтверждает оформление"
        ),
        "no_voice_ready_now": (
            "ни один голос не показывает достаточной готовности оформить кредит прямо сейчас"
        ),
        "interest_without_need": (
            "эмоциональный интерес к предложению не подкреплен реальной потребностью в кредите"
        ),
        "risk_pressure_too_high": (
            "уровень осторожности и риска слишком высок для положительного решения"
        ),
        "negative_news_headwind": (
            "внешний новостной фон для этой аудитории сейчас слишком негативен для немедленного положительного решения"
        ),
    }
    details = [mapping[item] for item in guardrails if item in mapping]
    if not details:
        details = ["сигналов за немедленное оформление кредита недостаточно"]
    prefix = "Итог был скорректирован guardrail-правилом: "
    suffix = f" (тип цели: {goal_intent})." if goal_intent else "."
    return prefix + "; ".join(details) + suffix


def _voice_snapshot(voice: VoiceOpinion) -> Dict[str, Any]:
    data = voice.model_dump() if hasattr(voice, "model_dump") else voice.dict()
    return {
        "stance": data.get("stance"),
        "readiness_level": data.get("readiness_level"),
        "desire_for_credit": round(float(data.get("desire_for_credit", 0.0)), 3),
        "need_for_credit": round(float(data.get("need_for_credit", 0.0)), 3),
        "readiness_now": round(float(data.get("readiness_now", 0.0)), 3),
        "risk_pressure": round(float(data.get("risk_pressure", 0.0)), 3),
        "confidence": round(float(data.get("confidence", 0.0)), 3),
        "theses": data.get("theses"),
        "key_factors": data.get("key_factors"),
        "blocking_factor": data.get("blocking_factor"),
    }


def _parse_voice_opinion(raw: Any) -> VoiceOpinion:
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif hasattr(raw, "dict"):
        data = raw.dict()
    else:
        data = dict(raw)

    desire_for_credit = _clamp_score(data.get("desire_for_credit"), default=0.5)
    need_for_credit = _clamp_score(data.get("need_for_credit"), default=0.4)
    readiness_now = _clamp_score(
        data.get("readiness_now"),
        default=desire_for_credit * 0.5 + need_for_credit * 0.3,
    )
    risk_pressure = _clamp_score(
        data.get("risk_pressure"),
        default=max(0.2, 0.6 - readiness_now * 0.2),
    )
    confidence = _clamp_score(data.get("confidence"), default=0.55)
    readiness_level = data.get("readiness_level") or _readiness_level_from_score(readiness_now)
    stance = _coerce_stance(data.get("stance")) or _derive_voice_stance(
        desire_for_credit=desire_for_credit,
        need_for_credit=need_for_credit,
        readiness_now=readiness_now,
        risk_pressure=risk_pressure,
        confidence=confidence,
    )

    payload = {
        "stance": stance,
        "readiness_level": readiness_level,
        "desire_for_credit": desire_for_credit,
        "need_for_credit": need_for_credit,
        "readiness_now": readiness_now,
        "risk_pressure": risk_pressure,
        "confidence": confidence,
        "theses": _coerce_text_list(data.get("theses"), fallback="позиция сформулирована нечетко"),
        "key_factors": _coerce_text_list(data.get("key_factors"), fallback="факторы были обозначены слишком общо"),
        "blocking_factor": str(data.get("blocking_factor") or data.get("risk_note") or "").strip() or None,
    }
    if hasattr(VoiceOpinion, "model_validate"):
        return VoiceOpinion.model_validate(payload)
    return VoiceOpinion(**payload)


def _parse_synthesis_narrative(raw: Any) -> SynthesizedDecisionNarrative:
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
    elif hasattr(raw, "dict"):
        data = raw.dict()
    else:
        data = dict(raw)
    payload = {
        "reasoning": str(data.get("reasoning") or "Итоговое объяснение было сформулировано неявно.").strip(),
        "emotional_factors": str(data.get("emotional_factors") or "Эмоциональные факторы были выражены слишком слабо.").strip(),
        "conflict_summary": data.get("conflict_summary"),
    }
    if hasattr(SynthesizedDecisionNarrative, "model_validate"):
        return SynthesizedDecisionNarrative.model_validate(payload)
    return SynthesizedDecisionNarrative(**payload)


def _synthesized_from_data(data: Dict[str, Any]) -> SynthesizedCreditDecision:
    if hasattr(SynthesizedCreditDecision, "model_validate"):
        return SynthesizedCreditDecision.model_validate(data)
    return SynthesizedCreditDecision(**data)


def _annotate_decision(
    decision: Any,
    decision_mode: str,
    voice_stances: Optional[Dict[str, str]] = None,
    conflict_summary: Optional[str] = None,
) -> DecisionOutcome:
    if hasattr(decision, "model_dump"):
        data = decision.model_dump()
    elif hasattr(decision, "dict"):
        data = decision.dict()
    else:
        data = dict(decision)
    data["decision_mode"] = _decision_mode_value(decision_mode)
    data["voice_stances"] = voice_stances
    data["conflict_summary"] = conflict_summary
    return _decision_from_data(data)


class EmotionalDecisionVoice:
    def build_prompt(self, context: Dict[str, Any]) -> str:
        lines = [
            "Ты эмоциональный голос клиента перед решением о кредите прямо сейчас.",
            "Смотри на желание откликнуться на предложение, импульс, тревогу, доверие, ощущение риска и внутреннюю готовность действовать.",
            "Важно: интерес к продукту не равен готовности оформить кредит.",
            "Верни JSON по схеме со следующими полями: stance, desire_for_credit, need_for_credit, readiness_now, risk_pressure, confidence, theses, key_factors, blocking_factor.",
            "Все числовые поля верни в диапазоне 0..1. theses и key_factors: от 1 до 3 очень коротких пунктов.",
            _serialize_lines("Профиль", context.get("profile")),
            _serialize_lines("Цель клиента", context.get("goal_description")),
            _serialize_lines("Тип цели", context.get("goal_intent")),
            _serialize_lines("Последние шаги", context.get("session_history")),
            _serialize_lines("Финальное эмоциональное состояние", context.get("final_emotional_state")),
            _serialize_lines("Push банка", context.get("push_message")),
            _serialize_lines("Тип реакции на push", context.get("reaction_type")),
            "Если клиенту предложение нравится, но он еще не готов оформить, это должно отражаться в высоком desire_for_credit и умеренном/низком readiness_now.",
        ]
        if context.get("news_context"):
            lines.extend(
                [
                    _serialize_lines("Внешний новостной фон", context.get("news_context")),
                    _serialize_lines("Сигналы внешнего фона", context.get("news_signals")),
                    "Внешний новостной фон — это окружение клиента, а не часть его профиля. Негативный фон может повышать тревогу и risk_pressure, а позитивные возможности могут слегка повышать desire_for_credit, но не заменяют реальную готовность.",
                ]
            )
        return "\n".join(lines)

    async def run(self, context: Dict[str, Any]) -> Tuple[VoiceOpinion, int]:
        prompt = self.build_prompt(context)
        opinion = await robust_llm_call(prompt, structured_output=VoiceOpinion, temperature=0.2)
        parsed = _parse_voice_opinion(opinion)
        parsed.stance = _derive_emotional_stance(parsed)
        return parsed, len(prompt)


class RationalDecisionVoice:
    def build_prompt(self, context: Dict[str, Any]) -> str:
        lines = [
            "Ты рациональный голос клиента перед решением о кредите прямо сейчас.",
            "Смотри на объективную необходимость кредита, уместность кредита для цели, риски долговой нагрузки и реалистичность действия прямо сейчас.",
            "Если цель в основном информационная, интерес к кредитным продуктам не должен автоматически означать готовность оформить кредит.",
            "Верни JSON по схеме со следующими полями: stance, desire_for_credit, need_for_credit, readiness_now, risk_pressure, confidence, theses, key_factors, blocking_factor.",
            "Все числовые поля верни в диапазоне 0..1. theses и key_factors: от 1 до 3 очень коротких пунктов.",
            _serialize_lines("Профиль", context.get("profile")),
            _serialize_lines("Цель клиента", context.get("goal_description")),
            _serialize_lines("Тип цели", context.get("goal_intent")),
            _serialize_lines("Последние шаги", context.get("session_history")),
            _serialize_lines("Финальное эмоциональное состояние", context.get("final_emotional_state")),
            _serialize_lines("Push банка", context.get("push_message")),
            _serialize_lines("Тип реакции на push", context.get("reaction_type")),
            "Если кредит не нужен для текущей цели, держи need_for_credit и readiness_now низкими, даже если предложение выглядит привлекательным.",
        ]
        if context.get("news_context"):
            lines.extend(
                [
                    _serialize_lines("Внешний новостной фон", context.get("news_context")),
                    _serialize_lines("Сигналы внешнего фона", context.get("news_signals")),
                    "Учитывай внешний новостной фон как внешнюю среду. Негативный кредитный фон увеличивает рациональные риски и снижает уместность кредита, особенно если цель клиента пока только информационная.",
                ]
            )
        return "\n".join(lines)

    async def run(self, context: Dict[str, Any]) -> Tuple[VoiceOpinion, int]:
        prompt = self.build_prompt(context)
        opinion = await robust_llm_call(prompt, structured_output=VoiceOpinion, temperature=0.1)
        parsed = _parse_voice_opinion(opinion)
        parsed.stance = _derive_rational_stance(parsed)
        return parsed, len(prompt)


class SynthesisNarrativeAgent:
    def build_prompt(
        self,
        context: Dict[str, Any],
        emotional_voice: VoiceOpinion,
        rational_voice: VoiceOpinion,
        precomputed_decision: Dict[str, Any],
        conflict_required: bool,
    ) -> str:
        conflict_rule = (
            "У голосов есть конфликт или тонкое расхождение. Поле conflict_summary заполни явно."
            if conflict_required
            else "Если явного конфликта нет, conflict_summary можешь оставить пустым или null."
        )
        lines = [
            "Ты объясняешь уже рассчитанное решение о кредите.",
            "Не пересчитывай will_take_credit и probability_score, а только кратко и связно объясни их.",
            "Важно сохранить эмоциональную тонкость и отдельно показать, чем отличаются эмоциональный и рациональный голоса.",
            conflict_rule,
            _serialize_lines("Профиль", context.get("profile")),
            _serialize_lines("Цель клиента", context.get("goal_description")),
            _serialize_lines("Тип цели", context.get("goal_intent")),
            _serialize_lines("Последние шаги", context.get("session_history")),
            _serialize_lines("Финальное эмоциональное состояние", context.get("final_emotional_state")),
            _serialize_lines("Push банка", context.get("push_message")),
            _serialize_lines("Тип реакции на push", context.get("reaction_type")),
            _serialize_lines("Эмоциональный голос", _voice_snapshot(emotional_voice)),
            _serialize_lines("Рациональный голос", _voice_snapshot(rational_voice)),
            _serialize_lines("Уже рассчитанный итог", precomputed_decision),
            "Верни JSON только с полями reasoning, emotional_factors и conflict_summary.",
        ]
        if context.get("news_context"):
            lines.extend(
                [
                    _serialize_lines("Сигналы внешнего фона", context.get("news_signals")),
                    "Если news-context повлиял на решение, кратко и явно отрази это в reasoning. Не пересказывай весь новостной контекст заново, а используй только уже рассчитанные сигналы внешнего фона.",
                ]
            )
        return "\n".join(lines)

    async def run(
        self,
        context: Dict[str, Any],
        emotional_voice: VoiceOpinion,
        rational_voice: VoiceOpinion,
        precomputed_decision: Dict[str, Any],
        conflict_required: bool,
    ) -> Tuple[SynthesizedDecisionNarrative, int]:
        prompt = self.build_prompt(
            context=context,
            emotional_voice=emotional_voice,
            rational_voice=rational_voice,
            precomputed_decision=precomputed_decision,
            conflict_required=conflict_required,
        )
        response = await robust_llm_call(prompt, structured_output=SynthesizedDecisionNarrative, temperature=0.0)
        parsed = _parse_synthesis_narrative(response)
        return parsed, len(prompt)


class CreditReasoningAgent:
    def __init__(self):
        self.emotional_voice = EmotionalDecisionVoice()
        self.rational_voice = RationalDecisionVoice()
        self.synthesis_agent = SynthesisNarrativeAgent()

    def _build_context(
        self,
        profile: Dict[str, Any],
        persona_history: List[Dict[str, Any]],
        emotional_state: Dict[str, Any],
        push_info: Dict[str, Any],
        goal: Optional[Dict[str, Any]] = None,
        reaction: Optional[Dict[str, Any]] = None,
        news_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        push = push_info.get("push", {}) if isinstance(push_info, dict) else {}
        reaction_payload = _as_clean_dict(reaction)
        compact_history = _compact_history(persona_history, limit=2)
        goal_description = _as_clean_dict(goal).get("goal_description")
        compact_news = _compact_news_context(news_context)
        news_signals = _derive_news_signals(compact_news)
        return {
            "profile": _compact_profile(_as_clean_dict(profile)),
            "goal_description": goal_description,
            "goal_intent": _infer_goal_intent(goal_description, compact_history),
            "session_history": compact_history,
            "final_emotional_state": _as_clean_dict(emotional_state),
            "push_message": _as_clean_dict(push).get("message"),
            "reaction_type": reaction_payload.get("reaction_type"),
            "news_context": compact_news,
            "news_signals": news_signals,
        }

    def _intent_thresholds(self, goal_intent: str) -> Dict[str, float]:
        if goal_intent == "application":
            return {"decision": 0.53, "need_gate": 0.34, "readiness_gate": 0.48, "intent_penalty": 0.02}
        if goal_intent == "exploratory":
            return {"decision": 0.57, "need_gate": 0.40, "readiness_gate": 0.50, "intent_penalty": 0.08}
        return {"decision": 0.61, "need_gate": 0.46, "readiness_gate": 0.56, "intent_penalty": 0.16}

    def _compute_decision(
        self,
        context: Dict[str, Any],
        emotional_voice: VoiceOpinion,
        rational_voice: VoiceOpinion,
    ) -> Dict[str, Any]:
        goal_intent = str(context.get("goal_intent") or "exploratory")
        thresholds = self._intent_thresholds(goal_intent)
        news_signals = context.get("news_signals") or {}
        news_credit_headwind = _clamp_score(news_signals.get("credit_headwind"), default=0.0)
        news_opportunity_support = _clamp_score(news_signals.get("opportunity_support"), default=0.0)
        news_stress_bias = _clamp_score(news_signals.get("stress_bias"), default=0.0)
        news_trust_drag = _clamp_score(news_signals.get("trust_drag"), default=0.0)
        news_credit_relevance = _clamp_score(news_signals.get("credit_relevance"), default=0.0)

        avg_desire = _mean([emotional_voice.desire_for_credit, rational_voice.desire_for_credit])
        avg_need = _mean([emotional_voice.need_for_credit, rational_voice.need_for_credit])
        avg_readiness = _mean([emotional_voice.readiness_now, rational_voice.readiness_now])
        max_readiness = max(emotional_voice.readiness_now, rational_voice.readiness_now)
        avg_risk = _mean([emotional_voice.risk_pressure, rational_voice.risk_pressure])
        avg_confidence = _mean([emotional_voice.confidence, rational_voice.confidence])
        readiness_gap = abs(emotional_voice.readiness_now - rational_voice.readiness_now)
        need_gap = abs(emotional_voice.need_for_credit - rational_voice.need_for_credit)
        desire_gap = abs(emotional_voice.desire_for_credit - rational_voice.desire_for_credit)
        interest_without_need = max(0.0, avg_desire - avg_need)

        emotional_pull = (
            0.50 * emotional_voice.desire_for_credit
            + 0.30 * emotional_voice.readiness_now
            + 0.20 * emotional_voice.confidence
        )
        rational_pull = (
            0.50 * rational_voice.need_for_credit
            + 0.30 * rational_voice.readiness_now
            + 0.20 * rational_voice.confidence
        )
        motivation_score = (
            0.34 * avg_desire
            + 0.36 * avg_need
            + 0.20 * avg_readiness
            + 0.10 * avg_confidence
            + 0.08 * news_opportunity_support
        )
        caution_score = (
            0.48 * avg_risk
            + 0.08 * readiness_gap
            + 0.05 * need_gap
            + 0.04 * desire_gap
            + 0.10 * interest_without_need
            + thresholds["intent_penalty"]
            + 0.12 * news_credit_headwind
            + 0.06 * news_stress_bias
            + 0.04 * news_trust_drag
        )
        if rational_voice.stance == "reject_credit":
            caution_score += 0.08
        if emotional_voice.stance == "take_credit" and rational_voice.stance != "take_credit":
            caution_score += 0.04
        if avg_need < 0.35:
            caution_score += 0.05
        if goal_intent == "application" and avg_need > 0.58:
            motivation_score += 0.05
        if goal_intent == "informational" and news_credit_relevance > 0.60:
            caution_score += 0.03 * news_credit_headwind
        if goal_intent == "application":
            motivation_score += 0.03 * news_opportunity_support

        probability_score = _clamp_score(0.50 + motivation_score - caution_score, default=0.5)

        will_take_credit = (
            probability_score >= thresholds["decision"]
            and rational_voice.need_for_credit >= thresholds["need_gate"]
            and max_readiness >= thresholds["readiness_gate"]
            and avg_risk < 0.78
        )

        score_breakdown = {
            "goal_intent": goal_intent,
            "avg_desire": round(avg_desire, 3),
            "avg_need": round(avg_need, 3),
            "avg_readiness": round(avg_readiness, 3),
            "max_readiness": round(max_readiness, 3),
            "avg_risk": round(avg_risk, 3),
            "avg_confidence": round(avg_confidence, 3),
            "emotional_pull": round(emotional_pull, 3),
            "rational_pull": round(rational_pull, 3),
            "motivation_score": round(motivation_score, 3),
            "caution_score": round(caution_score, 3),
            "probability_score": round(probability_score, 3),
            "decision_threshold": thresholds["decision"],
            "news_credit_relevance": round(news_credit_relevance, 3),
            "news_credit_headwind": round(news_credit_headwind, 3),
            "news_opportunity_support": round(news_opportunity_support, 3),
            "news_stress_bias": round(news_stress_bias, 3),
            "news_trust_drag": round(news_trust_drag, 3),
        }
        return {
            "will_take_credit": will_take_credit,
            "probability_score": probability_score,
            "score_breakdown": score_breakdown,
        }

    def _requires_llm_narrative(
        self,
        emotional_voice: VoiceOpinion,
        rational_voice: VoiceOpinion,
        computed: Dict[str, Any],
    ) -> bool:
        probability_score = float(computed["probability_score"])
        score_breakdown = computed["score_breakdown"]
        return any(
            [
                bool(computed["will_take_credit"]),
                emotional_voice.stance != rational_voice.stance,
                emotional_voice.readiness_level != rational_voice.readiness_level,
                abs(emotional_voice.need_for_credit - rational_voice.need_for_credit) > 0.15,
                abs(emotional_voice.readiness_now - rational_voice.readiness_now) > 0.15,
                score_breakdown["avg_desire"] - score_breakdown["avg_need"] > 0.18,
                0.40 <= probability_score <= 0.68,
            ]
        )

    def _build_template_narrative(
        self,
        context: Dict[str, Any],
        emotional_voice: VoiceOpinion,
        rational_voice: VoiceOpinion,
        computed: Dict[str, Any],
        conflict_required: bool,
    ) -> SynthesizedDecisionNarrative:
        goal_intent = context.get("goal_intent")
        score_breakdown = computed["score_breakdown"]
        decision_text = (
            "Клиент скорее готов взять кредит сейчас."
            if computed["will_take_credit"]
            else "Клиент пока не готов брать кредит прямо сейчас."
        )
        news_context = context.get("news_context") or {}
        news_phrase = ""
        if news_context:
            main_risk = (news_context.get("risks") or [None])[0]
            news_phrase = (
                f" Внешний новостной фон для аудитории сейчас {news_context.get('overall_reaction')}; "
                f"главный внешний риск: {main_risk or 'общая экономическая неопределенность'}."
            )
        reasoning = (
            f"{decision_text} Среднее желание откликнуться на предложение оценивается в {score_breakdown['avg_desire']}, "
            f"необходимость кредита для цели — в {score_breakdown['avg_need']}, готовность действовать сейчас — в {score_breakdown['avg_readiness']}, "
            f"а риск и осторожность — в {score_breakdown['avg_risk']}. "
            f"Тип цели: {goal_intent}. "
            f"Эмоциональный голос видит {', '.join(emotional_voice.theses[:2])}. "
            f"Рациональный голос подчеркивает {', '.join(rational_voice.theses[:2])}."
            f"{news_phrase}"
        ).strip()
        emotional_factors = (
            f"Эмоциональный вклад определяется сочетанием желания ({round(emotional_voice.desire_for_credit, 2)}), "
            f"готовности ({round(emotional_voice.readiness_now, 2)}) и субъективного риска ({round(emotional_voice.risk_pressure, 2)}). "
            f"Главный сдерживающий фактор: {emotional_voice.blocking_factor or rational_voice.blocking_factor or 'неопределенность в уместности кредита'}."
        )
        conflict_summary = None
        if conflict_required:
            conflict_summary = (
                f"Эмоциональный голос={emotional_voice.stance}/{emotional_voice.readiness_level}, "
                f"рациональный голос={rational_voice.stance}/{rational_voice.readiness_level}; "
                "итог сформирован по балансу мотивации, необходимости и риска."
            )
        return SynthesizedDecisionNarrative(
            reasoning=reasoning,
            emotional_factors=emotional_factors,
            conflict_summary=conflict_summary,
        )

    def _apply_guardrails(
        self,
        context: Dict[str, Any],
        emotional_voice: VoiceOpinion,
        rational_voice: VoiceOpinion,
        synthesized: SynthesizedCreditDecision,
    ) -> Tuple[SynthesizedCreditDecision, List[str]]:
        if not synthesized.will_take_credit:
            return synthesized, []

        guardrails: List[str] = []
        goal_intent = context.get("goal_intent")
        avg_need = _mean([emotional_voice.need_for_credit, rational_voice.need_for_credit])
        max_readiness = max(emotional_voice.readiness_now, rational_voice.readiness_now)
        avg_risk = _mean([emotional_voice.risk_pressure, rational_voice.risk_pressure])
        avg_desire = _mean([emotional_voice.desire_for_credit, rational_voice.desire_for_credit])
        news_signals = context.get("news_signals") or {}

        if goal_intent == "informational" and rational_voice.need_for_credit < 0.45:
            guardrails.append("informational_goal_requires_need")
        if rational_voice.stance == "reject_credit":
            guardrails.append("rational_reject_blocks_positive_decision")
        if max_readiness < 0.52:
            guardrails.append("no_voice_ready_now")
        if avg_desire - avg_need > 0.18 and goal_intent != "application":
            guardrails.append("interest_without_need")
        if avg_risk > 0.74:
            guardrails.append("risk_pressure_too_high")
        if goal_intent != "application" and _clamp_score(news_signals.get("credit_headwind"), default=0.0) > 0.68:
            guardrails.append("negative_news_headwind")

        if not guardrails:
            return synthesized, []

        payload = synthesized.model_dump() if hasattr(synthesized, "model_dump") else synthesized.dict()
        payload["will_take_credit"] = False
        payload["probability_score"] = min(float(payload.get("probability_score", 0.5)), 0.45)
        guardrail_note = _build_guardrail_note(guardrails, context)
        reasoning = str(payload.get("reasoning") or "").strip()
        emotional_factors = str(payload.get("emotional_factors") or "").strip()
        conflict_summary = str(payload.get("conflict_summary") or "").strip()
        payload["reasoning"] = f"{reasoning} {guardrail_note}".strip()
        payload["emotional_factors"] = f"{emotional_factors} {guardrail_note}".strip()
        payload["conflict_summary"] = (
            f"{conflict_summary} {guardrail_note}".strip()
            if conflict_summary
            else guardrail_note
        )
        return _synthesized_from_data(payload), guardrails

    async def make_final_decision(
        self,
        profile: Dict[str, Any],
        persona_history: List[Dict[str, Any]],
        emotional_state: Dict[str, Any],
        push_info: Dict[str, Any],
        goal: Optional[Dict[str, Any]] = None,
        reaction: Optional[Dict[str, Any]] = None,
        news_context: Optional[Dict[str, Any]] = None,
    ) -> DecisionOutcome:
        decision, _ = await self.make_final_decision_with_trace(
            profile=profile,
            persona_history=persona_history,
            emotional_state=emotional_state,
            push_info=push_info,
            goal=goal,
            reaction=reaction,
            news_context=news_context,
        )
        return decision

    async def make_final_decision_with_trace(
        self,
        profile: Dict[str, Any],
        persona_history: List[Dict[str, Any]],
        emotional_state: Dict[str, Any],
        push_info: Dict[str, Any],
        goal: Optional[Dict[str, Any]] = None,
        reaction: Optional[Dict[str, Any]] = None,
        news_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DecisionOutcome, Dict[str, Any]]:
        context = self._build_context(
            profile,
            persona_history,
            emotional_state,
            push_info,
            goal=goal,
            reaction=reaction,
            news_context=news_context,
        )

        emotional_voice, emotional_chars = await self.emotional_voice.run(context)
        rational_voice, rational_chars = await self.rational_voice.run(context)

        conflict_required = (
            emotional_voice.stance != rational_voice.stance
            or emotional_voice.readiness_level != rational_voice.readiness_level
            or abs(emotional_voice.need_for_credit - rational_voice.need_for_credit) > 0.15
            or abs(emotional_voice.readiness_now - rational_voice.readiness_now) > 0.15
            or emotional_voice.confidence < 0.55
            or rational_voice.confidence < 0.55
        )

        computed = self._compute_decision(context, emotional_voice, rational_voice)
        needs_llm_narrative = self._requires_llm_narrative(emotional_voice, rational_voice, computed)

        synthesis_chars = 0
        llm_calls = 2
        narrative_mode = "template"
        if needs_llm_narrative:
            narrative, synthesis_chars = await self.synthesis_agent.run(
                context=context,
                emotional_voice=emotional_voice,
                rational_voice=rational_voice,
                precomputed_decision=computed,
                conflict_required=conflict_required,
            )
            llm_calls += 1
            narrative_mode = "llm"
        else:
            narrative = self._build_template_narrative(
                context=context,
                emotional_voice=emotional_voice,
                rational_voice=rational_voice,
                computed=computed,
                conflict_required=conflict_required,
            )

        synthesized = _synthesized_from_data(
            {
                "will_take_credit": computed["will_take_credit"],
                "probability_score": computed["probability_score"],
                "reasoning": narrative.reasoning,
                "emotional_factors": narrative.emotional_factors,
                "conflict_summary": narrative.conflict_summary,
            }
        )
        synthesized, applied_guardrails = self._apply_guardrails(
            context=context,
            emotional_voice=emotional_voice,
            rational_voice=rational_voice,
            synthesized=synthesized,
        )

        conflict_summary = synthesized.conflict_summary
        if conflict_required and not conflict_summary:
            conflict_summary = (
                f"Эмоциональный голос={emotional_voice.stance}/{emotional_voice.readiness_level}, "
                f"рациональный голос={rational_voice.stance}/{rational_voice.readiness_level}; "
                "итог сформирован на основе баланса желания, необходимости, готовности и риска."
            )

        decision = _annotate_decision(
            decision=synthesized,
            decision_mode="compact_debate",
            voice_stances={
                "emotional": emotional_voice.stance,
                "rational": rational_voice.stance,
            },
            conflict_summary=conflict_summary,
        )

        trace = {
            "decision_mode": "compact_debate",
            "prompt_chars": emotional_chars + rational_chars + synthesis_chars,
            "prompt_char_counts": {
                "emotional": emotional_chars,
                "rational": rational_chars,
                "synthesis": synthesis_chars,
            },
            "llm_calls": llm_calls,
            "narrative_mode": narrative_mode,
            "context_summary": {
                "goal_intent": context.get("goal_intent"),
                "reaction_type": context.get("reaction_type"),
                "news_snapshot_id": (context.get("news_context") or {}).get("snapshot_id"),
            },
            "score_breakdown": computed["score_breakdown"],
            "guardrails_applied": applied_guardrails,
            "news_context": context.get("news_context"),
            "news_signals": context.get("news_signals"),
            "voices": {
                "emotional": emotional_voice.model_dump() if hasattr(emotional_voice, "model_dump") else emotional_voice.dict(),
                "rational": rational_voice.model_dump() if hasattr(rational_voice, "model_dump") else rational_voice.dict(),
            },
        }
        return decision, trace
