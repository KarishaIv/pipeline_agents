from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from credit_schemas import CreditNewsSignals, dump_model


CREDIT_KEYWORDS = [
    "кредит",
    "кредитн",
    "ипотек",
    "заем",
    "займ",
    "долг",
    "банк",
    "ставк",
    "платеж",
    "рассроч",
    "одобр",
    "заявк",
]

RISK_KEYWORDS = [
    "инфляц",
    "рост цен",
    "ставк",
    "подорож",
    "долг",
    "закредит",
    "риск",
    "нестабил",
    "сокращ",
    "безработ",
    "паден",
    "кризис",
]

OPPORTUNITY_KEYWORDS = [
    "льгот",
    "субсид",
    "поддерж",
    "программ",
    "выгод",
    "скид",
    "кэшб",
    "льготн",
    "одобр",
    "доступ",
]

TRUST_KEYWORDS = [
    "мошен",
    "обман",
    "недовер",
    "санкц",
    "утеч",
    "жалоб",
    "проблем",
    "навяз",
]


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _lower(value: Any) -> str:
    return _text(value).lower()


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _keyword_score(texts: Iterable[Any], keywords: List[str]) -> float:
    haystack = " ".join(_lower(text) for text in texts if _text(text))
    if not haystack:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return _clamp(hits / max(4.0, min(8.0, float(len(keywords)))))


def _normalize_news_payload(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None

    try:
        from src.schemas.news_context_schema import normalize_news_context_payload

        return normalize_news_context_payload(raw)
    except Exception:
        pass

    if "final_summary" in raw or "context_for_simulation" in raw:
        final_summary = raw.get("final_summary") if isinstance(raw.get("final_summary"), dict) else {}
        return {
            "snapshot_id": raw.get("snapshot_id") or raw.get("id") or "",
            "generated_at": raw.get("generated_at") or "",
            "audience": raw.get("target_audience") or raw.get("audience") or "",
            "question": raw.get("question") or "",
            "overall_reaction": final_summary.get("overall_reaction") or raw.get("overall_reaction") or "neutral",
            "confidence": final_summary.get("confidence") or raw.get("confidence") or 0.5,
            "impact_horizon": final_summary.get("impact_horizon") or raw.get("impact_horizon"),
            "summary_text": raw.get("summary_text") or final_summary.get("summary_text") or raw.get("context_for_simulation") or "",
            "factors": final_summary.get("factors") or raw.get("factors") or [],
            "risks": final_summary.get("risks") or raw.get("risks") or [],
            "opportunities": final_summary.get("opportunities") or raw.get("opportunities") or [],
            "audience_effects": final_summary.get("audience_effects") or raw.get("audience_effects") or [],
            "evidence": raw.get("evidence") or raw.get("agent_opinions") or [],
        }

    return dict(raw)


def _reaction_bias(reaction: str) -> Tuple[float, float]:
    reaction = _lower(reaction)
    if reaction == "negative":
        return 0.82, 0.12
    if reaction == "positive":
        return 0.12, 0.78
    return 0.35, 0.30


class CreditNewsAdapter:
    """Project a general news snapshot into signals useful for credit decisions."""

    def adapt(self, raw: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[CreditNewsSignals]]:
        data = _normalize_news_payload(raw)
        if data is None:
            return None, None

        evidence = data.get("evidence") or []
        evidence_texts: List[str] = []
        for row in _as_list(evidence):
            if isinstance(row, dict):
                evidence_texts.extend([_text(row.get("topic")), _text(row.get("summary")), _text(row.get("opinion"))])
            else:
                evidence_texts.append(_text(row))

        texts = [
            data.get("question"),
            data.get("summary_text"),
            *[str(item) for item in _as_list(data.get("factors"))],
            *[str(item) for item in _as_list(data.get("risks"))],
            *[str(item) for item in _as_list(data.get("opportunities"))],
            *[str(item) for item in _as_list(data.get("audience_effects"))],
            *evidence_texts,
        ]

        reaction = _lower(data.get("overall_reaction")) or "neutral"
        if reaction not in {"negative", "neutral", "positive"}:
            reaction = "neutral"
        negative_bias, positive_bias = _reaction_bias(reaction)
        confidence = _clamp(data.get("confidence"), default=0.5)
        relevance = max(_keyword_score(texts, CREDIT_KEYWORDS), 0.25 if any(_text(t) for t in texts) else 0.0)

        risk_score = _keyword_score(texts, RISK_KEYWORDS)
        opportunity_score = _keyword_score(texts, OPPORTUNITY_KEYWORDS)
        trust_score = _keyword_score(texts, TRUST_KEYWORDS)

        # Keep signals conservative: context can nudge explanations, but should not dominate user intent.
        credit_headwind = _clamp((0.58 * negative_bias + 0.42 * risk_score) * relevance * (0.55 + 0.45 * confidence))
        opportunity_support = _clamp((0.62 * positive_bias + 0.38 * opportunity_score) * relevance * (0.50 + 0.50 * confidence))
        stress_bias = _clamp((0.55 * negative_bias + 0.45 * risk_score) * (0.40 + 0.60 * relevance))
        trust_drag = _clamp((0.45 * negative_bias + 0.55 * trust_score) * (0.35 + 0.65 * relevance))

        summary = {
            "snapshot_id": data.get("snapshot_id", ""),
            "audience": data.get("audience", ""),
            "question": data.get("question", ""),
            "overall_reaction": reaction,
            "confidence": confidence,
            "impact_horizon": data.get("impact_horizon"),
            "summary_text": _text(data.get("summary_text"))[:700],
            "credit_relevance": round(relevance, 3),
            "top_risks": [_text(item) for item in _as_list(data.get("risks"))[:3]],
            "top_opportunities": [_text(item) for item in _as_list(data.get("opportunities"))[:3]],
        }

        signals = CreditNewsSignals(
            snapshot_id=str(data.get("snapshot_id") or ""),
            credit_relevance=round(relevance, 3),
            credit_headwind=round(credit_headwind, 3),
            opportunity_support=round(opportunity_support, 3),
            stress_bias=round(stress_bias, 3),
            trust_drag=round(trust_drag, 3),
            confidence=round(confidence, 3),
            overall_reaction=reaction,  # type: ignore[arg-type]
            impact_horizon=data.get("impact_horizon"),
        )
        return summary, signals


def adapt_credit_news(raw: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    summary, signals = CreditNewsAdapter().adapt(raw)
    return summary, dump_model(signals) if signals else None

