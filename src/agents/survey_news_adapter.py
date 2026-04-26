from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.schemas.news_context_schema import NewsContextSnapshot, normalize_news_context_payload


TOPICS = [
    "financial_risk",
    "financial_self_view",
    "trust_media",
    "advertising_attitudes",
    "values_norms",
    "openness_social",
    "general",
]

TOPIC_KEYWORDS = {
    "financial_risk": [
        "кредит",
        "ипотек",
        "ставк",
        "бан",
        "долг",
        "заем",
        "инвест",
        "облигац",
        "акци",
        "инфляц",
        "риск",
    ],
    "financial_self_view": [
        "доход",
        "зарплат",
        "работ",
        "обеспеч",
        "деньг",
        "финанс",
        "расход",
        "сбереж",
        "цен",
        "инфляц",
        "кредит",
    ],
    "trust_media": [
        "медиа",
        "сми",
        "газет",
        "новост",
        "источник",
        "канал",
        "редакц",
        "информац",
        "пропаганд",
    ],
    "advertising_attitudes": [
        "реклам",
        "бренд",
        "маркетинг",
        "продвижен",
        "кампан",
        "объявлен",
    ],
    "values_norms": [
        "ценност",
        "морал",
        "традиц",
        "роль",
        "норма",
        "семь",
        "правильн",
        "обязан",
    ],
    "openness_social": [
        "культур",
        "обыча",
        "иностран",
        "мигран",
        "общество",
        "разнообраз",
        "других людей",
        "другими людьми",
    ],
}

TOPIC_BRIDGES = {
    "financial_risk": {"financial_risk": 1.0, "financial_self_view": 0.72},
    "financial_self_view": {"financial_self_view": 1.0, "financial_risk": 0.78},
    "trust_media": {"trust_media": 1.0, "advertising_attitudes": 0.25},
    "advertising_attitudes": {"advertising_attitudes": 1.0, "trust_media": 0.35},
    "values_norms": {"values_norms": 1.0, "openness_social": 0.22},
    "openness_social": {"openness_social": 1.0, "values_norms": 0.22},
    "general": {"general": 1.0},
}

TOPIC_HEADWIND_SCALE = {
    "financial_risk": 1.0,
    "financial_self_view": 0.92,
    "trust_media": 0.78,
    "advertising_attitudes": 0.62,
    "values_norms": 0.42,
    "openness_social": 0.35,
    "general": 0.55,
}

TOPIC_OPPORTUNITY_SCALE = {
    "financial_risk": 0.85,
    "financial_self_view": 0.78,
    "trust_media": 0.55,
    "advertising_attitudes": 0.55,
    "values_norms": 0.32,
    "openness_social": 0.28,
    "general": 0.45,
}


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _coerce_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def _shorten_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _keyword_score(texts: List[str], keywords: List[str]) -> float:
    haystack = " ".join(_normalize_text(text) for text in texts if _normalize_text(text))
    if not haystack:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in haystack)
    return min(1.0, hits / max(4.0, min(6.0, float(len(keywords)))))


def _base_biases(reaction: str) -> Dict[str, float]:
    reaction_value = str(reaction or "").strip().lower()
    return {
        "negative_bias": {"negative": 0.82, "neutral": 0.35, "positive": 0.12}.get(reaction_value, 0.35),
        "positive_bias": {"positive": 0.78, "neutral": 0.30, "negative": 0.12}.get(reaction_value, 0.20),
    }


def _horizon_factor(value: str) -> float:
    return {"short_term": 1.0, "medium_term": 0.85, "long_term": 0.65}.get(str(value or "").strip().lower(), 0.8)


class SurveyNewsAdapter:
    def _normalize_snapshot(self, raw: Optional[Dict[str, Any]]) -> Optional[NewsContextSnapshot]:
        if raw is None:
            return None
        if isinstance(raw, NewsContextSnapshot):
            return raw
        data = normalize_news_context_payload(raw)
        if hasattr(NewsContextSnapshot, "model_validate"):
            return NewsContextSnapshot.model_validate(data)
        return NewsContextSnapshot(**data)

    def _topic_affinities(self, data: Dict[str, Any]) -> Dict[str, float]:
        texts: List[str] = [
            data.get("question"),
            data.get("summary_text"),
            *data.get("factors", []),
            *data.get("risks", []),
            *data.get("opportunities", []),
            *data.get("audience_effects", []),
        ]
        for row in data.get("evidence", []):
            texts.append(row.get("topic"))
            texts.append(row.get("summary"))

        affinities = {topic: _keyword_score(texts, TOPIC_KEYWORDS.get(topic, [])) for topic in TOPICS if topic != "general"}
        general_score = max(0.12, min(0.45, 0.12 + 0.35 * max(affinities.values(), default=0.0)))
        affinities["general"] = general_score
        return affinities

    def _source_strength(self, target_topic: str, affinities: Dict[str, float]) -> float:
        bridge = TOPIC_BRIDGES.get(target_topic, {"general": 1.0})
        total = sum(affinities.get(source_topic, 0.0) * weight for source_topic, weight in bridge.items())
        return _clamp(total, default=0.0)

    def _compact_evidence_rows(
        self,
        rows: List[Dict[str, Any]],
        target_topic: str,
        source_strength: float,
    ) -> List[Dict[str, Any]]:
        keywords = TOPIC_KEYWORDS.get(target_topic, [])
        filtered: List[Dict[str, Any]] = []
        for row in sorted(
            rows,
            key=lambda item: (
                item.get("rank") is None,
                item.get("rank") if item.get("rank") is not None else 999,
            ),
        ):
            text = _normalize_text(f"{row.get('topic', '')} {row.get('summary', '')}")
            if keywords and any(keyword in text for keyword in keywords):
                filtered.append(row)
        selected = filtered[:2]
        if not selected and source_strength >= 0.45:
            selected = rows[:2]
        compact: List[Dict[str, Any]] = []
        for row in selected[:2]:
            compact.append(
                {
                    "topic": row.get("topic"),
                    "rank": row.get("rank"),
                    "summary": _shorten_text(row.get("summary"), limit=140),
                    "source_type": row.get("source_type"),
                    "source_datetime": row.get("source_datetime"),
                }
            )
        return compact

    def _topic_entry(self, topic: str, data: Dict[str, Any], affinities: Dict[str, float]) -> Dict[str, Any]:
        source_strength = self._source_strength(topic, affinities)
        confidence = _clamp(data.get("confidence"), default=0.5)
        horizon_factor = _horizon_factor(data.get("impact_horizon"))
        biases = _base_biases(data.get("overall_reaction"))

        risks = _coerce_text_list(data.get("risks"))
        opportunities = _coerce_text_list(data.get("opportunities"))
        audience_effects = _coerce_text_list(data.get("audience_effects"))
        evidence_rows = self._compact_evidence_rows(data.get("evidence", []), topic, source_strength)

        risk_density = min(1.0, len(risks) / 3.0)
        opportunity_density = min(1.0, len(opportunities) / 3.0)
        audience_density = min(1.0, len(audience_effects) / 3.0)
        evidence_density = min(1.0, len(evidence_rows) / 2.0)

        question_relevance = round(source_strength, 3)
        headwind = _clamp(
            (0.52 * biases["negative_bias"] + 0.20 * risk_density + 0.18 * audience_density + 0.10 * evidence_density)
            * question_relevance
            * confidence
            * horizon_factor
            * TOPIC_HEADWIND_SCALE.get(topic, 0.5),
            default=0.0,
        )
        opportunity_support = _clamp(
            (0.50 * biases["positive_bias"] + 0.35 * opportunity_density + 0.15 * evidence_density)
            * question_relevance
            * confidence
            * horizon_factor
            * TOPIC_OPPORTUNITY_SCALE.get(topic, 0.4),
            default=0.0,
        )
        stability_support = _clamp(
            (0.44 * (1.0 - biases["negative_bias"]) + 0.16 * biases["positive_bias"] + 0.15 * opportunity_density)
            * question_relevance
            * confidence,
            default=0.0,
        )
        return {
            "summary_text": _shorten_text(data.get("summary_text"), limit=260),
            "factors": _coerce_text_list(data.get("factors"))[:3],
            "risks": risks[:3],
            "opportunities": opportunities[:3],
            "audience_effects": audience_effects[:3],
            "evidence": evidence_rows,
            "question_relevance": question_relevance,
            "negative_bias": round(biases["negative_bias"], 3),
            "positive_bias": round(biases["positive_bias"], 3),
            "headwind": round(headwind, 3),
            "opportunity_support": round(opportunity_support, 3),
            "stability_support": round(stability_support, 3),
            "source_topic_affinity": round(source_strength, 3),
        }

    def _audience_alignment(self, snapshot_audience: str, persona_context: Dict[str, Any]) -> float:
        audience_text = _normalize_text(snapshot_audience)
        persona_audience = _normalize_text(persona_context.get("target_audience_name"))
        age_group = _normalize_text(persona_context.get("age_group"))

        if not audience_text:
            return 0.75
        if persona_audience and (persona_audience in audience_text or audience_text in persona_audience):
            return 1.0
        if "пенсион" in audience_text:
            if any(token in age_group for token in ["60", "65", "70"]):
                return 0.72
            return 0.16
        if ("матер" in audience_text or "мам" in audience_text) and persona_audience == "mothers":
            return 1.0
        if ("отц" in audience_text or "пап" in audience_text) and persona_audience == "fathers":
            return 1.0
        if any(token in audience_text for token in ["родител", "семь"]) and persona_audience in {"mothers", "fathers"}:
            return 0.72
        return 0.38 if persona_audience else 0.65

    def build(self, raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = self._normalize_snapshot(raw)
        if normalized is None:
            return {}

        data = _dump_model(normalized)
        affinities = self._topic_affinities(data)
        source_topic = max(TOPICS, key=lambda topic: affinities.get(topic, 0.0))
        compact_evidence = [
            {
                "topic": row.get("topic"),
                "rank": row.get("rank"),
                "summary": _shorten_text(row.get("summary"), limit=140),
                "source_type": row.get("source_type"),
                "source_datetime": row.get("source_datetime"),
            }
            for row in sorted(
                data.get("evidence", []),
                key=lambda item: (
                    item.get("rank") is None,
                    item.get("rank") if item.get("rank") is not None else 999,
                ),
            )[:3]
        ]

        return {
            "snapshot_id": data.get("snapshot_id"),
            "generated_at": data.get("generated_at"),
            "audience": data.get("audience"),
            "question": data.get("question"),
            "overall_reaction": data.get("overall_reaction"),
            "confidence": _clamp(data.get("confidence"), default=0.5),
            "impact_horizon": data.get("impact_horizon"),
            "summary_text": _shorten_text(data.get("summary_text"), limit=260),
            "source_topic": source_topic,
            "fictional_warning": bool(data.get("fictional_warning", False)),
            "evidence": compact_evidence,
            "topic_contexts": {
                topic: self._topic_entry(topic, data, affinities)
                for topic in TOPICS
            },
        }

    def resolve(
        self,
        adapted_context: Optional[Dict[str, Any]],
        target_topic: str,
        persona_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(adapted_context, dict):
            return {
                "selected_context": {},
                "signals": {
                    "question_topic": target_topic,
                    "question_relevance": 0.0,
                    "negative_bias": 0.0,
                    "positive_bias": 0.0,
                    "headwind": 0.0,
                    "opportunity_support": 0.0,
                    "stability_support": 0.0,
                    "audience_alignment": 0.0,
                    "source_topic": None,
                },
            }

        topic_contexts = adapted_context.get("topic_contexts") or {}
        topic_payload = topic_contexts.get(target_topic) or {}
        audience_alignment = self._audience_alignment(adapted_context.get("audience"), persona_context)
        relevance = _clamp(topic_payload.get("question_relevance"), default=0.0) * audience_alignment
        headwind = _clamp(topic_payload.get("headwind"), default=0.0) * audience_alignment
        opportunity_support = _clamp(topic_payload.get("opportunity_support"), default=0.0) * audience_alignment
        stability_support = _clamp(topic_payload.get("stability_support"), default=0.0) * audience_alignment

        selected_context: Dict[str, Any] = {}
        if relevance >= 0.15:
            selected_context = {
                "snapshot_id": adapted_context.get("snapshot_id"),
                "audience": adapted_context.get("audience"),
                "question": adapted_context.get("question"),
                "source_topic": adapted_context.get("source_topic"),
                "overall_reaction": adapted_context.get("overall_reaction"),
                "confidence": adapted_context.get("confidence"),
                "impact_horizon": adapted_context.get("impact_horizon"),
                "topic_relevance": round(relevance, 3),
                "audience_alignment": round(audience_alignment, 3),
                "summary_text": topic_payload.get("summary_text"),
                "factors": topic_payload.get("factors"),
                "risks": topic_payload.get("risks"),
                "opportunities": topic_payload.get("opportunities"),
                "audience_effects": topic_payload.get("audience_effects"),
                "evidence": topic_payload.get("evidence"),
            }

        return {
            "selected_context": selected_context,
            "signals": {
                "question_topic": target_topic,
                "question_relevance": round(relevance, 3),
                "negative_bias": round(_clamp(topic_payload.get("negative_bias"), default=0.0), 3),
                "positive_bias": round(_clamp(topic_payload.get("positive_bias"), default=0.0), 3),
                "headwind": round(headwind, 3),
                "opportunity_support": round(opportunity_support, 3),
                "stability_support": round(stability_support, 3),
                "audience_alignment": round(audience_alignment, 3),
                "source_topic": adapted_context.get("source_topic"),
            },
        }
