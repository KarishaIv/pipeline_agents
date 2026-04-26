import hashlib
import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class NewsEvidenceItem(BaseModel):
    topic: str = Field(..., description="Тема evidence-блока.")
    rank: Optional[int] = Field(default=None, description="Порядок важности evidence.")
    summary: str = Field(..., description="Короткий текст evidence.")
    source_type: Optional[str] = Field(default=None, description="Тип источника.")
    source_id: Optional[str] = Field(default=None, description="Идентификатор источника.")
    source_url: Optional[str] = Field(default=None, description="Ссылка на источник, если есть.")
    source_datetime: Optional[str] = Field(default=None, description="Время публикации источника.")


class NewsContextSnapshot(BaseModel):
    snapshot_id: str = Field(..., description="Идентификатор news snapshot.")
    generated_at: str = Field(..., description="Время генерации snapshot.")
    audience: str = Field(..., description="Аудитория, к которой относится контекст.")
    question: str = Field(..., description="Вопрос, под который собран контекст.")
    overall_reaction: Literal["negative", "neutral", "positive"] = Field(
        ...,
        description="Общий знак внешнего фона.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Уверенность news-системы в общем выводе.",
    )
    impact_horizon: Optional[Literal["short_term", "medium_term", "long_term"]] = Field(
        default=None,
        description="Горизонт влияния news-context.",
    )
    summary_text: str = Field(..., description="Краткое summary news-context.")
    factors: List[str] = Field(default_factory=list, description="Главные причины общего фона.")
    risks: List[str] = Field(default_factory=list, description="Риски из внешнего контекста.")
    opportunities: List[str] = Field(default_factory=list, description="Потенциальные точки отклика.")
    audience_effects: List[str] = Field(
        default_factory=list,
        description="Как внешний фон влияет именно на эту аудиторию.",
    )
    evidence: List[NewsEvidenceItem] = Field(
        default_factory=list,
        description="Опорные evidence-элементы по новостному контексту.",
    )
    fictional_warning: Optional[bool] = Field(
        default=None,
        description="Флаг о возможной fictional / synthetic природе context.",
    )


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _coerce_text_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    text = _clean_text(value)
    return [text] if text else []


def _clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
        return max(0.0, min(1.0, numeric))
    except (TypeError, ValueError):
        pass

    text = _clean_text(value).lower()
    mapping = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.82,
        "низкий": 0.3,
        "средний": 0.6,
        "умеренный": 0.6,
        "высокий": 0.82,
    }
    return mapping.get(text, 0.5)


def _normalize_reaction(value: Any) -> str:
    text = _clean_text(value).lower()
    mapping = {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "негативный": "negative",
        "отрицательный": "negative",
        "настороженный": "negative",
        "neutral/ mixed": "neutral",
        "mixed": "neutral",
        "mixed/neutral": "neutral",
        "смешанный": "neutral",
        "нейтральный": "neutral",
        "positive/ mixed": "positive",
        "позитивный": "positive",
        "положительный": "positive",
    }
    return mapping.get(text, "neutral")


def _normalize_horizon(value: Any) -> Optional[str]:
    text = _clean_text(value).lower()
    if not text:
        return None
    mapping = {
        "short_term": "short_term",
        "medium_term": "medium_term",
        "long_term": "long_term",
        "краткосрочный": "short_term",
        "краткосрочно": "short_term",
        "среднесрочный": "medium_term",
        "среднесрочно": "medium_term",
        "долгосрочный": "long_term",
        "долгосрочно": "long_term",
    }
    return mapping.get(text)


def _parse_evidence_string(text: str, rank: int) -> Dict[str, Any]:
    body = _clean_text(text)
    match = re.match(r"^\[(.*?)\]\s*(.*)$", body)
    if match:
        topic = _clean_text(match.group(1)) or "news_context"
        summary = _clean_text(match.group(2)) or body
    else:
        topic = "news_context"
        summary = body
    return {
        "topic": topic,
        "rank": rank,
        "summary": summary,
        "source_type": None,
        "source_id": None,
        "source_url": None,
        "source_datetime": None,
    }


def _normalize_evidence(value: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return rows
    for idx, item in enumerate(value, start=1):
        if isinstance(item, str):
            parsed = _parse_evidence_string(item, idx)
            if parsed["summary"]:
                rows.append(parsed)
            continue
        if isinstance(item, dict):
            topic = _clean_text(item.get("topic")) or "news_context"
            summary = _clean_text(item.get("summary"))
            if not summary:
                joined = " ".join(
                    _clean_text(item.get(key))
                    for key in ("text", "content", "body", "opinion", "analysis")
                    if _clean_text(item.get(key))
                )
                summary = joined
            if not summary:
                continue
            rows.append(
                {
                    "topic": topic,
                    "rank": item.get("rank", idx),
                    "summary": summary,
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                    "source_url": item.get("source_url"),
                    "source_datetime": item.get("source_datetime"),
                }
            )
    return rows


def _coerce_result_package(data: Dict[str, Any]) -> Dict[str, Any]:
    if not any(key in data for key in ("final_summary", "context_for_simulation", "agent_opinions", "target_audience")):
        return data

    final_summary = data.get("final_summary") if isinstance(data.get("final_summary"), dict) else {}
    context_for_simulation = data.get("context_for_simulation")
    summary_text = _clean_text(
        data.get("summary_text")
        or final_summary.get("summary_text")
        or context_for_simulation
    )
    evidence_rows: List[Dict[str, Any]] = []
    opinions = data.get("agent_opinions")
    if isinstance(opinions, list):
        for idx, row in enumerate(opinions, start=1):
            if isinstance(row, dict):
                topic = _clean_text(row.get("topic") or row.get("agent") or row.get("name")) or "news_context"
                summary = _clean_text(row.get("summary") or row.get("opinion") or row.get("analysis"))
                if summary:
                    evidence_rows.append(
                        {
                            "topic": topic,
                            "rank": row.get("rank", idx),
                            "summary": summary,
                            "source_type": row.get("source_type"),
                            "source_id": row.get("source_id"),
                            "source_url": row.get("source_url"),
                            "source_datetime": row.get("source_datetime"),
                        }
                    )

    return {
        "snapshot_id": data.get("snapshot_id"),
        "generated_at": data.get("generated_at"),
        "audience": data.get("audience") or data.get("target_audience"),
        "question": data.get("question"),
        "overall_reaction": data.get("overall_reaction") or final_summary.get("sentiment"),
        "confidence": data.get("confidence") or final_summary.get("confidence"),
        "impact_horizon": data.get("impact_horizon") or final_summary.get("impact_horizon"),
        "summary_text": summary_text,
        "factors": data.get("factors") or final_summary.get("key_triggers") or [],
        "risks": data.get("risks") or final_summary.get("barriers") or [],
        "opportunities": data.get("opportunities") or final_summary.get("enablers") or [],
        "audience_effects": data.get("audience_effects") or final_summary.get("audience_effects") or [],
        "evidence": data.get("evidence") or evidence_rows,
        "fictional_warning": data.get("fictional_warning"),
    }


def normalize_news_context_payload(raw: Any) -> Dict[str, Any]:
    data = _dump_model(raw)
    data = _coerce_result_package(data)

    audience = _clean_text(data.get("audience"))
    question = _clean_text(data.get("question"))
    summary_text = _clean_text(data.get("summary_text"))
    factors = _coerce_text_list(data.get("factors"))
    risks = _coerce_text_list(data.get("risks"))
    opportunities = _coerce_text_list(data.get("opportunities"))
    audience_effects = _coerce_text_list(data.get("audience_effects"))
    evidence = _normalize_evidence(data.get("evidence"))

    if not summary_text:
        summary_text = _clean_text(
            ". ".join(part for part in [*factors[:2], *risks[:2], *opportunities[:2]] if part)
        )
    if not summary_text:
        summary_text = "Внешний новостной фон описан кратко и без детальной сводки."

    snapshot_id = _clean_text(data.get("snapshot_id"))
    if not snapshot_id:
        fingerprint = json.dumps(
            {
                "audience": audience,
                "question": question,
                "summary_text": summary_text,
                "factors": factors,
                "risks": risks,
                "opportunities": opportunities,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        snapshot_id = "compat_" + hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]

    generated_at = _clean_text(data.get("generated_at")) or "unknown"

    return {
        "snapshot_id": snapshot_id,
        "generated_at": generated_at,
        "audience": audience or "unspecified audience",
        "question": question or "unspecified question",
        "overall_reaction": _normalize_reaction(data.get("overall_reaction")),
        "confidence": _clamp_confidence(data.get("confidence")),
        "impact_horizon": _normalize_horizon(data.get("impact_horizon")),
        "summary_text": summary_text,
        "factors": factors,
        "risks": risks,
        "opportunities": opportunities,
        "audience_effects": audience_effects,
        "evidence": evidence,
        "fictional_warning": bool(data.get("fictional_warning", False)) if data.get("fictional_warning") is not None else None,
    }
