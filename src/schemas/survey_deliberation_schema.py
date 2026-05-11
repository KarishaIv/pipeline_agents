from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SurveyVoiceOpinion(BaseModel):
    stance: Literal["agree", "disagree", "uncertain"] = Field(
        ...,
        description="Позиция голоса по отношению к утверждению.",
    )
    support_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько голос склоняется к согласию с утверждением.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность голоса в своей позиции.",
    )
    salience: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько вопрос значим именно для этой перспективы.",
    )
    conflict_pressure: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько внутри этой перспективы есть сомнение или внутреннее сопротивление.",
    )
    reasoning: str = Field(
        ...,
        min_length=40,
        description="Короткое объяснение позиции голоса.",
    )
    reaction: str = Field(
        ...,
        min_length=10,
        description="Короткая итоговая реакция от лица голоса.",
    )
    key_factors: List[str] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="Главные причины позиции.",
    )
    blocking_factor: Optional[str] = Field(
        default=None,
        description="Фактор, мешающий голосу полностью согласиться с утверждением.",
    )


class SurveyDecisionOutput(BaseModel):
    reasoning: str = Field(
        ...,
        min_length=80,
        description="Итоговое объяснение решения.",
    )
    decision: bool = Field(
        ...,
        description="Финальное решение: True — согласен, False — не согласен.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность в финальном решении.",
    )
    conflict_summary: Optional[str] = Field(
        default=None,
        description="Краткое описание конфликта между голосами, если он есть.",
    )
    voice_stances: Optional[Dict[str, str]] = Field(
        default=None,
        description="Итоговые позиции голосов для трассировки.",
    )
    score_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Диагностические числовые сигналы итогового решения.",
    )
    survey_mode: Optional[str] = Field(
        default=None,
        description="Режим survey reasoning.",
    )
    news_context_used: Optional[bool] = Field(
        default=None,
        description="Использовался ли внешний news-context.",
    )


class SurveyPersonaModel(BaseModel):
    financial_caution: float = Field(..., ge=0.0, le=1.0)
    economic_pressure: float = Field(..., ge=0.0, le=1.0)
    institutional_trust: float = Field(..., ge=0.0, le=1.0)
    media_skepticism: float = Field(..., ge=0.0, le=1.0)
    traditionalism: float = Field(..., ge=0.0, le=1.0)
    social_conformity: float = Field(..., ge=0.0, le=1.0)
    consumer_pragmatism: float = Field(..., ge=0.0, le=1.0)
    openness_to_change: float = Field(..., ge=0.0, le=1.0)
    professional_identity_salience: float = Field(..., ge=0.0, le=1.0)
    summary: List[str] = Field(..., min_items=1, max_items=6)


class SurveyConflictResolution(BaseModel):
    support_adjustment: float = Field(
        ...,
        ge=-0.15,
        le=0.15,
        description="Небольшая поправка к итоговому support score после conflict-aware synthesis.",
    )
    confidence_adjustment: float = Field(
        ...,
        ge=-0.12,
        le=0.12,
        description="Поправка к финальной уверенности.",
    )
    conflict_summary: str = Field(
        ...,
        min_length=30,
        description="Краткое описание ключевого внутреннего конфликта между голосами.",
    )
    dominant_tension: Optional[str] = Field(
        default=None,
        description="Главная линия напряжения: например, risk_vs_growth или trust_vs_skepticism.",
    )
