from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class VoiceOpinion(BaseModel):
    stance: Literal["take_credit", "reject_credit", "uncertain"] = Field(
        ...,
        description="Позиция голоса относительно кредита.",
    )
    readiness_level: Optional[Literal["browsing", "considering", "ready_now"]] = Field(
        default=None,
        description="Категориальная интерпретация readiness_now для удобства анализа.",
    )
    desire_for_credit: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько предложение эмоционально привлекательно для клиента.",
    )
    need_for_credit: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько кредит реально нужен клиенту для текущей цели.",
    )
    readiness_now: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько клиент готов действовать прямо сейчас, а не просто интересуется.",
    )
    risk_pressure: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Насколько клиента сдерживают риски и осторожность.",
    )
    theses: List[str] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="Короткие тезисы позиции, не более трех.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность голоса в своей позиции.",
    )
    key_factors: List[str] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="Главные факторы, определяющие позицию.",
    )
    blocking_factor: Optional[str] = Field(
        default=None,
        description="Главный фактор, который мешает взять кредит сейчас.",
    )


class SynthesizedDecisionNarrative(BaseModel):
    reasoning: str = Field(..., description="Итоговое объяснение уже посчитанного решения.")
    emotional_factors: str = Field(..., description="Какие эмоции и их оттенки повлияли на итог.")
    conflict_summary: Optional[str] = Field(
        default=None,
        description="Краткое объяснение конфликта между голосами, если он был.",
    )


class SynthesizedCreditDecision(BaseModel):
    will_take_credit: bool = Field(..., description="Примет ли клиент кредитное предложение.")
    probability_score: float = Field(..., ge=0.0, le=1.0, description="Итоговая уверенность решения.")
    reasoning: str = Field(..., description="Итоговое объяснение решения.")
    emotional_factors: str = Field(..., description="Какие эмоции и их оттенки повлияли на решение.")
    conflict_summary: Optional[str] = Field(
        default=None,
        description="Краткое описание конфликта между голосами, если он был.",
    )


class DecisionPacket(BaseModel):
    packet_id: str = Field(..., description="Идентификатор frozen decision packet.")
    source_run_path: str = Field(..., description="Путь до исходного full_run.json.")
    profile: Dict[str, Any] = Field(..., description="Профиль персоны.")
    goal: Dict[str, Any] = Field(..., description="Цель клиента в сессии.")
    session_history: List[Dict[str, Any]] = Field(..., description="История шагов в приложении.")
    final_emotional_state: Dict[str, Any] = Field(..., description="Финальное эмоциональное состояние перед решением.")
    push_info: Dict[str, Any] = Field(..., description="Push-уведомление и prediction банка.")
    reaction: Dict[str, Any] = Field(..., description="Реакция клиента на push.")
    baseline_decision: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Исходное решение credit baseline, если оно есть.",
    )
