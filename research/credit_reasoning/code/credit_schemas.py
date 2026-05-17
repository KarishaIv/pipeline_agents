from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


CreditStance = Literal["take_credit", "reject_credit", "uncertain"]
ReadinessLevel = Literal["browsing", "considering", "ready"]
DecisionMode = Literal["direct", "compact_debate"]


class DecisionOutcome(BaseModel):
    """Final credit decision contract used by the historical credit branch."""

    model_config = ConfigDict(extra="allow")

    will_take_credit: bool
    probability_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    emotional_factors: str
    decision_mode: Optional[str] = None
    voice_stances: Dict[str, str] = Field(default_factory=dict)
    conflict_summary: str = ""
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    guardrails_applied: List[str] = Field(default_factory=list)
    news_context_used: bool = False


class VoiceOpinion(BaseModel):
    """One compact debate voice before final code-level aggregation."""

    model_config = ConfigDict(extra="allow")

    stance: CreditStance = "uncertain"
    desire_for_credit: float = Field(default=0.5, ge=0.0, le=1.0)
    need_for_credit: float = Field(default=0.5, ge=0.0, le=1.0)
    readiness_now: float = Field(default=0.5, ge=0.0, le=1.0)
    readiness_level: ReadinessLevel = "considering"
    risk_pressure: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    theses: List[str] = Field(default_factory=list)
    key_factors: List[str] = Field(default_factory=list)
    blocking_factor: Optional[str] = None


class SynthesizedCreditDecision(BaseModel):
    """Narrative layer written after voices and numeric aggregation."""

    model_config = ConfigDict(extra="allow")

    reasoning: str
    emotional_factors: str
    conflict_summary: str = ""


class CreditNewsSignals(BaseModel):
    """Credit-specific projection of an external news context snapshot."""

    model_config = ConfigDict(extra="allow")

    snapshot_id: str = ""
    credit_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    credit_headwind: float = Field(default=0.0, ge=0.0, le=1.0)
    opportunity_support: float = Field(default=0.0, ge=0.0, le=1.0)
    stress_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_drag: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    overall_reaction: Literal["negative", "neutral", "positive"] = "neutral"
    impact_horizon: Optional[str] = None


class DecisionPacket(BaseModel):
    """Frozen packet used for fair credit reasoning benchmarks."""

    model_config = ConfigDict(extra="allow")

    packet_id: str
    source_run_path: str = ""
    profile: Dict[str, Any] = Field(default_factory=dict)
    goal: Dict[str, Any] = Field(default_factory=dict)
    session_history: List[Dict[str, Any]] = Field(default_factory=list)
    final_emotional_state: Dict[str, Any] = Field(default_factory=dict)
    push_info: Dict[str, Any] = Field(default_factory=dict)
    reaction: Dict[str, Any] = Field(default_factory=dict)
    baseline_decision: Dict[str, Any] = Field(default_factory=dict)


def dump_model(value: Any) -> Dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value or {})


def parse_packet(raw: Dict[str, Any]) -> DecisionPacket:
    return DecisionPacket.model_validate(raw)

