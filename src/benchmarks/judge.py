from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from src.benchmarks.common import QuotaGuard, call_structured


class PersonaGymJudgeSchema(BaseModel):
    persona_consistency: int = Field(..., ge=1, le=5)
    behavior_plausibility: int = Field(..., ge=1, le=5)
    cross_turn_coherence: int = Field(..., ge=1, le=5)
    reasoning: str


class BehaviorJudgeSchema(BaseModel):
    consistency_score: int = Field(..., ge=1, le=5)
    reasoning: str


class CreditReasoningJudgeSchema(BaseModel):
    persona_alignment: int = Field(..., ge=1, le=5)
    emotional_nuance: int = Field(..., ge=1, le=5)
    decision_coherence: int = Field(..., ge=1, le=5)
    reasoning: str


class SurveyReasoningJudgeSchema(BaseModel):
    persona_alignment: int = Field(..., ge=1, le=5)
    reasoning_nuance: int = Field(..., ge=1, le=5)
    decision_coherence: int = Field(..., ge=1, le=5)
    reasoning: str = ""


def _clamp_score(value: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, value))


def _rule_score(text: str, history: Optional[Iterable[str]] = None) -> float:
    body = (text or "").strip()
    if not body:
        return 1.0

    score = 5.0
    lowered = body.lower()
    if "as an ai" in lowered or "как ии" in lowered:
        score -= 2.0
    if len(body) < 35:
        score -= 1.5
    elif len(body) < 70:
        score -= 0.8
    if body.count("\n") > 4:
        score -= 0.3

    hist = [h for h in (history or []) if h]
    if hist:
        overlap = max(sum(token in lowered for token in str(h).lower().split()[:6]) for h in hist)
        if overlap < 2:
            score -= 0.5
    return _clamp_score(score)


def _blend(llm_score: float, rule_score: float, llm_weight: float = 0.7) -> float:
    return _clamp_score(llm_score * llm_weight + rule_score * (1.0 - llm_weight))


def _credit_rule_scores(
    reasoning: str,
    emotional_factors: str,
    voice_stances: Optional[Dict[str, str]] = None,
    conflict_summary: Optional[str] = None,
) -> Dict[str, float]:
    persona_alignment = _rule_score(reasoning)
    emotional_nuance = _rule_score(emotional_factors)
    decision_coherence = _rule_score(reasoning)

    stances = list((voice_stances or {}).values())
    if stances and len(set(stances)) > 1 and not (conflict_summary or "").strip():
        decision_coherence -= 1.0
    if len((emotional_factors or "").strip()) < 20:
        emotional_nuance -= 1.0

    return {
        "persona_alignment": _clamp_score(persona_alignment),
        "emotional_nuance": _clamp_score(emotional_nuance),
        "decision_coherence": _clamp_score(decision_coherence),
    }


def _survey_rule_scores(
    reasoning: str,
    voice_stances: Optional[Dict[str, str]] = None,
    conflict_summary: Optional[str] = None,
    question: str = "",
    question_topic: str = "",
    confidence: float = 0.0,
) -> Dict[str, float]:
    persona_alignment = _rule_score(reasoning)
    reasoning_nuance = _rule_score(reasoning)
    decision_coherence = _rule_score(reasoning)

    stances = list((voice_stances or {}).values())
    if stances:
        if len(set(stances)) > 1 and not (conflict_summary or "").strip():
            reasoning_nuance -= 1.0
            decision_coherence -= 1.0
        if all(stance == "uncertain" for stance in stances):
            reasoning_nuance += 0.2
            if float(confidence or 0.0) > 0.72:
                decision_coherence -= 0.8

    lowered_reasoning = (reasoning or "").lower()
    lowered_question = (question or "").lower()
    if question_topic == "trust_media" and not any(token in lowered_reasoning for token in ["источник", "медиа", "газет", "довер"]):
        persona_alignment -= 0.6
    if question_topic == "advertising_attitudes" and not any(token in lowered_reasoning for token in ["реклам", "навяз", "маркет", "профес"]):
        reasoning_nuance -= 0.6
    if question_topic.startswith("financial") and not any(token in lowered_reasoning for token in ["деньг", "доход", "финанс", "риск", "обеспеч"]):
        persona_alignment -= 0.6
    if "не знаю" in lowered_reasoning or "сложно сказать" in lowered_reasoning:
        reasoning_nuance += 0.2
    if len(lowered_question) > 0 and len(lowered_reasoning) < 80:
        reasoning_nuance -= 0.5

    return {
        "persona_alignment": _clamp_score(persona_alignment),
        "reasoning_nuance": _clamp_score(reasoning_nuance),
        "decision_coherence": _clamp_score(decision_coherence),
    }


async def judge_personagym_response(
    persona_context: str,
    scenario: str,
    response_text: str,
    response_history: List[str],
    locale: str,
    guard: QuotaGuard,
) -> Dict[str, Any]:
    is_ru = (locale or "").lower() == "ru"
    rubric = (
        "Оцени ответ по шкале 1..5:\n"
        "1) persona_consistency — насколько ответ соответствует профилю.\n"
        "2) behavior_plausibility — насколько ответ реалистичен как поведение клиента.\n"
        "3) cross_turn_coherence — согласованность с предыдущими ответами.\n"
        "Верни JSON по схеме."
        if is_ru
        else "Rate the answer on 1..5:\n"
        "1) persona_consistency — fit to persona profile.\n"
        "2) behavior_plausibility — realistic customer behavior.\n"
        "3) cross_turn_coherence — coherence with previous turns.\n"
        "Return JSON only."
    )
    prompt = (
        f"{rubric}\n\n"
        f"Persona profile:\n{persona_context}\n\n"
        f"Scenario:\n{scenario}\n\n"
        f"Response:\n{response_text}\n\n"
        f"Previous responses:\n{response_history[-3:]}"
    )
    llm = await call_structured(prompt, PersonaGymJudgeSchema, guard=guard, temperature=0.0)
    rule = _rule_score(response_text, response_history[-3:])
    combined = {
        "persona_consistency": _blend(float(llm["persona_consistency"]), rule),
        "behavior_plausibility": _blend(float(llm["behavior_plausibility"]), rule),
        "cross_turn_coherence": _blend(float(llm["cross_turn_coherence"]), rule),
    }
    return {
        "llm_scores": llm,
        "rule_score": float(rule),
        "combined_scores": combined,
        "reasoning": llm.get("reasoning", ""),
    }


async def judge_behavior_consistency(
    persona_context: str,
    scenario: str,
    response_text: str,
    locale: str,
    guard: QuotaGuard,
) -> Dict[str, Any]:
    is_ru = (locale or "").lower() == "ru"
    rubric = (
        "Оцени согласованность ответа с профилем персоны по шкале 1..5. "
        "1 — сильное противоречие, 5 — полное соответствие. Верни JSON."
        if is_ru
        else "Rate consistency of this response with persona profile on 1..5. "
        "1 means contradiction, 5 means strong alignment. Return JSON."
    )
    prompt = (
        f"{rubric}\n\n"
        f"Persona profile:\n{persona_context}\n\n"
        f"Scenario:\n{scenario}\n\n"
        f"Response:\n{response_text}"
    )
    llm = await call_structured(prompt, BehaviorJudgeSchema, guard=guard, temperature=0.0)
    rule = _rule_score(response_text)
    combined = _blend(float(llm["consistency_score"]), rule)
    return {
        "llm_score": float(llm["consistency_score"]),
        "rule_score": float(rule),
        "combined_score": float(combined),
        "reasoning": llm.get("reasoning", ""),
    }


async def judge_credit_reasoning(
    packet: Dict[str, Any],
    decision: Dict[str, Any],
    locale: str,
    guard: QuotaGuard,
) -> Dict[str, Any]:
    is_ru = (locale or "").lower() == "ru"
    rubric = (
        "Оцени итоговое кредитное решение по шкале 1..5.\n"
        "1) persona_alignment — насколько решение согласовано с профилем и целью клиента.\n"
        "2) emotional_nuance — насколько решение отражает тонкие эмоции, а не только общий вывод.\n"
        "3) decision_coherence — насколько итог не противоречит входным данным и внутренним голосам.\n"
        "Верни JSON по схеме."
        if is_ru
        else "Rate the credit decision on 1..5.\n"
        "1) persona_alignment — fit to persona profile and goal.\n"
        "2) emotional_nuance — emotional subtlety of the explanation.\n"
        "3) decision_coherence — consistency with inputs and internal voices.\n"
        "Return JSON only."
    )
    prompt = (
        f"{rubric}\n\n"
        f"Packet profile:\n{packet.get('profile', {})}\n\n"
        f"Goal:\n{packet.get('goal', {})}\n\n"
        f"Recent session history:\n{packet.get('session_history', [])[-2:]}\n\n"
        f"Final emotional state:\n{packet.get('final_emotional_state', {})}\n\n"
        f"Push info:\n{packet.get('push_info', {})}\n\n"
        f"Reaction:\n{packet.get('reaction', {})}\n\n"
        f"Decision output:\n{decision}"
    )
    llm = await call_structured(prompt, CreditReasoningJudgeSchema, guard=guard, temperature=0.0)
    rule_scores = _credit_rule_scores(
        reasoning=str(decision.get("reasoning", "")),
        emotional_factors=str(decision.get("emotional_factors", "")),
        voice_stances=decision.get("voice_stances"),
        conflict_summary=decision.get("conflict_summary"),
    )
    combined = {
        "persona_alignment": _blend(float(llm["persona_alignment"]), rule_scores["persona_alignment"]),
        "emotional_nuance": _blend(float(llm["emotional_nuance"]), rule_scores["emotional_nuance"]),
        "decision_coherence": _blend(float(llm["decision_coherence"]), rule_scores["decision_coherence"]),
    }
    return {
        "llm_scores": llm,
        "rule_scores": rule_scores,
        "combined_scores": combined,
        "reasoning": llm.get("reasoning", ""),
    }


async def judge_survey_reasoning(
    profile: Dict[str, Any],
    question: str,
    answer: Dict[str, Any],
    locale: str,
    guard: QuotaGuard,
) -> Dict[str, Any]:
    is_ru = (locale or "").lower() == "ru"
    rubric = (
        "Оцени survey-ответ по шкале 1..5.\n"
        "1) persona_alignment — насколько ответ согласован с профилем персоны.\n"
        "2) reasoning_nuance — насколько reasoning отражает тонкость позиции и внутренние оговорки, а не только грубый yes/no.\n"
        "3) decision_coherence — насколько итоговый ответ не противоречит вопросу, reasoning и внутренним голосам.\n"
        "Верни JSON по схеме."
        if is_ru
        else "Rate the survey answer on 1..5.\n"
        "1) persona_alignment — fit to persona profile.\n"
        "2) reasoning_nuance — nuance and subtlety of the answer.\n"
        "3) decision_coherence — consistency with question, reasoning and internal voices.\n"
        "Return JSON only."
    )
    prompt = (
        f"{rubric}\n\n"
        f"Persona profile:\n{profile}\n\n"
        f"Survey question:\n{question}\n\n"
        f"Survey answer:\n{answer}\n"
    )
    llm = await call_structured(prompt, SurveyReasoningJudgeSchema, guard=guard, temperature=0.0)
    rule_scores = _survey_rule_scores(
        reasoning=str(answer.get("reasoning", "")),
        voice_stances=answer.get("voice_stances"),
        conflict_summary=answer.get("conflict_summary"),
        question=question,
        question_topic=str(answer.get("question_topic") or ""),
        confidence=float(answer.get("confidence") or 0.0),
    )
    combined = {
        "persona_alignment": _blend(float(llm["persona_alignment"]), rule_scores["persona_alignment"]),
        "reasoning_nuance": _blend(float(llm["reasoning_nuance"]), rule_scores["reasoning_nuance"]),
        "decision_coherence": _blend(float(llm["decision_coherence"]), rule_scores["decision_coherence"]),
    }
    return {
        "llm_scores": llm,
        "rule_scores": rule_scores,
        "combined_scores": combined,
        "reasoning": llm.get("reasoning", ""),
    }
