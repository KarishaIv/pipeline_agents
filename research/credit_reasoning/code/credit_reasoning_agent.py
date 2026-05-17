from __future__ import annotations

import asyncio
import json
import time
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from credit_news_adapter import CreditNewsAdapter
from credit_schemas import (
    CreditNewsSignals,
    DecisionOutcome,
    DecisionPacket,
    SynthesizedCreditDecision,
    VoiceOpinion,
    dump_model,
    parse_packet,
)


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


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _mean(values: List[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _round(value: float) -> float:
    return round(_clamp(value), 3)


def _parse_llm_json(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


class CreditReasoningAgent:
    """Restored credit final-decision layer.

    `compact_debate` mirrors the preserved artifacts: an emotional voice and a
    rational voice are normalized into numeric factors; Python aggregation then
    applies conservative guardrails and only the final explanation can be LLM-written.
    """

    def __init__(
        self,
        decision_mode: str = "compact_debate",
        narrative_mode: str = "heuristic",
        model: Any = None,
        news_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if decision_mode not in {"direct", "compact_debate"}:
            raise ValueError("decision_mode must be 'direct' or 'compact_debate'")
        if narrative_mode not in {"heuristic", "llm"}:
            raise ValueError("narrative_mode must be 'heuristic' or 'llm'")
        self.decision_mode = decision_mode
        self.narrative_mode = narrative_mode
        self.model = model
        self.news_summary, self.news_signals = CreditNewsAdapter().adapt(news_context)

    async def decide(self, raw_packet: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        packet = parse_packet(raw_packet)
        if self.decision_mode == "direct":
            result = await self._direct_decision(packet)
        else:
            result = await self._compact_debate(packet)
        result["latency_seconds"] = round(time.perf_counter() - started, 4)
        return result

    async def _direct_decision(self, packet: DecisionPacket) -> Dict[str, Any]:
        prompt = self._direct_prompt(packet)
        llm_calls = 0
        if self.narrative_mode == "llm":
            llm_calls = 1
            try:
                from src.utils import robust_llm_call

                raw = await robust_llm_call(prompt, model=self.model, temperature=0.2, structured_output=DecisionOutcome)
                outcome = raw if isinstance(raw, DecisionOutcome) else DecisionOutcome.model_validate(_parse_llm_json(raw))
            except Exception:
                outcome = self._heuristic_direct(packet)
        else:
            outcome = self._heuristic_direct(packet)

        data = dump_model(outcome)
        data.update(
            {
                "decision_mode": "direct",
                "voice_stances": {},
                "conflict_summary": "",
                "prompt_chars": len(prompt),
                "prompt_char_counts": {"direct": len(prompt)},
                "llm_calls": llm_calls,
                "goal_intent": self._infer_goal_intent(packet),
                "narrative_mode": self.narrative_mode,
                "context_summary": self._context_summary(packet),
                "trace_voices": {},
                "baseline_decision": packet.baseline_decision,
            }
        )
        return data

    async def _compact_debate(self, packet: DecisionPacket) -> Dict[str, Any]:
        goal_intent = self._infer_goal_intent(packet)
        prompts = {
            "emotional": self._voice_prompt(packet, "emotional", goal_intent),
            "rational": self._voice_prompt(packet, "rational", goal_intent),
        }
        llm_calls = 0

        if self.narrative_mode == "llm":
            llm_calls += 2
            emotional_task = self._llm_voice(packet, "emotional", prompts["emotional"])
            rational_task = self._llm_voice(packet, "rational", prompts["rational"])
            emotional, rational = await asyncio.gather(emotional_task, rational_task)
        else:
            emotional = self._heuristic_voice(packet, "emotional", goal_intent)
            rational = self._heuristic_voice(packet, "rational", goal_intent)

        voices = {"emotional": emotional, "rational": rational}
        outcome, synthesis_seed = self._aggregate(packet, voices, goal_intent)
        synthesis_prompt = self._synthesis_prompt(packet, voices, outcome, synthesis_seed)

        if self.narrative_mode == "llm":
            llm_calls += 1
            try:
                from src.utils import robust_llm_call

                raw = await robust_llm_call(
                    synthesis_prompt,
                    model=self.model,
                    temperature=0.2,
                    structured_output=SynthesizedCreditDecision,
                )
                synthesis = raw if isinstance(raw, SynthesizedCreditDecision) else SynthesizedCreditDecision.model_validate(_parse_llm_json(raw))
                outcome.reasoning = synthesis.reasoning or outcome.reasoning
                outcome.emotional_factors = synthesis.emotional_factors or outcome.emotional_factors
                outcome.conflict_summary = synthesis.conflict_summary or outcome.conflict_summary
            except Exception:
                pass

        data = dump_model(outcome)
        data.update(
            {
                "decision_mode": "compact_debate",
                "voice_stances": {name: voice.stance for name, voice in voices.items()},
                "conflict_summary": outcome.conflict_summary,
                "prompt_chars": sum(len(value) for value in [*prompts.values(), synthesis_prompt]),
                "prompt_char_counts": {
                    "emotional": len(prompts["emotional"]),
                    "rational": len(prompts["rational"]),
                    "synthesis": len(synthesis_prompt),
                },
                "llm_calls": llm_calls,
                "goal_intent": goal_intent,
                "narrative_mode": self.narrative_mode,
                "context_summary": self._context_summary(packet),
                "trace_voices": {name: dump_model(voice) for name, voice in voices.items()},
                "baseline_decision": packet.baseline_decision,
            }
        )
        if self.news_summary:
            data["news_snapshot_id"] = self.news_summary.get("snapshot_id")
            data["news_context_summary"] = self.news_summary
            data["news_signal_summary"] = dump_model(self.news_signals)
        return data

    async def _llm_voice(self, packet: DecisionPacket, voice_name: str, prompt: str) -> VoiceOpinion:
        try:
            from src.utils import robust_llm_call

            raw = await robust_llm_call(prompt, model=self.model, temperature=0.2, structured_output=VoiceOpinion)
            if isinstance(raw, VoiceOpinion):
                return raw
            return VoiceOpinion.model_validate(_parse_llm_json(raw))
        except Exception:
            return self._heuristic_voice(packet, voice_name, self._infer_goal_intent(packet))

    def _heuristic_direct(self, packet: DecisionPacket) -> DecisionOutcome:
        emotional_state = packet.final_emotional_state
        stress = _clamp(emotional_state.get("stress"), 0.5)
        confidence = _clamp(emotional_state.get("confidence"), 0.5)
        trust = _clamp(emotional_state.get("trust_in_bank"), 0.5)
        urgency = _clamp(emotional_state.get("urgency"), 0.4)
        intent = self._infer_goal_intent(packet)
        application_bonus = {"application": 0.18, "exploratory": 0.05, "informational": -0.04}.get(intent, 0.0)
        news_penalty = self.news_signals.credit_headwind * 0.08 if self.news_signals else 0.0
        news_bonus = self.news_signals.opportunity_support * 0.05 if self.news_signals else 0.0
        probability = _round(0.28 + 0.20 * confidence + 0.18 * trust + 0.16 * urgency - 0.12 * stress + application_bonus - news_penalty + news_bonus)
        threshold = {"application": 0.55, "exploratory": 0.60, "informational": 0.64}.get(intent, 0.60)
        will_take = probability >= threshold
        return DecisionOutcome(
            will_take_credit=will_take,
            probability_score=probability,
            reasoning=(
                "Прямой слой оценивает готовность к кредиту как достаточную."
                if will_take
                else "Прямой слой не видит достаточной текущей готовности к оформлению кредита."
            ),
            emotional_factors=f"confidence={confidence:.2f}, trust={trust:.2f}, urgency={urgency:.2f}, stress={stress:.2f}",
            decision_mode="direct",
            score_breakdown={
                "goal_intent": intent,
                "decision_threshold": threshold,
                "probability_score": probability,
            },
            news_context_used=self.news_signals is not None,
        )

    def _heuristic_voice(self, packet: DecisionPacket, voice_name: str, goal_intent: str) -> VoiceOpinion:
        profile = packet.profile
        emotional_state = packet.final_emotional_state
        goal_text = _lower(packet.goal.get("goal_description"))
        motivation = _lower(packet.goal.get("motivation"))
        reaction_type = _lower(packet.reaction.get("reaction_type"))
        push_prediction = packet.push_info.get("prediction") if isinstance(packet.push_info.get("prediction"), dict) else {}

        is_credit_goal = any(token in f"{goal_text} {motivation}" for token in ["кредит", "ипотек", "заем", "займ"])
        is_application = goal_intent == "application"
        is_info = goal_intent == "informational"
        interested = any(token in reaction_type for token in ["заинтерес", "интерес", "позитив"])

        confidence = _clamp(emotional_state.get("confidence"), 0.55)
        trust = _clamp(emotional_state.get("trust_in_bank"), 0.55)
        urgency = _clamp(emotional_state.get("urgency"), 0.45)
        stress = _clamp(emotional_state.get("stress"), 0.45)
        predicted = _clamp(push_prediction.get("probability_take_credit"), 0.5)

        news_headwind = self.news_signals.credit_headwind if self.news_signals else 0.0
        news_opportunity = self.news_signals.opportunity_support if self.news_signals else 0.0
        news_stress = self.news_signals.stress_bias if self.news_signals else 0.0
        news_trust = self.news_signals.trust_drag if self.news_signals else 0.0

        if voice_name == "emotional":
            desire = 0.38 + 0.22 * predicted + 0.14 * confidence + 0.10 * trust + 0.10 * interested + 0.08 * news_opportunity
            need = 0.25 + 0.18 * is_credit_goal + 0.16 * urgency + 0.12 * is_application + 0.06 * interested
            readiness = 0.28 + 0.20 * confidence + 0.14 * trust + 0.12 * is_application + 0.08 * interested - 0.08 * is_info
            risk = 0.30 + 0.22 * stress + 0.12 * news_stress + 0.10 * news_headwind
            theses = [
                "эмоционально откликается на кредитное предложение",
                "интерес усиливается за счет доверия и уверенности",
                "текущая готовность зависит от срочности цели",
            ]
        else:
            income_text = _lower(profile.get("income_level"))
            stable_income = any(token in income_text for token in ["выше", "средн", "45000", "123500"])
            desire = 0.30 + 0.16 * predicted + 0.10 * is_credit_goal + 0.06 * news_opportunity
            need = 0.18 + 0.20 * is_application + 0.12 * urgency + 0.08 * is_credit_goal - 0.06 * is_info
            readiness = 0.22 + 0.18 * stable_income + 0.16 * is_application + 0.10 * confidence - 0.08 * is_info
            risk = 0.38 + 0.20 * stress + 0.16 * news_headwind + 0.12 * news_trust - 0.08 * stable_income
            theses = [
                "рационально проверяет необходимость кредита",
                "учитывает доход, риск долговой нагрузки и срочность",
                "не считает информационный интерес заявкой на кредит",
            ]

        desire = _round(desire)
        need = _round(need)
        readiness = _round(readiness)
        risk = _round(risk)
        confidence_voice = _round(0.56 + 0.18 * confidence + 0.10 * trust - 0.10 * risk)
        if readiness >= 0.68 and need >= 0.55 and risk < 0.70:
            stance = "take_credit"
            readiness_level = "ready"
            blocking = None
        elif desire >= 0.55 and readiness >= 0.40 and risk < 0.75:
            stance = "uncertain" if voice_name == "rational" and is_info else "take_credit"
            readiness_level = "considering"
            blocking = "нужно уточнить реальную необходимость и условия кредита"
        else:
            stance = "reject_credit"
            readiness_level = "browsing"
            blocking = "нет достаточной текущей потребности в кредите"

        return VoiceOpinion(
            stance=stance,
            desire_for_credit=desire,
            need_for_credit=need,
            readiness_now=readiness,
            readiness_level=readiness_level,
            risk_pressure=risk,
            confidence=confidence_voice,
            theses=theses,
            key_factors=[
                f"intent={goal_intent}",
                f"confidence={confidence:.2f}",
                f"trust={trust:.2f}",
                f"news_headwind={news_headwind:.2f}",
            ],
            blocking_factor=blocking,
        )

    def _aggregate(
        self,
        packet: DecisionPacket,
        voices: Dict[str, VoiceOpinion],
        goal_intent: str,
    ) -> Tuple[DecisionOutcome, Dict[str, Any]]:
        voice_list = list(voices.values())
        emotional = voices["emotional"]
        rational = voices["rational"]

        avg_desire = _mean([voice.desire_for_credit for voice in voice_list])
        avg_need = _mean([voice.need_for_credit for voice in voice_list])
        avg_readiness = _mean([voice.readiness_now for voice in voice_list])
        max_readiness = max(voice.readiness_now for voice in voice_list)
        avg_risk = _mean([voice.risk_pressure for voice in voice_list])
        avg_confidence = _mean([voice.confidence for voice in voice_list])

        emotional_pull = _clamp(
            0.45 * emotional.desire_for_credit
            + 0.25 * emotional.need_for_credit
            + 0.20 * emotional.readiness_now
            + 0.10 * (1.0 - emotional.risk_pressure)
        )
        rational_pull = _clamp(
            0.25 * rational.desire_for_credit
            + 0.45 * rational.need_for_credit
            + 0.25 * rational.readiness_now
            + 0.05 * (1.0 - rational.risk_pressure)
        )

        news_headwind = self.news_signals.credit_headwind if self.news_signals else 0.0
        news_opportunity = self.news_signals.opportunity_support if self.news_signals else 0.0
        news_trust = self.news_signals.trust_drag if self.news_signals else 0.0

        motivation_score = _clamp(0.34 * avg_desire + 0.33 * avg_need + 0.33 * max_readiness)
        caution_score = _clamp(0.52 * avg_risk + 0.28 * (1.0 - avg_readiness) + 0.10 * news_headwind + 0.10 * news_trust)
        probability = _clamp(
            0.54 * motivation_score
            + 0.24 * emotional_pull
            + 0.22 * rational_pull
            - 0.25 * caution_score
            + 0.05 * news_opportunity
            - 0.08 * news_headwind
        )

        threshold = {"application": 0.52, "exploratory": 0.57, "informational": 0.61}.get(goal_intent, 0.59)
        will_take = probability >= threshold and avg_need >= 0.45 and max_readiness >= 0.55 and avg_risk < 0.75
        guardrails: List[str] = []
        if goal_intent == "informational" and avg_need < 0.55:
            guardrails.append("informational_low_need")
        if rational.stance == "reject_credit" and rational.need_for_credit < 0.45:
            guardrails.append("rational_rejects_credit")
        if max_readiness < 0.50:
            guardrails.append("no_readiness_now")
        if avg_risk >= 0.75:
            guardrails.append("excessive_risk_pressure")
        if self.news_signals and self.news_signals.overall_reaction == "negative" and goal_intent != "application":
            guardrails.append("negative_news_non_application")

        if will_take and guardrails:
            will_take = False
            probability = min(probability, 0.49)

        score_breakdown: Dict[str, Any] = {
            "avg_confidence": round(avg_confidence, 3),
            "avg_desire": round(avg_desire, 3),
            "avg_need": round(avg_need, 3),
            "avg_readiness": round(avg_readiness, 3),
            "avg_risk": round(avg_risk, 3),
            "caution_score": round(caution_score, 3),
            "decision_threshold": threshold,
            "emotional_pull": round(emotional_pull, 3),
            "goal_intent": goal_intent,
            "max_readiness": round(max_readiness, 3),
            "motivation_score": round(motivation_score, 3),
            "probability_score": round(probability, 3),
            "rational_pull": round(rational_pull, 3),
        }
        if self.news_signals:
            score_breakdown.update(
                {
                    "news_credit_headwind": self.news_signals.credit_headwind,
                    "news_credit_relevance": self.news_signals.credit_relevance,
                    "news_opportunity_support": self.news_signals.opportunity_support,
                    "news_stress_bias": self.news_signals.stress_bias,
                    "news_trust_drag": self.news_signals.trust_drag,
                }
            )

        conflict = self._conflict_summary(voices)
        synthesis_seed = {
            "will_take_credit": will_take,
            "probability_score": round(probability, 3),
            "conflict_summary": conflict,
            "score_breakdown": score_breakdown,
            "guardrails_applied": guardrails,
        }

        outcome = DecisionOutcome(
            will_take_credit=will_take,
            probability_score=round(probability, 3),
            reasoning=self._heuristic_reasoning(packet, voices, synthesis_seed),
            emotional_factors=self._heuristic_emotional_factors(packet, voices),
            decision_mode="compact_debate",
            voice_stances={name: voice.stance for name, voice in voices.items()},
            conflict_summary=conflict,
            score_breakdown=score_breakdown,
            guardrails_applied=guardrails,
            news_context_used=self.news_signals is not None,
        )
        return outcome, synthesis_seed

    def _infer_goal_intent(self, packet: DecisionPacket) -> str:
        text = _lower(f"{packet.goal.get('goal_description')} {packet.goal.get('motivation')}")
        history_text = _lower(" ".join(_text(row.get("action")) for row in packet.session_history))
        combined = f"{text} {history_text}"
        if any(token in combined for token in ["подать заявку", "оформить", "получить кредит", "взять кредит", "заявку на кредит"]):
            return "application"
        if any(token in combined for token in ["рассчитать", "подобрать", "сравнить", "условия", "программы", "лояльности"]):
            return "informational"
        if any(token in combined for token in ["кредит", "ипотек", "заем", "займ"]):
            return "exploratory"
        return "informational"

    def _context_summary(self, packet: DecisionPacket) -> Dict[str, Any]:
        return {
            "goal_intent": self._infer_goal_intent(packet),
            "reaction_type": packet.reaction.get("reaction_type"),
        }

    def _conflict_summary(self, voices: Dict[str, VoiceOpinion]) -> str:
        emotional = voices["emotional"]
        rational = voices["rational"]
        if emotional.stance == rational.stance:
            return f"Оба голоса сходятся в позиции `{emotional.stance}`."
        return (
            "Эмоциональный голос сильнее реагирует на интерес и привлекательность предложения, "
            "а рациональный голос проверяет необходимость, готовность и долговой риск."
        )

    def _heuristic_reasoning(self, packet: DecisionPacket, voices: Dict[str, VoiceOpinion], seed: Dict[str, Any]) -> str:
        profile = packet.profile
        audience = _text(profile.get("target_audience_name")) or "unknown audience"
        goal = _text(packet.goal.get("goal_description"))
        decision = "готовность к кредиту достаточна" if seed["will_take_credit"] else "готовность к кредиту недостаточна"
        score = seed["score_breakdown"]
        news_part = ""
        if self.news_signals:
            news_part = (
                f" News context учитывался как внешний фон: headwind={self.news_signals.credit_headwind:.2f}, "
                f"opportunity={self.news_signals.opportunity_support:.2f}."
            )
        return (
            f"Для аудитории `{audience}` и цели `{goal}` {decision}. "
            f"Средняя потребность={score['avg_need']:.2f}, готовность={score['avg_readiness']:.2f}, "
            f"риск={score['avg_risk']:.2f}, итоговая вероятность={seed['probability_score']:.2f}. "
            f"{seed['conflict_summary']}{news_part}"
        )

    def _heuristic_emotional_factors(self, packet: DecisionPacket, voices: Dict[str, VoiceOpinion]) -> str:
        state = packet.final_emotional_state
        return (
            f"mood={_clamp(state.get('mood'), 0.5):.2f}, stress={_clamp(state.get('stress'), 0.5):.2f}, "
            f"confidence={_clamp(state.get('confidence'), 0.5):.2f}, trust={_clamp(state.get('trust_in_bank'), 0.5):.2f}; "
            f"emotional stance={voices['emotional'].stance}, rational stance={voices['rational'].stance}"
        )

    def _direct_prompt(self, packet: DecisionPacket) -> str:
        return (
            "Ты финальный decision agent кредитного сценария. Верни JSON DecisionOutcome.\n"
            f"PACKET:\n{_json(packet.model_dump())}\n"
            f"NEWS:\n{_json(self.news_summary) if self.news_summary else 'null'}\n"
        )

    def _voice_prompt(self, packet: DecisionPacket, voice_name: str, goal_intent: str) -> str:
        role = (
            "эмоциональный голос: оцени желание, доверие, импульс и субъективную привлекательность"
            if voice_name == "emotional"
            else "рациональный голос: оцени необходимость, платежеспособность, риск и готовность сейчас"
        )
        return (
            f"Ты {role}. Верни JSON VoiceOpinion без лишнего текста.\n"
            f"GOAL_INTENT: {goal_intent}\n"
            f"PACKET:\n{_json(packet.model_dump())}\n"
            f"NEWS_SIGNALS:\n{_json(dump_model(self.news_signals)) if self.news_signals else 'null'}\n"
        )

    def _synthesis_prompt(
        self,
        packet: DecisionPacket,
        voices: Dict[str, VoiceOpinion],
        outcome: DecisionOutcome,
        synthesis_seed: Dict[str, Any],
    ) -> str:
        return (
            "Синтезируй короткое финальное объяснение кредитного решения. "
            "Не меняй binary decision и probability_score. Верни JSON SynthesizedCreditDecision.\n"
            f"PACKET_ID: {packet.packet_id}\n"
            f"VOICES: {_json({name: dump_model(voice) for name, voice in voices.items()})}\n"
            f"OUTCOME: {_json(dump_model(outcome))}\n"
            f"SCORES: {_json(synthesis_seed)}\n"
        )

