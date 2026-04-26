from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.schemas.decision_schema import DecisionOutcome
from src.utils import robust_llm_call


def _decision_from_data(data: Dict[str, Any]) -> DecisionOutcome:
    if hasattr(DecisionOutcome, "model_validate"):
        return DecisionOutcome.model_validate(data)
    return DecisionOutcome(**data)


class DecisionAgent:
    """
    Агент принимает решение от лица клиента после push уведомления.
    Он учитывает эмоции, поведение и стремится к улучшению благосостояния клиента.
    """

    def build_prompt(
        self,
        profile: Dict[str, Any],
        persona_history: List[Dict[str, Any]],
        emotional_state: Dict[str, Any],
        push_info: Dict[str, Any],
    ) -> str:
        return f"""
        Клиент с профилем {profile} получил push-уведомление: {push_info}.
        Его история действий: {persona_history}.
        Эмоциональное состояние: {emotional_state}.

        От лица клиента прими решение: брать ли кредит.
        Учитывай личные цели, эмоции и стремление к максимизации качества жизни и финансового благополучия.
        Верни итоговое решение, объяснение и эмоциональные факторы.
        """

    async def make_final_decision(
        self,
        profile: Dict[str, Any],
        persona_history: List[Dict[str, Any]],
        emotional_state: Dict[str, Any],
        push_info: Dict[str, Any],
        goal: Optional[Dict[str, Any]] = None,
        reaction: Optional[Dict[str, Any]] = None,
    ) -> DecisionOutcome:
        decision, _ = await self.make_final_decision_with_trace(
            profile=profile,
            persona_history=persona_history,
            emotional_state=emotional_state,
            push_info=push_info,
            goal=goal,
            reaction=reaction,
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
    ) -> Tuple[DecisionOutcome, Dict[str, Any]]:
        prompt = self.build_prompt(profile, persona_history, emotional_state, push_info)
        raw_decision = await robust_llm_call(prompt, structured_output=DecisionOutcome)
        if hasattr(raw_decision, "model_dump"):
            data = raw_decision.model_dump()
        elif hasattr(raw_decision, "dict"):
            data = raw_decision.dict()
        else:
            data = dict(raw_decision)
        data["decision_mode"] = "direct"
        data["voice_stances"] = None
        data["conflict_summary"] = None
        decision = _decision_from_data(data)
        trace = {
            "decision_mode": "direct",
            "prompt_chars": len(prompt),
            "prompt_char_counts": {"direct": len(prompt)},
            "llm_calls": 1,
        }
        return decision, trace
