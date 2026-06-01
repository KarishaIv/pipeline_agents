import asyncio
from src.agents.credit_reasoning_agent import CreditReasoningAgent
from src.agents.persona_agent import PersonaAgent
from src.agents.financial_agent import FinancialAgent
from src.agents.decision_agent import DecisionAgent

class MultiAgentSystem:
    """Оркестратор симуляции клиента, банка и эмоций"""

    def __init__(
        self,
        profile: dict,
        steps: int = 3,
        decision_mode: str = "direct",
        news_context: dict | None = None,
    ):
        self.profile = profile
        self.steps = steps
        self.decision_mode = decision_mode
        self.news_context = news_context
        self.persona = PersonaAgent(profile)
        self.financial_agent = FinancialAgent()
        self.decision_agent = DecisionAgent()
        self.credit_reasoning_agent = CreditReasoningAgent() if decision_mode == "compact_debate" else None
        if decision_mode not in {"direct", "compact_debate"}:
            raise ValueError(f"Unsupported decision_mode: {decision_mode}")

    async def run_simulation(self) -> dict:
        # 1. Инициализация
        await self.persona.initialize()

        # 2. Симуляция шагов в приложении
        for i in range(1, self.steps + 1):
            await self.persona.act_step(i)

        # 3. Финансовый агент создаёт push
        push_info = await self.financial_agent.generate_push(self.profile, [h.dict() for h in self.persona.history])

        # 4. Персона реагирует на push
        reaction = await self.persona.react_to_push(push_info["push"]["message"])

        # 5. Агент решения принимает итоговое решение
        persona_history = [h.dict() for h in self.persona.history]
        goal = self.persona.goal.dict() if self.persona.goal is not None else None
        reaction_data = reaction.dict()
        if self.decision_mode == "compact_debate":
            decision = await self.credit_reasoning_agent.make_final_decision(
                self.profile,
                persona_history,
                self.persona.state.dict(),
                push_info,
                goal=goal,
                reaction=reaction_data,
                news_context=self.news_context,
            )
        else:
            decision = await self.decision_agent.make_final_decision(
                self.profile,
                persona_history,
                self.persona.state.dict(),
                push_info,
                goal=goal,
                reaction=reaction_data,
            )

        return {
            "profile": self.profile,
            "goal": goal,
            "session_history": persona_history,
            "final_emotional_state": self.persona.state.dict(),
            "push_info": push_info,
            "reaction": reaction_data,
            "decision_mode": self.decision_mode,
            "news_context": {
                "snapshot_id": self.news_context.get("snapshot_id"),
                "generated_at": self.news_context.get("generated_at"),
                "audience": self.news_context.get("audience"),
                "question": self.news_context.get("question"),
                "overall_reaction": self.news_context.get("overall_reaction"),
                "impact_horizon": self.news_context.get("impact_horizon"),
                "confidence": self.news_context.get("confidence"),
            } if isinstance(self.news_context, dict) else None,
            "decision": decision.dict(),
        }
