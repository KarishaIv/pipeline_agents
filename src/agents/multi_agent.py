import asyncio
from src.agents.persona_agent import PersonaAgent
from src.agents.financial_agent import FinancialAgent
from src.agents.decision_agent import DecisionAgent

class MultiAgentSystem:
    """Оркестратор симуляции клиента, банка и эмоций"""

    def __init__(self, profile: dict, steps: int = 3):
        self.profile = profile
        self.steps = steps
        self.persona = PersonaAgent(profile)
        self.financial_agent = FinancialAgent()
        self.decision_agent = DecisionAgent()

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
        decision = await self.decision_agent.make_final_decision(
            self.profile,
            [h.dict() for h in self.persona.history],
            self.persona.state.dict(),
            push_info
        )

        return {
            "profile": self.profile,
            "goal": self.persona.goal.dict(),
            "session_history": [h.dict() for h in self.persona.history],
            "final_emotional_state": self.persona.state.dict(),
            "push_info": push_info,
            "reaction": reaction.dict(),
            "decision": decision.dict(),
        }
