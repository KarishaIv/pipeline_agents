from src.utils import robust_llm_call
from src.schemas.decision_schema import DecisionOutcome

class DecisionAgent:
    """
    Агент принимает решение от лица клиента после push уведомления.
    Он учитывает эмоции, поведение и стремится к улучшению благосостояния клиента.
    """

    async def make_final_decision(self, profile: dict, persona_history: list, emotional_state: dict, push_info: dict) -> DecisionOutcome:
        prompt = f"""
        Клиент с профилем {profile} получил push-уведомление: {push_info}.
        Его история действий: {persona_history}.
        Эмоциональное состояние: {emotional_state}.

        От лица клиента прими решение: брать ли кредит.
        Учитывай личные цели, эмоции и стремление к максимизации качества жизни и финансового благополучия.
        Верни итоговое решение, объяснение и эмоциональные факторы.
        """
        decision = await robust_llm_call(prompt, structured_output=DecisionOutcome)
        return decision
