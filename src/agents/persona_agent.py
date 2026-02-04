import asyncio
from typing import Dict, List
from src.agents.emotion_agent import EmotionAgent
from src.agents.tool_agent import ToolAgent
from src.utils import robust_llm_call
from src.schemas.persona_schema import (
    PersonaGoal,
    PersonaAction,
    PersonaReaction,
    PersonaSessionRecord,
)
from src.schemas.tool_schema import ToolResponseSchema


class PersonaAgent:
    """Персона-клиент, которая взаимодействует с банковским приложением"""

    def __init__(self, profile: Dict):
        self.profile = profile
        self.emotion_agent = EmotionAgent()
        self.state = None
        self.goal: PersonaGoal | None = None
        self.history: List[PersonaSessionRecord] = []
        self.tool_agent = ToolAgent()

    async def initialize(self):
        """Инициализация цели и эмоций"""
        self.goal = await self.determine_session_goal()
        self.state = await self.emotion_agent.initialize_state(self.profile, self.goal.goal_description)

    async def determine_session_goal(self) -> PersonaGoal:
        prompt = f"""
        Клиент заходит в банковское приложение с определённой целью.
        Профиль клиента: {self.profile}.
        Определи, что он хочет сделать (например, оплатить счета, проверить баланс, оформить кредит, узнать о программе лояльности).
        Объясни мотивацию.
        """
        result = await robust_llm_call(prompt, structured_output=PersonaGoal)
        return result

    async def act_step(self, step_num: int) -> PersonaSessionRecord:
        """Шаг симуляции: клиент делает действие, получает ответ, эмоции обновляются"""
        recent_actions = [h.action for h in self.history[-3:]] if self.history else []

        prompt = f"""
        Клиент находится в приложении банка. 
        Его цель: {self.goal.goal_description}.
        Текущее эмоциональное состояние: {self.state.dict()}.
        Последние действия: {recent_actions}.

        Определи, что он сделает дальше (нажмет, выберет, введет, отреагирует и т.д.).
        Объясни, почему он так поступает. Его действие включает только одну операцию за этот шаг.
        """
        persona_action = await robust_llm_call(prompt, structured_output=PersonaAction)
        tool_result_dict = await self.tool_agent.respond(persona_action.next_action, self.profile, [h.dict() for h in self.history])
        tool_result = ToolResponseSchema(**tool_result_dict)

        self.state = await self.emotion_agent.update_state(self.state.dict(), persona_action.next_action, tool_result_dict)

        record = PersonaSessionRecord(
            step=step_num,
            action=persona_action.next_action,
            tool_result=tool_result,
            emotional_state=self.state.dict()
        )
        self.history.append(record)
        return record

    async def react_to_push(self, push: str) -> PersonaReaction:
        """Реакция на push от банка"""
        prompt = f"""
        Клиент получает push-уведомление от банка:
        "{push}"

        Эмоциональное состояние перед уведомлением: {self.state.dict()}.
        Определи, как он реагирует: откроет, проигнорирует, заинтересуется и т.д.
        Объясни эмоции и мотивы реакции.
        """
        reaction = await robust_llm_call(prompt, structured_output=PersonaReaction)
        self.state = await self.emotion_agent.update_state(self.state.dict(), "push", {"message": push})
        return reaction
