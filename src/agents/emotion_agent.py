from src.utils import robust_llm_call
from src.schemas.emotional_schema import EmotionalStateSchema

class EmotionAgent:
    """
    Агент, который моделирует эмоциональное состояние клиента:
    - определяет начальное состояние
    - обновляет эмоции после каждого шага
    """

    async def initialize_state(self, profile: dict, goal: str) -> EmotionalStateSchema:
        prompt = f"""
        Определи начальное эмоциональное состояние клиента по профилю:
        {profile}

        и его задаче: 
        {goal}

        Укажи настроение, стресс, уверенность, доверие к банку и срочность в диапазоне [0-1].
        Представь, что ты психолог, оценивающий эмоциональное состояние клиента перед использованием приложения банка. 
        Предложенные значения должны быть лишь одним из возможных вариантов, учитывая разные кейсы эмоционального состояния клиентов в разные моменты времени.
        """
        result = await robust_llm_call(prompt, structured_output=EmotionalStateSchema)
        return result

    async def update_state(self, current_state: dict, action: str, tool_response: dict) -> EmotionalStateSchema:
        prompt = f"""
        Клиент совершил действие в приложении: "{action}".
        Приложение ответило: {tool_response}.
        Текущее состояние: {current_state}.

        Определи новое эмоциональное состояние клиента после этого события.
        Учитывай, как реальные люди реагируют на успехи, ошибки, одобрения и отказы банка.
        Верни новые значения эмоций.
        """
        result = await robust_llm_call(prompt, structured_output=EmotionalStateSchema)
        return result
