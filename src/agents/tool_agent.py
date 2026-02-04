from src.utils import robust_llm_call
from src.schemas.tool_schema import ToolResponseSchema

class ToolAgent:
    """Эмулирует банковское приложение: отвечает на действия клиента естественно"""

    async def respond(self, action: str, profile: dict, history: list) -> dict:
        prompt = f"""
        Ты — интерфейс банковского приложения. Клиент выполняет действие: "{action}".
        Его профиль: {profile}.
        История взаимодействий: {history}.

        Эмулируй реальный ответ приложения:
        - покажи сообщение пользователю
        - верни статус (success / fail / info)
        - предложи следующие возможные действия

        Формулируй как реальный UX-текст банковского интерфейса.
        """
        result = await robust_llm_call(prompt, structured_output=ToolResponseSchema)
        return result.dict()
