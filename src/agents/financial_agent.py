from src.utils import robust_llm_call
from src.schemas.financial_schema import FinancialPrediction, FinancialPush

class FinancialAgent:
    """
    Финансовый агент создает персонализированное push-уведомление
    и прогнозирует, как часто клиент согласится на кредит.
    """

    async def generate_push(self, profile: dict, session_history: list) -> dict:
        prompt_push = f"""
        На основе описания клиента: {profile}
        и его поведения в приложении: {session_history},

        Сформулируй персонализированное push-уведомление, которое
        должно убедить клиента взять кредит. 
        Сделай текст реалистичным, не навязчивым, главная цель - продать пользователю кредит.
        """
        push = await robust_llm_call(prompt_push, structured_output=FinancialPush)

        prompt_pred = f"""
        Клиент с профилем {profile} получил уведомление:
        "{push.message}"

        По его поведению в приложении ({session_history}),
        оцени в скольки процентах случаев он бы согласился взять кредит после этого пуша.
        Объясни почему.
        """
        prediction = await robust_llm_call(prompt_pred, structured_output=FinancialPrediction)

        return {
            "push": push.dict(),
            "prediction": prediction.dict(),
        }
