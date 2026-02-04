from pydantic import BaseModel, Field

class FinancialPush(BaseModel):
    message: str = Field(..., description="Текст push-уведомления")
    tone: str = Field(..., description="Тон уведомления (дружелюбный, срочный, рациональный)")
    rationale: str = Field(..., description="Почему уведомление выбрано именно таким")

class FinancialPrediction(BaseModel):
    probability_take_credit: float = Field(..., ge=0, le=1, description="Вероятность согласия на кредит")
    reasoning: str = Field(..., description="Пояснение вероятности")
