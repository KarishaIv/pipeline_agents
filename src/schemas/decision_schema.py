from pydantic import BaseModel, Field

class DecisionOutcome(BaseModel):
    will_take_credit: bool = Field(..., description="Примет ли клиент кредитное предложение")
    probability_score: float = Field(..., ge=0, le=1, description="Уверенность агента в решении")
    reasoning: str = Field(..., description="Обоснование решения")
    emotional_factors: str = Field(..., description="Какие эмоции повлияли на решение")
