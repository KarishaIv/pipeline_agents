from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

class DecisionOutcome(BaseModel):
    will_take_credit: bool = Field(..., description="Примет ли клиент кредитное предложение")
    probability_score: float = Field(..., ge=0, le=1, description="Уверенность агента в решении")
    reasoning: str = Field(..., description="Обоснование решения")
    emotional_factors: str = Field(..., description="Какие эмоции повлияли на решение")
    decision_mode: Optional[Literal["direct", "compact_debate"]] = Field(
        default=None,
        description="Режим принятия решения: baseline direct или compact debate.",
    )
    voice_stances: Optional[Dict[str, str]] = Field(
        default=None,
        description="Итоговые позиции внутренних голосов, если использовался debate.",
    )
    conflict_summary: Optional[str] = Field(
        default=None,
        description="Краткое описание конфликта между голосами, если он был.",
    )
