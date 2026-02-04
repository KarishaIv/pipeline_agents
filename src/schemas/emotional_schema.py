from pydantic import BaseModel, Field

class EmotionalStateSchema(BaseModel):
    """Эмоциональное состояние клиента"""
    mood: float = Field(..., ge=0, le=1, description="Общее настроение клиента")
    stress: float = Field(..., ge=0, le=1, description="Уровень стресса")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность в своих действиях")
    trust_in_bank: float = Field(..., ge=0, le=1, description="Доверие к банку")
    urgency: float = Field(..., ge=0, le=1, description="Насколько клиент чувствует срочность своей задачи")

    class Config:
        title = "EmotionalState"
        json_schema_extra = {"example": {
            "mood": 0.7,
            "stress": 0.2,
            "confidence": 0.8,
            "trust_in_bank": 0.6,
            "urgency": 0.5
        }}
