from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from src.schemas.emotional_schema import EmotionalStateSchema
from src.schemas.tool_schema import ToolResponseSchema

class PersonaGoal(BaseModel):
    goal_description: str = Field(..., description="Основная цель клиента в приложении")
    motivation: str = Field(..., description="Почему он выбрал именно эту цель")

class PersonaAction(BaseModel):
    next_action: str = Field(..., description="Следующее действие клиента в приложении")
    reasoning: str = Field(..., description="Объяснение мотивации этого действия")

class PersonaReaction(BaseModel):
    reaction_type: str = Field(..., description="Как клиент отреагировал на уведомление (открыл, проигнорировал, заинтересовался)")
    reasoning: str = Field(..., description="Почему он так отреагировал")
    emotional_change: Optional[str] = Field(None, description="Как изменилось эмоциональное состояние клиента")

class PersonaSessionRecord(BaseModel):
    step: int = Field(..., description="Номер шага")
    action: str = Field(..., description="Действие, выполненное пользователем")
    tool_result: ToolResponseSchema = Field(..., description="Ответ приложения")
    emotional_state: Dict = Field(..., description="Эмоции после шага")
