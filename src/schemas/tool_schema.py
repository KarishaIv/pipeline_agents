from pydantic import BaseModel, Field
from typing import List

class ToolResponseSchema(BaseModel):
    message: str = Field(..., description="Текст, который отображается пользователю в приложении")
    status: str = Field(..., description="Тип ответа: success / fail / info")
    suggested_actions: List[str] = Field(..., description="Доступные действия после этого шага")

    class Config:
        title = "ToolResponse"
        json_schema_extra = {"example": {
            "message": "Ваш перевод успешно выполнен!",
            "status": "success",
            "suggested_actions": ["Посмотреть детали", "Вернуться на главную"]
        }}
