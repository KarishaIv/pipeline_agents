persona_response_schema = {
    "title": "PersonaResponse",
    "description": "Структурированный ответ персоны на предложение банка о кредитной карте.",
    "type": "object",
    "properties": {
        "will_take_credit": {
            "type": "boolean",
            "description": "Примет ли персона предложение кредита"
        },
        "probability_score": {
            "type": "number",
            "enum": ["0", "1"],
            "description": "Оценка вероятности в диапазоне 0-1"
        },
        "key_factors": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Ключевые факторы (1-2 слова), повлиявшие на решение"
        },
        "risk_assessment": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Оценка риска: низкий, средний, высокий"
        },
        "call_to_action_response": {
            "type": "string",
            "enum": ["likely_click", "hesitant", "will_ignore"],
            "description": "Реакция на предложение: кликнет, сомневается, проигнорирует"
        },
        "reasoning": {
            "type": "string",
            "description": "Объяснение решения персоны в 2-3х коротких предложениях"
        }
    },
    "required": [
        "will_take_credit",
        "probability_score",
        "key_factors",
        "risk_assessment",
        "call_to_action_response",
        "reasoning",
    ]
}