"""Каталог коллекций Qdrant для мета-агента.

Содержит список доступных коллекций и их описания. Используется в промптах и инструментах.
"""

from typing import Literal

# Список доступных коллекций Qdrant (должен совпадать с реальными коллекциями в базе)
AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations"]
CollectionName = Literal["questions", "personas", "target_audiences", "simulations"]

COLLECTION_ENUM_DESC = "Имя коллекции Qdrant (как в базе): " + ", ".join(AVAILABLE_COLLECTIONS)

# Краткие описания для промптов агента-извлекателя
COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "questions": "Тексты вопросов/сценариев опроса",
    "personas": "Синтетические персоны с социо-демографическими и психологическими характеристиками",
    "target_audiences": "Сегменты целевой аудитории с описанием и агрегированными характеристиками",
    "simulations": "Результаты симуляции ответов персон на вопросы с рассуждениями и решением",
}


def get_collection_catalog() -> str:
    """Возвращает форматированный каталог коллекций для включения в системные промпты."""
    return "\n".join(
        f"  • {name} — {COLLECTION_DESCRIPTIONS.get(name, 'Нет описания')}"
        for name in AVAILABLE_COLLECTIONS
    )
