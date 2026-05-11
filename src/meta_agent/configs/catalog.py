"""Catalog of Qdrant collections for meta-agent.

Contains list of available collections and their descriptions.
Used in prompts and tools.
"""

from typing import Literal

# List of available Qdrant collections (must match real collections in the database)
AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations", "world_contexts"]
CollectionName = Literal["questions", "personas", "target_audiences", "simulations", "world_contexts"]

COLLECTION_ENUM_DESC = "Имя коллекции Qdrant (как в базе): " + ", ".join(AVAILABLE_COLLECTIONS)

# Brief descriptions for agent prompts
COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "questions": "Тексты вопросов/сценариев опроса",
    "personas": "Синтетические персоны с социо-демографическими и психологическими характеристиками и названием целевой аудитории в target_audience_name",
    "target_audiences": "Сегменты целевой аудитории с описанием и агрегированными характеристиками",
    "simulations": "Результаты симуляции ответов персон на вопросы с рассуждениями и решением. С уверенностью в решении, указанными UUID персон, вопросов и контекста",
    "world_contexts": "Новостной/мировой контекст мира для целевых аудиторий",
}


def get_collection_catalog() -> str:
    """Return a formatted collection catalog for inclusion in system prompts."""
    return "\n".join(
        f"  • {name} — {COLLECTION_DESCRIPTIONS.get(name, 'Нет описания')}"
        for name in AVAILABLE_COLLECTIONS
    )
