"""
RAG-система с использованием Qdrant (локальный режим).
Интеграция через search_rag.py от Карина.
"""

from typing import Dict, List
from multi_agent_rag.config import CATEGORY_SLUG

# Используем актуальный поиск из папки rag/
from rag.search_rag import search as qdrant_search


class RAGSystem:
    """RAG-система с использованием Qdrant (локальный режим)"""

    def __init__(self, collection_name: str = "telegram_news", local: bool = True):
        self.collection_name = collection_name
        self.local = local
        self.storage_path = "qdrant_data"
        self.news_cache: Dict[str, List[dict]] = {}

        print("RAG-система инициализирована (Qdrant)")
        print(f"Коллекция: {collection_name}")
        print(f"Локальный режим: {local}")
        print(f"Путь к данным: {self.storage_path}")

    def get_relevant_context(self, category: str, query: str, k: int = 5) -> List[str]:
        """Получает релевантные фрагменты через Qdrant search."""
        agent = CATEGORY_SLUG.get(category)

        try:
            results = qdrant_search(
                query=query,
                collection_name=self.collection_name,
                local=self.local,
                storage_path=self.storage_path,
                top_k=k,
                agent=agent
            )

            # Кэширование
            cache_key = f"{category}_{query}"
            self.news_cache[cache_key] = results

            # Извлекаем тексты
            texts = [r["text"] for r in results if r.get("text")]
            print(f"{category}: найдено {len(texts)} новостей")
            return texts

        except Exception as e:
            print(f"Ошибка поиска в {category}: {e}")
            return []

    def get_cached_results(self, category: str, query: str) -> List[dict]:
        """Возвращает полные результаты поиска (с метаданными)"""
        cache_key = f"{category}_{query}"
        return self.news_cache.get(cache_key, [])