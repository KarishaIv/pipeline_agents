"""
Поиск по RAG: запрос => эмбеддинг => поиск в Qdrant => топ-k документов.
Использует ту же модель и коллекцию, что и ingest_rag.py.
Поддерживает offline-режим (local_files_only).
"""
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["SENTENCE_TRANSFORMERS_OFFLINE"] = "1"

import argparse
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Попробуем загрузить sentence-transformers, но с фоллбэком
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Те же настройки модели, что и при ingest
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
E5_QUERY_PREFIX = "query: "


class SimpleEmbeddings:
    """
    Заглушка для эмбеддингов: простые векторы на основе частотности ключевых слов.
    Используется, если sentence-transformers не доступен или модель не загружена.
    """

    def __init__(self):
        # Ключевые слова для банковского контекста
        self.keywords = [
            'кредит', 'ипотека', 'банк', 'ставка', 'доллар', 'евро', 'рубль',
            'инфляция', 'доход', 'расход', 'сбережения', 'вклад', 'заем',
            'платеж', 'процент', 'валюта', 'рынок', 'финансы', 'экономика'
        ]

    def encode(self, texts: List[str], show_progress_bar: bool = False, **kwargs) -> 'np.ndarray':
        import numpy as np
        result = []
        for text in texts:
            text_lower = text.lower()
            # Вектор: частота каждого ключевого слова (нормализованная)
            vector = [text_lower.count(kw) / max(1, len(text_lower)) for kw in self.keywords]
            # Дополняем до 384 dim нулями
            while len(vector) < 384:
                vector.append(0.0)
            result.append(vector[:384])
        return np.array(result)

    def get_sentence_embedding_dimension(self) -> int:
        return 384


def get_embedding_model(name: str = EMBEDDING_MODEL, local_only: bool = True):
    """
    Загружает embedding-модель с поддержкой offline-режима.

    Args:
        name: имя модели
        local_only: если True, не пытаться подключаться к интернету

    Returns:
        tuple: (модель, размерность эмбеддинга)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"sentence-transformers не установлен, используем заглушку")
        return SimpleEmbeddings(), 384

    try:
        # Пытаемся загрузить модель
        model = SentenceTransformer(
            name,
            trust_remote_code=True,
            local_files_only=local_only  # Ключевой параметр: не подключаться к интернету
        )
        dim = model.get_sentence_embedding_dimension()
        print(f"Модель загружена: {name} ({dim} dim)")
        return model, dim

    except Exception as e:
        print(f"Не удалось загрузить модель {name}: {e}")
        print("Используем упрощенные эмбеддинги (ключевые слова)")
        return SimpleEmbeddings(), 384


def search(
    query: str,
    collection_name: str = "telegram_news",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    local: bool = False,
    storage_path: str = "qdrant_data",
    top_k: int = 5,
    agent: Optional[str] = None,
    model: Optional = None,
) -> List[Dict[str, Any]]:
    """
    Семантический поиск по коллекции Qdrant.

    Args:
        query: текст запроса
        collection_name: имя коллекции в Qdrant
        local: если True, использовать локальное хранилище
        storage_path: путь к локальному хранилищу
        top_k: количество результатов
        agent: опциональный фильтр по агенту
        model: опционально переданная модель (чтобы не загружать каждый раз)

    Returns:
        список словарей: [{"id", "text", "metadata", "score"}, ...]
    """
    # Загружаем модель, если не передана
    if model is None:
        # Определяем, нужно ли работать в offline-режиме
        local_only = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1" or local
        model, _ = get_embedding_model(local_only=local_only)

    # Эмбеддинг запроса с префиксом для E5 (если модель поддерживает encode с preprocessed текстом)
    try:
        q_vec = model.encode([E5_QUERY_PREFIX + query], show_progress_bar=False).tolist()[0]
    except Exception:
        # Фоллбэк: кодируем без префикса
        q_vec = model.encode([query], show_progress_bar=False).tolist()[0]

    # Подключаемся к Qdrant
    if local:
        client = QdrantClient(path=storage_path)
    else:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Фильтр по агенту, если задан
    query_filter = None
    if agent:
        query_filter = Filter(
            must=[FieldCondition(key="agent", match=MatchValue(value=agent))]
        )

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=q_vec,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
    except ResponseHandlingException as e:
        if "Connection refused" in str(e) or "refused" in str(e).lower():
            print("Ошибка: Qdrant не запущен. Используйте --local, если загружали через ingest_rag.py --local")
        raise

    # Формируем результат
    out = []
    for hit in response.points:
        out.append({
            "id": hit.id,
            "text": hit.payload.get("text", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            "score": float(hit.score) if hit.score is not None else 0.0,
        })
    return out


def main():
    """CLI для тестирования поиска."""
    parser = argparse.ArgumentParser(description="Поиск по RAG (Qdrant)")
    parser.add_argument("query", nargs="+", help="Текст запроса")
    parser.add_argument("--collection", default="telegram_news")
    parser.add_argument("--local", action="store_true", help="Локальное хранилище")
    parser.add_argument("--storage-path", default="qdrant_data")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--agent", default=None, help="Фильтр по агенту")
    args = parser.parse_args()

    query = " ".join(args.query)
    print(f"Запрос: {query}")
    print(f"Агент: {args.agent or 'все'}")

    # Загружаем модель один раз
    local_only = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1" or args.local
    model, dim = get_embedding_model(local_only=local_only)
    print(f"Модель: {dim} dim")

    # Поиск
    print("Поиск")
    results = search(
        query,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        local=args.local,
        storage_path=args.storage_path,
        top_k=args.top_k,
        agent=args.agent,
        model=model,
    )

    # Вывод результатов
    for i, r in enumerate(results, 1):
        print(f"\n--- {i} (score={r['score']:.4f}) [{r['metadata'].get('agent', '')}] ---")
        print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))


if __name__ == "__main__":
    main()