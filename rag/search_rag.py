"""
Поиск по RAG: запрос → эмбеддинг → поиск в Qdrant → топ-k документов.

Использует ту же модель и коллекцию, что и ingest_rag.py.
Сначала один раз запускаем ingest_rag.py, потом вызываем этот скрипт или search() из кода
"""

import argparse
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
E5_QUERY_PREFIX = "query: "
_embedding_model = None


def _build_embedding_model() -> SentenceTransformer:
    """Создаёт singleton локальной embedding-модели."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_embedding(text: str, query: bool = True) -> list:
    """Возвращает embedding текста для query или document."""
    model = _build_embedding_model()
    prefix = E5_QUERY_PREFIX if query else ""
    return model.encode([prefix + text], show_progress_bar=False).tolist()[0]


def search(
    query: str,
    collection_name: str = "telegram_news",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    local: bool = False,
    storage_path: str = "qdrant_data",
    top_k: int = 5,
    agent: Optional[str] = None,
):
    """
    Семантический поиск по коллекции Qdrant.

    query — текст запроса (например: "курс доллара", "ипотека").
    agent — опциональный фильтр: только документы для этого агента
            (macroeconomy, banks, currency, real_estate, social_news).
    local=True и storage_path — те же, что при ingest_rag.py --local.

    Возвращает список dict: [{"id", "text", "metadata", "score"}, ...].
    """
    q_vec = get_embedding(query, query=True)

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

    out = []
    for hit in response.points:
        out.append({
            "id": hit.id,
            "text": hit.payload.get("text", ""),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            "score": hit.score,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Поиск по RAG (Qdrant)")
    parser.add_argument("query", nargs="+", help="Текст запроса в кавычках, например: \"курс доллара\"")
    parser.add_argument("--collection", default="telegram_news")
    parser.add_argument("--local", action="store_true", help="Локальное хранилище (как при ingest_rag.py --local)")
    parser.add_argument("--storage-path", default="qdrant_data", help="Папка при --local")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--agent", default=None, help="Фильтр: macroeconomy, banks, currency, real_estate, social_news")
    args = parser.parse_args()

    q = " ".join(args.query)
    print("Поиск...")
    results = search(
        q,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        local=args.local,
        storage_path=args.storage_path,
        top_k=args.top_k,
        agent=args.agent,
    )
    for i, r in enumerate(results, 1):
        print(f"\n--- {i} (score={r['score']:.4f}) [{r['metadata'].get('agent', '')}] ---")
        print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))


if __name__ == "__main__":
    main()
