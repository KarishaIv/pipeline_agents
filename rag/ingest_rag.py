"""
Загрузка RAG-документов в Qdrant

Когда запускать:
  - Один раз при первой загрузке (после того как есть rag_docs.jsonl)
  - Потом — каждый раз, когда обновили rag_docs.jsonl и хотим обновить векторную БД

Поиск потом делается скриптом search_rag.py

Запуск:
  Без Docker (локальная папка):
    python ingest_rag.py --local
  С Docker:
    python ingest_rag.py
    python ingest_rag.py --input rag_docs.jsonl --collection telegram_news --recreate
"""

import argparse
import json
from pathlib import Path
from typing import Generator
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BATCH_SIZE = 64
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
E5_DOC_PREFIX = "passage: "


def load_jsonl(path: str) -> Generator[dict, None, None]:
    """Читает JSONL, выдаёт по одному dict на строку."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


_embedding_model = None


def _build_embedding_model() -> SentenceTransformer:
    """Создаёт singleton локальной embedding-модели."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_document_embeddings(texts: list) -> list:
    """Возвращает embeddings для списка документов."""
    model = _build_embedding_model()
    prefixed = [E5_DOC_PREFIX + text for text in texts]
    return model.encode(prefixed, show_progress_bar=False).tolist()


def get_vector_size() -> int:
    """Определяет размер вектора модели."""
    model = _build_embedding_model()
    return model.get_sentence_embedding_dimension()


def create_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool = False):
    """Создаёт коллекцию в Qdrant (при recreate удаляет существующую)."""
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _make_qdrant_client(local: bool = False, storage_path: str = "qdrant_data", host: str = "localhost", port: int = 6333):
    """Создаёт клиент: локальное хранилище (без Docker) или подключение к серверу."""
    if local:
        return QdrantClient(path=storage_path)
    return QdrantClient(host=host, port=port)


def _make_point_id(doc_id: str):
    """
    Для локального Qdrant используем UUID/int id.
    Делаем стабильный UUID из исходного doc_id, чтобы upsert был детерминированным.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def ingest(
    input_jsonl: str = "rag_docs.jsonl",
    collection_name: str = "telegram_news",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    local: bool = False,
    storage_path: str = "qdrant_data",
    batch_size: int = BATCH_SIZE,
    recreate_collection: bool = False,
):
    """
    Читает JSONL, считает эмбеддинги, загружает в Qdrant.
    При local=True хранилище в папке storage_path (Docker не нужен).
    """
    print("Чтение документов из JSONL...")
    docs = list(load_jsonl(input_jsonl))
    if not docs:
        print("Нет документов в файле.")
        return

    print("Инициализация локальной embedding-модели...")
    vector_size = get_vector_size()

    if local:
        print("Подключение к локальному хранилищу Qdrant (папка:", storage_path, ")...")
        client = _make_qdrant_client(local=True, storage_path=storage_path)
    else:
        print("Подключение к Qdrant...")
        try:
            client = _make_qdrant_client(host=qdrant_host, port=qdrant_port)
            client.get_collections()
        except ResponseHandlingException as e:
            if "Connection refused" in str(e) or "refused" in str(e).lower():
                print("\nОшибка: не удаётся подключиться к Qdrant (сервер не запущен).")
                print("Вариант без Docker: запустите с --local (данные в папке qdrant_data):")
                print("  python ingest_rag.py --local")
                print("Либо запустите Qdrant: docker run -p 6333:6333 qdrant/qdrant\n")
            raise

    print("Создание коллекции...")
    create_collection(client, collection_name, vector_size, recreate=recreate_collection)

    print(f"Эмбеддинги и загрузка в Qdrant ({len(docs)} документов, батч {batch_size})...")
    points = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Upsert"):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        vectors = get_document_embeddings(texts)

        for j, (doc, vec) in enumerate(zip(batch, vectors)):
            point_id = _make_point_id(doc["id"])
            payload = {
                "doc_id": doc["id"],
                "text": doc["text"],
                **doc["metadata"],
            }
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)

    print(f"Готово. В коллекции '{collection_name}' загружено {len(docs)} документов.")


def main():
    parser = argparse.ArgumentParser(description="Загрузка rag_docs.jsonl в Qdrant")
    parser.add_argument("--input", default="rag_docs.jsonl", help="Путь к JSONL")
    parser.add_argument("--collection", default="telegram_news", help="Имя коллекции Qdrant")
    parser.add_argument("--local", action="store_true", help="Локальное хранилище в папке (Docker не нужен)")
    parser.add_argument("--storage-path", default="qdrant_data", help="Папка при --local")
    parser.add_argument("--host", default="localhost", help="Хост Qdrant (если не --local)")
    parser.add_argument("--port", type=int, default=6333, help="Порт Qdrant (если не --local)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Размер батча эмбеддингов")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию (удалить и создать заново)")
    args = parser.parse_args()

    ingest(
        input_jsonl=args.input,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        local=args.local,
        storage_path=args.storage_path,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
    )


if __name__ == "__main__":
    main()
