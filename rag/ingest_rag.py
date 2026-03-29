"""
Загрузка RAG в Qdrant: читает выход prep_rag (parquet или jsonl), считает эмбеддинги,
делает upsert по стабильному doc_id.

Когда запускать:
  - После rag/prep_rag.py: во входе  rag_docs.parquet 
  - Повторно — каждый день/порцию: тот же скрипт без --recreate: новые точки добавятся,
    уже существующие doc_id перезапишутся
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import DatetimeRange, Distance, FieldCondition, Filter, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BATCH_SIZE = 64
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
E5_DOC_PREFIX = "passage: "


def load_jsonl(path: str) -> Generator[dict, None, None]:
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
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_document_embeddings(texts: list) -> list:
    model = _build_embedding_model()
    prefixed = [E5_DOC_PREFIX + text for text in texts]
    return model.encode(prefixed, show_progress_bar=False).tolist()


def get_vector_size() -> int:
    model = _build_embedding_model()
    return model.get_sentence_embedding_dimension()


def create_collection(client: QdrantClient, collection_name: str, vector_size: int, recreate: bool = False):
    if recreate:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    exists = False
    try:
        exists = bool(client.collection_exists(collection_name=collection_name))
    except Exception:
        try:
            client.get_collection(collection_name=collection_name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _make_qdrant_client(local: bool = False, storage_path: str = "qdrant_data", host: str = "localhost", port: int = 6333):
    if local:
        return QdrantClient(path=storage_path)
    return QdrantClient(host=host, port=port)


def _make_point_id(doc_id: str):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _load_docs(input_path: str) -> list[dict]:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")

    if p.suffix.lower() == ".jsonl":
        return list(load_jsonl(str(p)))

    if p.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("Для чтения parquet нужен pandas + pyarrow/fastparquet") from e

        df = pd.read_parquet(str(p))
        docs = []
        for _, row in df.iterrows():
            meta = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
            docs.append({"id": row["doc_id"], "text": row["text"], "metadata": meta})
        return docs

    raise ValueError(f"Неподдерживаемый формат: {p.suffix}. Ожидаю .jsonl или .parquet")


def _prune_by_payload_date(
    client: QdrantClient,
    collection_name: str,
    older_than_days: int,
) -> None:
    if older_than_days <= 0:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    print(f"Удаление документов с date старше {older_than_days} дней (до {cutoff.isoformat()})...")
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="date", range=DatetimeRange(lt=cutoff))]
        ),
    )


def ingest(
    input_path: str = "rag_docs.parquet",
    collection_name: str = "telegram_news",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    local: bool = False,
    storage_path: str = "qdrant_data",
    batch_size: int = BATCH_SIZE,
    recreate_collection: bool = False,
    prune_older_than_days: int = 60,
):
    print("Чтение документов...")
    docs = _load_docs(input_path)
    if not docs:
        print("Нет документов в файле.")
        return

    print("Инициализация локальной embedding-модели...")
    vector_size = get_vector_size()
    print(f"Размер вектора: {vector_size}")
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

    print("Проверка/создание коллекции...")
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

    _prune_by_payload_date(client, collection_name, prune_older_than_days)

    print(f"Готово. В коллекции '{collection_name}' обработано {len(docs)} документов.")


def main():
    parser = argparse.ArgumentParser(description="Загрузка rag_docs в Qdrant (инкрементально через upsert)")
    parser.add_argument("--input", default="rag_docs.parquet", help="Путь к .parquet или .jsonl")
    parser.add_argument("--collection", default="telegram_news", help="Имя коллекции Qdrant")
    parser.add_argument("--local", action="store_true", help="Локальное хранилище в папке (Docker не нужен)")
    parser.add_argument("--storage-path", default="qdrant_data", help="Папка при --local")
    parser.add_argument("--host", default="localhost", help="Хост Qdrant (если не --local)")
    parser.add_argument("--port", type=int, default=6333, help="Порт Qdrant (если не --local)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Размер батча эмбеддингов")
    parser.add_argument("--recreate", action="store_true", help="Пересоздать коллекцию (удалить и создать заново)")
    parser.add_argument(
        "--prune-older-than-days",
        type=int,
        default=60,
        help="После upsert удалить точки, у которых payload date старше N дней (0 = не удалять). По умолчанию 60.",
    )
    args = parser.parse_args()

    ingest(
        input_path=args.input,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        local=args.local,
        storage_path=args.storage_path,
        batch_size=args.batch_size,
        recreate_collection=args.recreate,
        prune_older_than_days=args.prune_older_than_days,
    )


if __name__ == "__main__":
    main()
