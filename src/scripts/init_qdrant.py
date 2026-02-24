"""Инициализация коллекций Qdrant из parquet-файлов."""

import logging
from pathlib import Path
from typing import List

import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)


def create_simple_collection_from_parquet(
    client: QdrantClient,
    collection_name: str,
    parquet_path: Path,
    payload_keys: List[str] = ["text"],
    distance: Distance = Distance.COSINE,
) -> None:
    """Создаёт простую коллекцию из parquet файла."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    df = pd.read_parquet(parquet_path)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=df.iloc[0]["embedding"].shape[0], distance=distance),
    )

    for index, row in df.iterrows():
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=index,
                    vector=row["embedding"],
                    payload={key: row[key] for key in payload_keys},
                ),
            ]
        )


def init_qdrant() -> QdrantClient:
    """Подключается к Qdrant и создаёт коллекции из локальных parquet-файлов."""
    client = QdrantClient(url="http://localhost:6333")

    create_simple_collection_from_parquet(
        client=client,
        collection_name="questions",
        parquet_path=Path("data_4_qdrant/questions.parquet"),
        payload_keys=["question"],
    )

    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = init_qdrant()
    logger.info("Qdrant initialised, collections ready.")
