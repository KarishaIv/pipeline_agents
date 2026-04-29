"""Инициализация коллекций Qdrant из parquet-файлов."""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DATA_DIR = Path("data_4_qdrant")

COLLECTIONS = [
    {
        "name": "questions",
        "parquet": "questions.parquet",
        "index_column": "UUID",
        "vector_keys": ["embedding"],
        "payload_keys": ["question"],
    },
    {
        "name": "personas",
        "parquet": "personas.parquet",
        "index_column": "UUID",
        "vector_keys": ["embedding"],
        "payload_keys": ["region", "age_group", "education", "income_level", "marital_status", "gender", "children_group", "occupation","cluster_id", "group_key", "n_americans_in_group", "n_clusters_total", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "cluster_weight", "target_audience_id", "target_audience_name", "data_source"],
    },
    {
        "name": "target_audiences",
        "parquet": "target_audiences.parquet",
        "index_column": "UUID",
        "vector_keys": ["embedding"],
        "payload_keys": None,
    },
    {
        "name": "simulations",
        "parquet": "simulations.parquet",
        "index_column": "UUID",
        "vector_keys": ["emotional_vector", "rational_vector", "social_vector", "ideological_vector", "decision_vector", "general_vector"],
        "payload_keys": None,
    },
]


def _infer_schema_type(series: pd.Series) -> Optional[PayloadSchemaType]:
    """Сопоставить колонку DataFrame с типом Qdrant PayloadSchemaType или вернуть None для пропуска."""
    if pd.api.types.is_bool_dtype(series):
        return PayloadSchemaType.BOOL
    if pd.api.types.is_integer_dtype(series):
        return PayloadSchemaType.INTEGER
    if pd.api.types.is_float_dtype(series):
        return PayloadSchemaType.FLOAT

    sample = series.dropna()
    if sample.empty:
        return PayloadSchemaType.KEYWORD

    first = sample.iloc[0]
    if isinstance(first, (list, tuple)):
        if not first:
            return None
        elem = first[0]
        if isinstance(elem, str):
            return PayloadSchemaType.KEYWORD
        if isinstance(elem, (int, np.integer)):
            return PayloadSchemaType.INTEGER
        if isinstance(elem, (float, np.floating)):
            return PayloadSchemaType.FLOAT
        if isinstance(elem, bool):
            return PayloadSchemaType.BOOL
        return None
    if isinstance(first, (dict, np.ndarray)):
        return None
    if isinstance(first, (np.bool_, bool)):
        return PayloadSchemaType.BOOL
    if isinstance(first, (np.integer, int)):
        return PayloadSchemaType.INTEGER
    if isinstance(first, (np.floating, float)):
        return PayloadSchemaType.FLOAT
    if isinstance(first, str):
        max_len = int(sample.astype(str).str.len().max())
        if max_len > 200:
            return PayloadSchemaType.TEXT
        return PayloadSchemaType.KEYWORD
    return PayloadSchemaType.KEYWORD


def _create_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    df: pd.DataFrame,
    payload_keys: List[str],
) -> None:
    """Создать payload-индексы для всех индексируемых payload-колонок."""
    for field in payload_keys:
        if field not in df.columns:
            continue
        schema_type = _infer_schema_type(df[field])
        if schema_type is None:
            logger.info("Пропускаем индекс %s.%s (нескалярный тип)", collection_name, field)
            continue
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema_type,
                wait=True,
            )
            logger.info("Создан индекс %s.%s как %s", collection_name, field, schema_type.value)
        except Exception as exc:
            logger.warning("Не удалось создать индекс %s.%s: %s", collection_name, field, exc)


def create_collection_from_parquet(
    client: QdrantClient,
    collection_name: str,
    parquet_path: Path,
    index_column: str = "UUID",
    payload_keys: Optional[List[str]] = None,
    vector_keys: List[str] = ["embedding"],
    distance: Distance = Distance.COSINE,
) -> int:
    """Создаёт коллекцию из parquet файла с именованными векторами."""
    if payload_keys is None:
        payload_keys = []

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    df = pd.read_parquet(parquet_path)
    vectors_config = {
        key: VectorParams(size=len(df.iloc[0][key]), distance=distance)
        for key in vector_keys
    }
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=row[index_column],
                vector={k: row[k].tolist() for k in vector_keys},
                payload={k: row[k].tolist() if isinstance(row[k], np.ndarray) else row[k] for k in payload_keys},
            )
            for _, row in df.iterrows()
        ],
    )

    _create_payload_indexes(client, collection_name, df, payload_keys)

    return df.shape[0]


def init_qdrant() -> QdrantClient:
    """Подключается к Qdrant и создаёт коллекции из локальных parquet-файлов."""
    client = QdrantClient(host="localhost", port=6333)

    for collection in COLLECTIONS:
        parquet_path = DATA_DIR / collection["parquet"]
        if not parquet_path.exists():
            logger.warning(f"Skipping {collection['name']}: {parquet_path} not found")
            continue

        if collection["payload_keys"] is None:
            collection["payload_keys"] = list(
                set(pq.read_schema(parquet_path).names)
                - set(collection["vector_keys"])
                - set([collection["index_column"]])
            )

        points_count = create_collection_from_parquet(
            client=client,
            collection_name=collection["name"],
            parquet_path=parquet_path,
            index_column=collection["index_column"],
            vector_keys=collection["vector_keys"],
            payload_keys=collection["payload_keys"],
        )
        logger.info(f"Created collection '{collection['name']}' ({points_count} points)")

    return client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = init_qdrant()
    logger.info("Qdrant инициализирован, коллекции готовы.")
