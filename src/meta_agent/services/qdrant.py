"""Сервис для работы с Qdrant.

Предоставляет низкоуровневые функции поиска, фильтрации и прокрутки.
Реализован как singleton для упрощения использования в инструментах.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from src.meta_agent.catalog import CollectionName
from src.utils import get_embedding


class QdrantService:
    """Singleton сервис для взаимодействия с Qdrant.

    Методы выполняют операции без обработки ошибок. Все исключения (включая
    ошибки QdrantClient) должны перехватываться в qdrant_tools.py.
    """

    _instance: QdrantService | None = None

    def __new__(cls) -> "QdrantService":
        """Singleton pattern."""
        if cls._instance is None:
            from src.meta_agent.config import QDRANT_HOST, QDRANT_PORT

            cls._instance = super().__new__(cls)
            cls._instance.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )
        return cls._instance

    def _point_to_dict(self, point: Any) -> Dict[str, Any]:
        """Преобразует одну точку Qdrant в словарь."""
        return {
            "id": getattr(point, "id", None),
            "score": getattr(point, "score", None),
            "payload": getattr(point, "payload", None),
        }

    def _points_to_dict(self, points: List[Any]) -> List[Dict[str, Any]]:
        """Преобразует список точек Qdrant в список словарей."""
        return [self._point_to_dict(p) for p in points]

    def search(
        self,
        collection: CollectionName,
        query: str,
        vector_name: str = "embedding",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Поиск по векторному сходству."""
        vector = get_embedding(query)
        hits = self.client.query_points(
            collection_name=collection,
            query=vector,
            using=vector_name,
            limit=limit,
        ).points
        return [self._point_to_dict(h) for h in hits]

    def filter_points(
        self,
        collection: CollectionName,
        field: str,
        value: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Фильтрация по полю payload."""
        results, _ = self.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value),
                    )
                ]
            ),
            limit=limit,
        )
        return [self._point_to_dict(p) for p in results]

    def scroll_points(
        self,
        collection: CollectionName,
        limit: int = 10,
        offset: Optional[str] = None,
        payload_fields: Optional[List[str]] = None,
        filter_field: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Прокрутка (scroll) точек с optional фильтром и выборкой полей."""
        scroll_filter = None
        if filter_field is not None and filter_value is not None:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_field,
                        match=models.MatchValue(value=filter_value),
                    )
                ]
            )

        with_payload: bool | models.PayloadSelectorInclude = (
            models.PayloadSelectorInclude(include=payload_fields) if payload_fields else True
        )

        results, next_offset = self.client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
        )
        return {
            "points": [self._point_to_dict(p) for p in results],
            "next_offset": next_offset,
        }

    def retrieve_by_id(self, collection: CollectionName, ids: List[str]) -> List[Dict[str, Any]]:
        """Получение точек по ID (возвращает список)."""
        points = self.client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=True,
        )
        if points:
            return self._points_to_dict(points)
        raise ValueError(f"Point {ids} not found in collection {collection}")

    def get_collection_schema(self, collection: CollectionName) -> Dict[str, Any]:
        """Получение схемы коллекции (поля payload и вектора)."""
        info = self.client.get_collection(collection)
        return {
            "collection": collection,
            "payload_schema": getattr(info, "payload_schema", {}),
            "vectors": getattr(info, "vectors", {}),
        }

