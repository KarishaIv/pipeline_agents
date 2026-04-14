"""Функции клиента Qdrant и обёртки BaseTool для агентов."""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field
from qdrant_client import QdrantClient, models

from sgr_agent_core.base_tool import BaseTool

from src.meta_agent.tools.dto_tools import dto_summary_view, register_dto
from src.utils import get_embedding

logger = logging.getLogger("meta_agent.qdrant")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations"]
COLLECTION_ENUM_DESC = "Имя коллекции Qdrant (как в базе): " + ", ".join(AVAILABLE_COLLECTIONS)

CollectionName = Literal["questions", "personas", "target_audiences", "simulations"]

# ---------------------------------------------------------------------------
# Низкоуровневый клиент Qdrant
# ---------------------------------------------------------------------------

_qdrant = QdrantClient(host="localhost", port=6333)


def _qdrant_failure_payload(operation: str, exc: BaseException) -> str:
    """Вернуть JSON-ошибку для агента вместо падения цикла выполнения."""

    return json.dumps(
        {
            "error": "ошибка_запроса_qdrant",
            "operation": operation,
            "detail": str(exc),
        },
        ensure_ascii=False,
    )


def _point_to_dict(point) -> Dict[str, Any]:
    return {"id": point.id, "score": getattr(point, "score", None), "payload": point.payload}


def search(collection: str, query: str, vector_name: str = "embedding", limit: int = 5) -> List[Dict[str, Any]]:
    vector = get_embedding(query)
    hits = _qdrant.query_points(
        collection_name=collection,
        query=vector,
        using=vector_name,
        limit=limit,
    ).points
    return [_point_to_dict(h) for h in hits]


def filter_points(collection: str, field: str, value: str, limit: int = 10) -> List[Dict[str, Any]]:
    results, _ = _qdrant.scroll(
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
    return [_point_to_dict(p) for p in results]


def scroll_points(
    collection: str,
    limit: int = 10,
    offset: Optional[str] = None,
    payload_fields: Optional[List[str]] = None,
    filter_field: Optional[str] = None,
    filter_value: Optional[str] = None,
) -> Dict[str, Any]:
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

    with_payload: bool | models.PayloadSelectorInclude
    if payload_fields:
        with_payload = models.PayloadSelectorInclude(include=payload_fields)
    else:
        with_payload = True

    points, next_offset = _qdrant.scroll(
        collection_name=collection,
        limit=limit,
        offset=offset,
        scroll_filter=scroll_filter,
        with_payload=with_payload,
    )
    return {
        "points": [_point_to_dict(p) for p in points],
        "next_offset": next_offset,
    }


def retrieve_by_id(collection: str, ids: List[str]) -> List[Dict[str, Any]]:
    points = _qdrant.retrieve(
        collection_name=collection,
        ids=ids,
    )
    return [_point_to_dict(p) for p in points]


def build_collection_schema(collection_name: str) -> Dict[str, Any]:
    """Вызвать ``get_collection`` и вернуть имя, статус, число точек,
    имена векторов и поля payload (из ``payload_schema``)."""
    info = _qdrant.get_collection(collection_name=collection_name)

    params = getattr(info.config, "params", None)
    vec = getattr(params, "vectors", None) if params else None
    if isinstance(vec, dict):
        vector_names = list(vec.keys())
    elif vec is not None:
        vector_names = ["embedding"]
    else:
        vector_names = []

    payload_fields: Dict[str, Any] = {}
    for key, psi in (getattr(info, "payload_schema", None) or {}).items():
        payload_fields[key] = {
            "data_type": str(psi.data_type) if getattr(psi, "data_type", None) is not None else None,
        }

    return {
        "collection_name": collection_name,
        "status": str(info.status) if info.status is not None else None,
        "points_count": info.points_count,
        "vector_names": vector_names,
        "payload_fields": payload_fields,
    }


# ---------------------------------------------------------------------------
# Обёртки BaseTool для агента
# ---------------------------------------------------------------------------


class QdrantCollectionSchema(BaseTool):
    """Метаданные коллекции: имя, статус, число точек, векторы, поля payload."""

    tool_name = "collection_schema"
    description = (
        "Вернуть имя коллекции, статус, число точек, имена векторов и поля payload "
        "(тип данных). "
        "Вызывай в первую очередь, чтобы узнать поля и векторы перед поиском и фильтрацией."
    )

    reasoning: str = Field(description="Зачем сейчас нужна схема этой коллекции")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)

    async def __call__(self, context, config, **_) -> str:
        try:
            result = build_collection_schema(self.collection)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant collection_schema завершился ошибкой: %s", exc)
            return _qdrant_failure_payload("collection_schema", exc)


class QdrantSearchTool(BaseTool):
    """Семантический поиск по коллекции Qdrant (сходство векторов)."""

    tool_name = "search"
    description = "Семантический поиск по коллекции Qdrant по текстовому запросу (векторное сходство)."

    reasoning: str = Field(description="Зачем нужен этот поиск")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    query: str = Field(description="Поисковая фраза на естественном языке")
    vector_name: str = Field(
        default="embedding",
        description=(
            "Имя вектора для поиска (из поля vector_names из схемы коллекции)."
        ),
    )
    limit: int = Field(default=5, description="Максимальное количество возвращаемых результатов")

    async def __call__(self, context, config, **_) -> str:
        try:
            result = search(
                collection=self.collection,
                query=self.query,
                vector_name=self.vector_name,
                limit=self.limit,
            )
            dto_name, dto_payload = register_dto(
                context,
                source=f"search_{self.collection}",
                data=result,
                summary_text=f"Поиск по {self.collection}: '{self.query[:60]}'",
                meta={"vector_name": self.vector_name, "limit": self.limit},
            )
            return json.dumps(dto_summary_view(dto_name, dto_payload), ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant search завершился ошибкой: %s", exc)
            return _qdrant_failure_payload("search", exc)


class QdrantFilterTool(BaseTool):
    """Точное совпадение значения поля payload в коллекции Qdrant."""

    tool_name = "filter_points"
    description = "Отобрать точки коллекции по точному совпадению значения поля payload."

    reasoning: str = Field(description="Зачем нужна эта фильтрация")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    field: str = Field(description='Имя поля payload для фильтра (например "question", "name")')
    value: str = Field(description="Ожидаемое точное значение поля")
    limit: int = Field(default=10, description="Максимальное количество возвращаемых результатов")

    async def __call__(self, context, config, **_) -> str:
        try:
            result = filter_points(
                collection=self.collection,
                field=self.field,
                value=self.value,
                limit=self.limit,
            )
            dto_name, dto_payload = register_dto(
                context,
                source=f"filter_{self.collection}",
                data=result,
                summary_text=f"Фильтр {self.collection}: {self.field} == {self.value}",
                meta={"field": self.field, "value": self.value, "limit": self.limit},
            )
            return json.dumps(dto_summary_view(dto_name, dto_payload), ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant filter_points завершился ошибкой: %s", exc)
            return _qdrant_failure_payload("filter_points", exc)


class QdrantScrollTool(BaseTool):
    """Постраничный обход точек коллекции с выбором полей и опциональным фильтром."""

    tool_name = "scroll_points"
    description = (
        "Постраничный обход точек коллекции Qdrant. "
        "Можно вернуть только указанные поля payload (payload_fields) "
        "и отфильтровать по точному совпадению поля (filter_field и filter_value)."
    )

    reasoning: str = Field(description="Зачем нужен постраничный обход")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    limit: int = Field(default=10, description="Размер страницы (число точек)")
    offset: Optional[str] = Field(
        default=None,
        description="Смещение для продолжения: значение next_offset из предыдущего вызова scroll",
    )
    payload_fields: List[str] = Field(
        default=[],
        description=(
            "Вернуть только эти поля payload для каждой точки. "
            "Пустой список — вернуть все поля. "
            "Пример: [\"region\", \"age_group\"]"
        ),
    )
    filter_field: Optional[str] = Field(
        default=None,
        description="Имя поля payload для фильтра (точное совпадение). Задавай вместе с filter_value.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Значение, которому должно равняться filter_field.",
    )

    async def __call__(self, context, config, **_) -> str:
        try:
            result = scroll_points(
                collection=self.collection,
                limit=self.limit,
                offset=self.offset,
                payload_fields=self.payload_fields or None,
                filter_field=self.filter_field,
                filter_value=self.filter_value,
            )
            dto_name, dto_payload = register_dto(
                context,
                source=f"scroll_{self.collection}",
                data=result.get("points", []),
                summary_text=f"Scroll {self.collection} (limit={self.limit})",
                meta={
                    "next_offset": result.get("next_offset"),
                    "filter_field": self.filter_field,
                    "filter_value": self.filter_value,
                    "payload_fields": self.payload_fields,
                },
            )
            response = dto_summary_view(dto_name, dto_payload)
            response["next_offset"] = result.get("next_offset")
            return json.dumps(response, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant scroll_points завершился ошибкой: %s", exc)
            return _qdrant_failure_payload("scroll_points", exc)


class QdrantRetrieveTool(BaseTool):
    """Получить точки коллекции по списку идентификаторов (строки UUID)."""

    tool_name = "retrieve_by_id"
    description = "Загрузить указанные точки из коллекции Qdrant по их идентификаторам (строки UUID)."

    reasoning: str = Field(description="Зачем нужны именно эти точки")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    ids: List[str] = Field(description="Список идентификаторов точек (UUID в виде строк) для загрузки")

    async def __call__(self, context, config, **_) -> str:
        try:
            result = retrieve_by_id(collection=self.collection, ids=self.ids)
            dto_name, dto_payload = register_dto(
                context,
                source=f"retrieve_{self.collection}",
                data=result,
                summary_text=f"retrieve_by_id {self.collection}: {len(self.ids)} id(s)",
                meta={"ids_count": len(self.ids)},
            )
            return json.dumps(dto_summary_view(dto_name, dto_payload), ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant retrieve_by_id завершился ошибкой: %s", exc)
            return _qdrant_failure_payload("retrieve_by_id", exc)
