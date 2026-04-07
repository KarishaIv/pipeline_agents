"""Инструменты для работы с Qdrant.

QdrantService — singleton.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Optional

from pydantic import Field
from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

from src.meta_agent.configs import CollectionName, COLLECTION_ENUM_DESC
from src.meta_agent.services.qdrant import QdrantService
from src.meta_agent.tools.dto_tools import dto_summary_view, register_dto
from src.meta_agent.utils.json_responses import json_error, serialize_tool_result

logger = logging.getLogger("meta_agent.qdrant")


def failure_payload(operation: str, exc: Exception | str | None = None) -> str:
    """Формирует JSON-ошибку для возврата из qdrant tools.

    Все ошибки (включая из QdrantService) обрабатываются здесь
    и возвращаются в стандартизированном формате JSON
    с полями error, operation и detail.
    
    Wrapper for backward compatibility; use json_error directly in new code.
    """
    if isinstance(exc, str) or exc is None:
        exc = RuntimeError(str(exc) if exc else "unknown qdrant error")
    elif not isinstance(exc, Exception):
        exc = RuntimeError(str(exc))
    return json_error(
        f"Ошибка запроса к Qdrant: {operation}",
        error_type="qdrant_error",
        details={"operation": operation, "detail": str(exc)},
    )


def get_qdrant_service() -> QdrantService:
    """Return the singleton QdrantService instance."""
    return QdrantService()


def build_collection_schema(collection_name: str) -> Dict[str, Any]:
    """Call ``get_collection`` and return name, status, points_count,
    vector_names, and payload_fields (from ``payload_schema``)."""
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
# Tools
# ---------------------------------------------------------------------------


class QdrantCollectionSchema(BaseTool):
    """Метаданные коллекции: имя, статус, число точек, векторы, поля payload."""

    tool_name = "collection_schema"
    description = (
        "Вернуть имя коллекции, статус, число точек, имена векторов и поля payload "
        "(тип данных). Вызывай в первую очередь, чтобы узнать поля и векторы перед поиском."
    )

    reasoning: str = Field(description="Зачем сейчас нужна схема этой коллекции")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            qdrant_service = get_qdrant_service()
            result = qdrant_service.get_collection_schema(self.collection)
            return serialize_tool_result(result)
        except Exception as exc:
            logger.warning("Qdrant collection_schema завершился ошибкой: %s", exc)
            return failure_payload("collection_schema", exc)


class QdrantSearchTool(BaseTool):
    """Семантический поиск по коллекции Qdrant (векторное сходство)."""

    tool_name = "search"
    description = "Семантический поиск по коллекции Qdrant по текстовому запросу (векторное сходство)."

    reasoning: str = Field(description="Зачем нужен этот поиск")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    query: str = Field(description="Поисковая фраза на естественном языке")
    vector_name: str = Field(
        default="embedding",
        description="Имя вектора для поиска (из схемы коллекции).",
    )
    limit: int = Field(default=5, description="Максимальное количество возвращаемых результатов")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            qdrant_service = get_qdrant_service()
            result = qdrant_service.search(
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
            return serialize_tool_result(dto_summary_view(dto_name, dto_payload))
        except Exception as exc:
            logger.warning("Qdrant search завершился ошибкой: %s", exc)
            return failure_payload("search", exc)


class QdrantFilterTool(BaseTool):
    """Отобрать точки коллекции по точному совпадению значения поля payload."""

    tool_name = "filter_points"
    description = "Отобрать точки коллекции по точному совпадению значения поля payload."

    reasoning: str = Field(description="Зачем нужна эта фильтрация")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    field: str = Field(description='Имя поля payload для фильтра (например "question", "name")')
    value: str = Field(description="Ожидаемое точное значение поля")
    limit: int = Field(default=10, description="Максимальное количество возвращаемых результатов")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            qdrant_service = get_qdrant_service()
            result = qdrant_service.filter_points(
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
            return serialize_tool_result(dto_summary_view(dto_name, dto_payload))
        except Exception as exc:
            logger.warning("Qdrant filter_points завершился ошибкой: %s", exc)
            return failure_payload("filter_points", exc)


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
    limit: int = Field(default=10, description="Размер страницы")
    offset: Optional[str] = Field(default=None, description="Смещение из предыдущего scroll")
    payload_fields: List[str] = Field(
        default_factory=list,
        description="Список полей payload для возврата (пустой — все поля)",
    )
    filter_field: Optional[str] = Field(
        default=None,
        description="Имя поля payload для фильтра (точное совпадение). Задавай вместе с filter_value.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Значение, которому должно равняться значение из поля filter_field.",
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            qdrant_service = get_qdrant_service()
            result = qdrant_service.scroll_points(
                collection=self.collection,
                limit=self.limit,
                offset=self.offset,
                payload_fields=self.payload_fields or None,
                filter_field=self.filter_field,
                filter_value=self.filter_value,
            )
            points = result.get("points", []) if isinstance(result, dict) else []
            dto_name, dto_payload = register_dto(
                context,
                source=f"scroll_{self.collection}",
                data=points,
                summary_text=f"Scroll {self.collection} (limit={self.limit})",
                meta={"limit": self.limit, "offset": self.offset},
            )
            return serialize_tool_result(dto_summary_view(dto_name, dto_payload))
        except Exception as exc:
            logger.warning("Qdrant scroll_points завершился ошибкой: %s", exc)
            return failure_payload("scroll_points", exc)


class QdrantRetrieveTool(BaseTool):
    """Получить точки коллекции по списку идентификаторов (строки UUID)."""

    tool_name = "retrieve_by_id"
    description = "Получить указанные точки из коллекции Qdrant по их идентификаторам (строки UUID)."

    reasoning: str = Field(description="Зачем нужно получить именно эти точки")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    ids: List[str] = Field(description="Список ID точек для извлечения (строки UUID)")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            qdrant_service = get_qdrant_service()

            if not self.ids:
                return json_error(
                    "Список ID пуст",
                    error_type="validation_error",
                    details={"issue": "ids_empty"},
                )

            result = qdrant_service.retrieve_by_id(self.collection, self.ids)
            dto_name, dto_payload = register_dto(
                context,
                source=f"retrieve_{self.collection}",
                data=result,
                summary_text=f"retrieve_by_id {self.collection}: {len(self.ids)} id(s)",
                meta={"ids_count": len(self.ids)},
            )
            return serialize_tool_result(dto_summary_view(dto_name, dto_payload))
        except Exception as exc:
            logger.warning("Qdrant retrieve_by_id завершился ошибкой: %s", exc)
            return failure_payload("retrieve_by_id", exc)
