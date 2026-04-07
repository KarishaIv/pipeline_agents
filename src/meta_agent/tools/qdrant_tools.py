"""Qdrant client functions and agent BaseTool wrappers."""

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field
from qdrant_client import QdrantClient, models

from sgr_agent_core.base_tool import BaseTool

from src.utils import get_embedding

logger = logging.getLogger("meta_agent.qdrant")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations"]
COLLECTION_ENUM_DESC = "Collection name: " + ", ".join(AVAILABLE_COLLECTIONS)

CollectionName = Literal["questions", "personas", "target_audiences", "simulations"]

# ---------------------------------------------------------------------------
# Low-level Qdrant client
# ---------------------------------------------------------------------------

_qdrant = QdrantClient(host="localhost", port=6333)


def _qdrant_failure_payload(operation: str, exc: BaseException) -> str:
    """Return JSON the agent can read instead of crashing the agent loop."""

    return json.dumps(
        {"error": "qdrant_request_failed", "operation": operation, "detail": str(exc)},
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
# Agent BaseTool wrappers
# ---------------------------------------------------------------------------


class QdrantCollectionSchema(BaseTool):
    """Return collection metadata: name, status, points_count, vector_names, and payload fields."""

    tool_name = "collection_schema"
    description = (
        "Return collection_name, status, points_count, vector_names, and payload_fields "
        "(field data_type and indexed points count) via Qdrant get_collection API. "
        "Call first to discover available fields and vectors before search/filter."
    )

    reasoning: str = Field(description="Why you need this collection's schema now")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)

    async def __call__(self, context, config, **_) -> str:
        try:
            result = build_collection_schema(self.collection)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant collection_schema failed: %s", exc)
            return _qdrant_failure_payload("collection_schema", exc)


class QdrantSearchTool(BaseTool):
    """Semantic search over a Qdrant collection via cosine similarity."""

    tool_name = "search"
    description = "Semantic search over a Qdrant collection via cosine similarity."

    reasoning: str = Field(description="Why this search is needed")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    query: str = Field(description="Natural-language search string")
    vector_name: str = Field(
        default="embedding",
        description=(
            "Named vector to search against. "
            "questions/personas/target_audiences use 'embedding'. "
            "simulations uses: emotional_vector, rational_vector, "
            "social_vector, ideological_vector, decision_vector, general_vector"
        ),
    )
    limit: int = Field(default=5, description="Maximum number of results")

    async def __call__(self, context, config, **_) -> str:
        try:
            result = search(
                collection=self.collection,
                query=self.query,
                vector_name=self.vector_name,
                limit=self.limit,
            )
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant search failed: %s", exc)
            return _qdrant_failure_payload("search", exc)


class QdrantFilterTool(BaseTool):
    """Filter a Qdrant collection by an exact payload field match."""

    tool_name = "filter_points"
    description = "Filter a Qdrant collection by an exact payload field match."

    reasoning: str = Field(description="Why this filter is needed")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    field: str = Field(description='Payload field name to filter on (e.g. "question", "name")')
    value: str = Field(description="Expected exact value of the field")
    limit: int = Field(default=10, description="Maximum number of results")

    async def __call__(self, context, config, **_) -> str:
        try:
            result = filter_points(
                collection=self.collection,
                field=self.field,
                value=self.value,
                limit=self.limit,
            )
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant filter_points failed: %s", exc)
            return _qdrant_failure_payload("filter_points", exc)


class QdrantScrollTool(BaseTool):
    """Paginated scroll through all points in a Qdrant collection with optional field selection and filtering."""

    tool_name = "scroll_points"
    description = (
        "Paginated scroll through all points in a Qdrant collection. "
        "Supports returning only specific payload fields (payload_fields) "
        "and filtering by an exact payload field value (filter_field + filter_value)."
    )

    reasoning: str = Field(description="Why paginated scroll is needed")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    limit: int = Field(default=10, description="Page size")
    offset: Optional[str] = Field(
        default=None,
        description="UUID string to start from (value of next_offset returned by the previous scroll call)",
    )
    payload_fields: List[str] = Field(
        default=[],
        description=(
            "Return only these payload fields per point. "
            "Leave empty to return all fields. "
            "Example: [\"region\", \"age_group\"]"
        ),
    )
    filter_field: Optional[str] = Field(
        default=None,
        description="Payload field name to filter on (exact match). Must be set together with filter_value.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Exact value the filter_field must equal.",
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
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant scroll_points failed: %s", exc)
            return _qdrant_failure_payload("scroll_points", exc)


class QdrantRetrieveTool(BaseTool):
    """Retrieve specific points from a Qdrant collection by their UUID string IDs."""

    tool_name = "retrieve_by_id"
    description = "Retrieve specific points from a Qdrant collection by their UUID string IDs."

    reasoning: str = Field(description="Why these specific points are needed")
    collection: CollectionName = Field(description=COLLECTION_ENUM_DESC)
    ids: List[str] = Field(description="List of UUID string point IDs to retrieve")

    async def __call__(self, context, config, **_) -> str:
        try:
            result = retrieve_by_id(collection=self.collection, ids=self.ids)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Qdrant retrieve_by_id failed: %s", exc)
            return _qdrant_failure_payload("retrieve_by_id", exc)
