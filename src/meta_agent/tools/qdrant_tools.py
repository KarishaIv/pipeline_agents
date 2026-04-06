"""Qdrant client functions and IronAgent BaseTool wrappers."""

import json
from typing import Any, Dict, List, Optional

from pydantic import Field
from qdrant_client import QdrantClient, models

from sgr_agent_core.base_tool import BaseTool

from src.utils import get_embedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations"]
COLLECTION_ENUM_DESC = "Collection name: " + ", ".join(AVAILABLE_COLLECTIONS)

# ---------------------------------------------------------------------------
# Low-level Qdrant client
# ---------------------------------------------------------------------------

_qdrant = QdrantClient(host="localhost", port=6333)


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


def scroll_points(collection: str, limit: int = 10, offset: Optional[int] = None) -> Dict[str, Any]:
    points, next_offset = _qdrant.scroll(
        collection_name=collection,
        limit=limit,
        offset=offset,
    )
    return {
        "points": [_point_to_dict(p) for p in points],
        "next_offset": next_offset,
    }


def retrieve_by_id(collection: str, ids: List[int]) -> List[Dict[str, Any]]:
    points = _qdrant.retrieve(
        collection_name=collection,
        ids=ids,
    )
    return [_point_to_dict(p) for p in points]


# ---------------------------------------------------------------------------
# IronAgent BaseTool wrappers
# ---------------------------------------------------------------------------

class QdrantSearchTool(BaseTool):
    """Semantic search over a Qdrant collection via cosine similarity."""

    tool_name = "search"
    description = "Semantic search over a Qdrant collection via cosine similarity."

    reasoning: str = Field(description="Why this search is needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
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
        result = search(
            collection=self.collection,
            query=self.query,
            vector_name=self.vector_name,
            limit=self.limit,
        )
        return json.dumps(result, ensure_ascii=False, default=str)


class QdrantFilterTool(BaseTool):
    """Filter a Qdrant collection by an exact payload field match."""

    tool_name = "filter_points"
    description = "Filter a Qdrant collection by an exact payload field match."

    reasoning: str = Field(description="Why this filter is needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    field: str = Field(description='Payload field name to filter on (e.g. "question", "name")')
    value: str = Field(description="Expected exact value of the field")
    limit: int = Field(default=10, description="Maximum number of results")

    async def __call__(self, context, config, **_) -> str:
        result = filter_points(
            collection=self.collection,
            field=self.field,
            value=self.value,
            limit=self.limit,
        )
        return json.dumps(result, ensure_ascii=False, default=str)


class QdrantScrollTool(BaseTool):
    """Paginated scroll through all points in a Qdrant collection."""

    tool_name = "scroll_points"
    description = "Paginated scroll through all points in a Qdrant collection."

    reasoning: str = Field(description="Why paginated scroll is needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    limit: int = Field(default=10, description="Page size")
    offset: Optional[int] = Field(
        default=None,
        description="Point id to start from (from previous next_offset)",
    )

    async def __call__(self, context, config, **_) -> str:
        result = scroll_points(
            collection=self.collection,
            limit=self.limit,
            offset=self.offset,
        )
        return json.dumps(result, ensure_ascii=False, default=str)


class QdrantRetrieveTool(BaseTool):
    """Retrieve specific points from a Qdrant collection by their integer IDs."""

    tool_name = "retrieve_by_id"
    description = "Retrieve specific points from a Qdrant collection by their integer IDs."

    reasoning: str = Field(description="Why these specific points are needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    ids: List[int] = Field(description="List of integer point IDs to retrieve")

    async def __call__(self, context, config, **_) -> str:
        result = retrieve_by_id(collection=self.collection, ids=self.ids)
        return json.dumps(result, ensure_ascii=False, default=str)
