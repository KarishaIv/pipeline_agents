import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from src.utils import get_embedding

AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations"]
COLLECTION_ENUM_DESC = "Collection name: " + ", ".join(AVAILABLE_COLLECTIONS)

_qdrant = QdrantClient(host="localhost", port=6333)


# ---------------------------------------------------------------------------
# Pydantic argument schemas
# ---------------------------------------------------------------------------

class SearchArgs(BaseModel):
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


class FilterArgs(BaseModel):
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    field: str = Field(description='Payload field name to filter on (e.g. "question", "name")')
    value: str = Field(description="Expected exact value of the field")
    limit: int = Field(default=10, description="Maximum number of results")


class ScrollArgs(BaseModel):
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    limit: int = Field(default=10, description="Page size")
    offset: Optional[int] = Field(default=None, description="Point id to start from (from previous next_offset)")


class RetrieveArgs(BaseModel):
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    ids: List[int] = Field(description="List of integer point IDs to retrieve")


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Semantic search over a Qdrant collection. "
                "Embeds the query and returns the closest points by cosine similarity. "
                f"Available collections: {', '.join(AVAILABLE_COLLECTIONS)}."
            ),
            "parameters": SearchArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_points",
            "description": (
                "Filter points in a Qdrant collection by an exact payload field match. "
                f"Available collections: {', '.join(AVAILABLE_COLLECTIONS)}."
            ),
            "parameters": FilterArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_points",
            "description": (
                "Paginated scroll through all points in a Qdrant collection. "
                "Returns a page of points and the next page offset. "
                f"Available collections: {', '.join(AVAILABLE_COLLECTIONS)}."
            ),
            "parameters": ScrollArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_by_id",
            "description": (
                "Retrieve specific points from a Qdrant collection by their integer IDs. "
                f"Available collections: {', '.join(AVAILABLE_COLLECTIONS)}."
            ),
            "parameters": RetrieveArgs.model_json_schema(),
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_to_dict(point) -> Dict[str, Any]:
    return {"id": point.id, "score": getattr(point, "score", None), "payload": point.payload}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search(collection: str, query: str, vector_name: str = "embedding", limit: int = 5) -> List[Dict[str, Any]]:
    vector = get_embedding(query)
    hits = _qdrant.search(
        collection_name=collection,
        query_vector=models.NamedVector(name=vector_name, vector=vector),
        limit=limit,
    )
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
# Dispatcher
# ---------------------------------------------------------------------------

_REGISTRY = {
    "search": (SearchArgs, search),
    "filter_points": (FilterArgs, filter_points),
    "scroll_points": (ScrollArgs, scroll_points),
    "retrieve_by_id": (RetrieveArgs, retrieve_by_id),
}


def execute_tool(name: str, raw_args: dict) -> str:
    """Validate args with the corresponding Pydantic model and call the tool."""
    if name not in _REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    schema, fn = _REGISTRY[name]
    try:
        args = schema.model_validate(raw_args)
        result = fn(**args.model_dump())
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
