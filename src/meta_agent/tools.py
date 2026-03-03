import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models

from src.utils import get_embedding

COLLECTION_NAME = "questions"

_qdrant = QdrantClient(url="http://localhost:6333")


# ---------------------------------------------------------------------------
# Pydantic argument schemas
# ---------------------------------------------------------------------------

class SearchArgs(BaseModel):
    query: str = Field(description="Natural-language search string")
    limit: int = Field(default=5, description="Maximum number of results to return")


class FilterArgs(BaseModel):
    field: str = Field(description='Payload field name to filter on (e.g. "question")')
    value: str = Field(description="Expected exact value of the field")
    limit: int = Field(default=10, description="Maximum number of results to return")


class ScrollArgs(BaseModel):
    limit: int = Field(default=10, description="Page size")
    offset: Optional[int] = Field(default=None, description="Point id to start from (from previous next_offset)")


class RetrieveArgs(BaseModel):
    ids: List[int] = Field(description="List of integer point IDs to retrieve")


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Semantic search over the questions collection. Embeds the query and returns the closest points by cosine similarity.",
            "parameters": SearchArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_points",
            "description": "Filter points in the questions collection by an exact payload field match.",
            "parameters": FilterArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_points",
            "description": "Paginated scroll through all points in the questions collection. Returns a page of points and the next page offset.",
            "parameters": ScrollArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_by_id",
            "description": "Retrieve specific points from the questions collection by their integer IDs.",
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

def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    vector = get_embedding(query)
    hits = _qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
    )
    return [_point_to_dict(h) for h in hits]


def filter_points(field: str, value: str, limit: int = 10) -> List[Dict[str, Any]]:
    results, _ = _qdrant.scroll(
        collection_name=COLLECTION_NAME,
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


def scroll_points(limit: int = 10, offset: Optional[int] = None) -> Dict[str, Any]:
    points, next_offset = _qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        offset=offset,
    )
    return {
        "points": [_point_to_dict(p) for p in points],
        "next_offset": next_offset,
    }


def retrieve_by_id(ids: List[int]) -> List[Dict[str, Any]]:
    points = _qdrant.retrieve(
        collection_name=COLLECTION_NAME,
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
