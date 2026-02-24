from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from qdrant_client import QdrantClient, models

from src.utils import get_embedding

COLLECTION_NAME = "questions"

_qdrant = QdrantClient(url="http://localhost:6333")


def _point_to_dict(point: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": point["id"], "score": point.get("score"), "payload": point["payload"]}


@tool
def search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Semantic search over the questions collection.

    Embeds the query text and returns the closest points by cosine similarity.

    Args:
        query: Natural-language search string.
        limit: Maximum number of results (default 5).
    """
    vector = get_embedding(query)
    hits = _qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
    )
    return [_point_to_dict(h) for h in hits]


@tool
def filter_points(
    field: str,
    value: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Filter points in the questions collection by an exact payload field match.

    Args:
        field: Payload field name to filter on (e.g. "question").
        value: Expected value of the field.
        limit: Maximum number of results (default 10).
    """
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


@tool
def scroll_points(
    limit: int = 10,
    offset: Optional[int] = None,
) -> Dict[str, Any]:
    """Paginated scroll through all points in the questions collection.

    Returns a page of points and the next page offset.

    Args:
        limit: Page size (default 10).
        offset: Point id to start from (use the value returned as next_offset).
    """
    points, next_offset = _qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        offset=offset,
    )
    return {
        "points": [_point_to_dict(p) for p in points],
        "next_offset": next_offset,
    }


@tool
def retrieve_by_id(ids: List[int]) -> List[Dict[str, Any]]:
    """Retrieve specific points from the questions collection by their IDs.

    Args:
        ids: List of integer point IDs to retrieve.
    """
    points = _qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=ids,
    )
    return [_point_to_dict(p) for p in points]


qdrant_tools = [search, filter_points, scroll_points, retrieve_by_id]
