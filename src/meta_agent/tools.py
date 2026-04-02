import json
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from src.utils import get_embedding

AVAILABLE_COLLECTIONS = ["questions", "personas", "target_audiences", "simulations"]
COLLECTION_ENUM_DESC = "Collection name: " + ", ".join(AVAILABLE_COLLECTIONS)

_qdrant = QdrantClient(host="localhost", port=6333)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_to_dict(point) -> Dict[str, Any]:
    return {"id": point.id, "score": getattr(point, "score", None), "payload": point.payload}


# ---------------------------------------------------------------------------
# Qdrant query functions
# ---------------------------------------------------------------------------

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
