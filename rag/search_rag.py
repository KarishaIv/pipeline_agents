"""
Семантический поиск по коллекции Qdrant 

Примеры:
  Локальная БД в папке qdrant_data:
    python3 rag/search_rag.py "льготная ипотека и ставки" --local --agent real_estate --top-k 8

  Окно по дате:
    python3 rag/search_rag.py "ипотека" --local --agent real_estate --window-days 30

  Qdrant на localhost:6333 (без --local):
    python3 rag/search_rag.py "курс доллара" --host localhost --port 6333 --agent currency

  Отключить бонус за свежесть:
    python3 rag/search_rag.py "новости рынка" --local --no-prefer-recent
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from datetime import datetime, timezone, timedelta
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import DatetimeRange, Filter, FieldCondition, MatchValue
from rag.e5_embeddings import get_query_embedding


EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def _parse_iso_to_ts(val) -> Optional[int]:
    if not val:
        return None
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def search(
    query: str,
    collection_name: str = "telegram_news_e5_large",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    local: bool = False,
    storage_path: str = "qdrant_data",
    top_k: int = 5,
    agent: Optional[str] = None,
    window_days: int = 14,
    prefer_recent: bool = True,
    recency_weight: float = 0.15,
):
    q_vec = get_query_embedding(query)

    if local:
        client = QdrantClient(path=storage_path)
    else:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)

    must_conditions = []
    if agent:
        must_conditions.append(FieldCondition(key="agent", match=MatchValue(value=agent)))
    if window_days > 0:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=window_days)
        must_conditions.append(FieldCondition(key="date", range=DatetimeRange(gte=cutoff_dt)))

    query_filter = Filter(must=must_conditions) if must_conditions else None

    candidate_k = top_k * 5 if prefer_recent else top_k
    try:
        response = client.query_points(
            collection_name=collection_name,
            query=q_vec,
            limit=candidate_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
    except ResponseHandlingException as e:
        if "Connection refused" in str(e) or "refused" in str(e).lower():
            print("Ошибка: Qdrant не запущен. Используйте --local, если загружали через ingest_rag.py --local")
        raise

    out = []
    for hit in response.points:
        meta = {k: v for k, v in hit.payload.items() if k != "text"}
        out.append({
            "id": hit.id,
            "text": hit.payload.get("text", ""),
            "metadata": meta,
            "score": hit.score,
        })

    # Свежесть
    if prefer_recent and out:
        for r in out:
            r["_date_unix"] = _parse_iso_to_ts(r["metadata"].get("date"))

        unix_vals = [r["_date_unix"] for r in out if r["_date_unix"] is not None]
        min_u = min(unix_vals) if unix_vals else None
        max_u = max(unix_vals) if unix_vals else None

        def recency_norm(v: Optional[int]) -> float:
            if v is None or min_u is None or max_u is None or max_u == min_u:
                return 0.0
            return (v - min_u) / (max_u - min_u)

        for r in out:
            r["_final_score"] = float(r["score"]) + recency_weight * recency_norm(r.get("_date_unix"))
        out.sort(key=lambda x: x["_final_score"], reverse=True)
        out = out[:top_k]
        for r in out:
            r.pop("_final_score", None)
            r.pop("_date_unix", None)
    return out[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Поиск по RAG (Qdrant)")
    parser.add_argument("query", nargs="+", help="Текст запроса в кавычках, например: \"курс доллара\"")
    parser.add_argument("--collection", default="telegram_news_e5_large")
    parser.add_argument("--local", action="store_true", help="Локальное хранилище (как при ingest_rag.py --local)")
    parser.add_argument("--storage-path", default="qdrant_data", help="Папка при --local")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--agent", default=None, help="Фильтр: macroeconomy, banks, currency, real_estate, social_news")
    parser.add_argument("--window-days", type=int, default=14, help="Искать только в окне последних N дней (0 = без фильтра)")
    parser.add_argument(
        "--prefer-recent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Бонус за свежесть при ранжировании (по умолчанию включён). Отключить: --no-prefer-recent",
    )
    parser.add_argument(
        "--recency-weight",
        type=float,
        default=0.15,
        help="Вес свежести при ранжировании (по умолчанию 0.15)",
    )
    args = parser.parse_args()

    q = " ".join(args.query)
    print("Поиск...")
    results = search(
        q,
        collection_name=args.collection,
        qdrant_host=args.host,
        qdrant_port=args.port,
        local=args.local,
        storage_path=args.storage_path,
        top_k=args.top_k,
        agent=args.agent,
        window_days=args.window_days,
        prefer_recent=args.prefer_recent,
        recency_weight=args.recency_weight,
    )
    for i, r in enumerate(results, 1):
        print(f"\n--- {i} (score={r['score']:.4f}) [{r['metadata'].get('agent', '')}] ---")
        print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))


if __name__ == "__main__":
    main()