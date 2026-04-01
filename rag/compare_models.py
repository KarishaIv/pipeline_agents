"""
Сравнение качества поиска для 3 эмбеддинг-моделей (small/base/large),
когда для каждой модели своя коллекция Qdrant
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TRIPLES = [
    ("intfloat/multilingual-e5-small", "telegram_news_e5_small"),
    ("intfloat/multilingual-e5-base", "telegram_news_e5_base"),
    ("intfloat/multilingual-e5-large", "telegram_news_e5_large"),
]

DEFAULT_QUERIES = [
    "ключевая ставка инфляция прогноз",
    "бюджет расходы доходы нефть газ",
    "дивиденды сбер и другие банки",
    "индекс мосбиржи падение рост причины",
    "курс доллара евро рубля прогноз",
    "льготная ипотека ставки новостройки",
]


def _run_search_subprocess(
    *,
    model_name: str,
    collection: str,
    query: str,
    local: bool,
    storage_path: str,
    host: str,
    port: int,
    top_k: int,
    agent: str | None,
    window_days: int,
    prefer_recent: bool,
    recency_weight: float,
    timeout_s: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], float]:
    code = r"""
import json, os, sys
from rag.search_rag import search

query = os.environ["CMP_QUERY"]
collection = os.environ["CMP_COLLECTION"]
local = os.environ.get("CMP_LOCAL","0") == "1"
storage_path = os.environ.get("CMP_STORAGE_PATH","qdrant_data")
host = os.environ.get("CMP_HOST","localhost")
port = int(os.environ.get("CMP_PORT","6333"))
top_k = int(os.environ.get("CMP_TOP_K","5"))
agent = os.environ.get("CMP_AGENT") or None
window_days = int(os.environ.get("CMP_WINDOW_DAYS","14"))
prefer_recent = os.environ.get("CMP_PREFER_RECENT","1") == "1"
recency_weight = float(os.environ.get("CMP_RECENCY_WEIGHT","0.15"))

out = search(
    query=query,
    collection_name=collection,
    qdrant_host=host,
    qdrant_port=port,
    local=local,
    storage_path=storage_path,
    top_k=top_k,
    agent=agent,
    window_days=window_days,
    prefer_recent=prefer_recent,
    recency_weight=recency_weight,
)
print(json.dumps(out, ensure_ascii=False))
"""
    env = os.environ.copy()
    env["RAG_EMBEDDING_MODEL"] = model_name
    env["CMP_QUERY"] = query
    env["CMP_COLLECTION"] = collection
    env["CMP_LOCAL"] = "1" if local else "0"
    env["CMP_STORAGE_PATH"] = storage_path
    env["CMP_HOST"] = host
    env["CMP_PORT"] = str(port)
    env["CMP_TOP_K"] = str(top_k)
    env["CMP_AGENT"] = agent or ""
    env["CMP_WINDOW_DAYS"] = str(window_days)
    env["CMP_PREFER_RECENT"] = "1" if prefer_recent else "0"
    env["CMP_RECENCY_WEIGHT"] = str(recency_weight)

    if verbose:
        print(f"[run] model={model_name} collection={collection}", flush=True)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "search subprocess timed out\n"
            f"model={model_name} collection={collection}\n"
            f"timeout_s={timeout_s}\n"
        ) from e
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        raise RuntimeError(
            "search subprocess failed\n"
            f"model={model_name} collection={collection}\n"
            f"stderr:\n{p.stderr}"
        )
    try:
        return json.loads(p.stdout), elapsed_ms
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from subprocess stdout: {p.stdout[:500]}") from e


def main() -> None:
    ap = argparse.ArgumentParser(description="Сравнение 3 коллекций (small/base/large) по одинаковым запросам")
    ap.add_argument("--local", action="store_true", help="Искать в локальном Qdrant (qdrant_data)")
    ap.add_argument("--storage-path", default="qdrant_data", help="Папка локального Qdrant (если --local)")
    ap.add_argument("--host", default="localhost", help="Qdrant host (если не --local)")
    ap.add_argument("--port", type=int, default=6333, help="Qdrant port (если не --local)")
    ap.add_argument("--top-k", type=int, default=8, help="Сколько документов выводить")
    ap.add_argument("--agent", default=None, help="Фильтр по агенту (macroeconomy/banks/currency/real_estate/social_news)")
    ap.add_argument("--window-days", type=int, default=30, help="Окно по дате (0 = без фильтра)")
    ap.add_argument("--no-prefer-recent", action="store_true", help="Отключить бонус за свежесть")
    ap.add_argument("--recency-weight", type=float, default=0.15, help="Вес свежести (если prefer_recent)")
    ap.add_argument("--queries-file", default=None, help="Путь к txt файлу: по 1 запросу на строку")
    ap.add_argument("--out-dir", default="outputs", help="Папка для результатов")
    ap.add_argument("--timeout-s", type=int, default=600, help="Таймаут одного поиска (сек)")
    ap.add_argument("--verbose", action="store_true", help="Подробные логи прогресса")
    args = ap.parse_args()

    if args.queries_file:
        q_path = Path(args.queries_file)
        queries = [ln.strip() for ln in q_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        queries = DEFAULT_QUERIES

    triples = DEFAULT_TRIPLES
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"compare_models_{ts}.json"
    md_path = out_dir / f"compare_models_{ts}.md"

    prefer_recent = not args.no_prefer_recent

    results: dict[str, Any] = {
        "created_at": ts,
        "qdrant": {
            "local": bool(args.local),
            "storage_path": args.storage_path,
            "host": args.host,
            "port": args.port,
        },
        "search_params": {
            "top_k": args.top_k,
            "agent": args.agent,
            "window_days": args.window_days,
            "prefer_recent": prefer_recent,
            "recency_weight": args.recency_weight,
        },
        "queries": queries,
        "models": [{"model": m, "collection": c} for (m, c) in triples],
        "runs": [],
    }

    md_lines: list[str] = []
    md_lines.append("## Compare models (small/base/large)\n")
    md_lines.append(f"- Queries: {len(queries)}\n")
    md_lines.append(f"- top_k: {args.top_k}, agent: {args.agent or 'ALL'}, window_days: {args.window_days}\n")
    md_lines.append(f"- prefer_recent: {prefer_recent}, recency_weight: {args.recency_weight}\n")
    md_lines.append("\n")

    for q in queries:
        entry: dict[str, Any] = {"query": q, "by_model": []}
        if args.verbose:
            print(f"\n[query] {q}", flush=True)
        md_lines.append(f"### Query: {q}\n")
        for model_name, collection in triples:
            if args.verbose:
                print(f"  -> {model_name} / {collection} ...", flush=True)
            out, elapsed_ms = _run_search_subprocess(
                model_name=model_name,
                collection=collection,
                query=q,
                local=bool(args.local),
                storage_path=args.storage_path,
                host=args.host,
                port=args.port,
                top_k=args.top_k,
                agent=args.agent,
                window_days=args.window_days,
                prefer_recent=prefer_recent,
                recency_weight=args.recency_weight,
                timeout_s=args.timeout_s,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"     done in {elapsed_ms:.0f} ms, hits={len(out)}", flush=True)
            entry["by_model"].append(
                {"model": model_name, "collection": collection, "elapsed_ms": elapsed_ms, "results": out}
            )

            md_lines.append(f"#### {model_name} (`{collection}`) — {elapsed_ms:.0f} ms\n")
            if not out:
                md_lines.append("- (no results)\n\n")
                continue
            for i, r in enumerate(out, 1):
                text = (r.get("text") or "").replace("\n", " ").strip()
                text = text[:240] + ("…" if len(text) > 240 else "")
                md_lines.append(
                    f"- {i}. score={r.get('score'):.4f} | {r.get('metadata', {}).get('agent','')} | {text}\n"
                )
            md_lines.append("\n")

        results["runs"].append(entry)
        md_lines.append("\n")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("".join(md_lines), encoding="utf-8")

    print(f"Готово: {json_path}")
    print(f"Кратко: {md_path}")


if __name__ == "__main__":
    main()

