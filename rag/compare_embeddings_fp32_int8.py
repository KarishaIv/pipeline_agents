"""
Сравнение эмбеддингов ONNX FP32 vs ONNX INT8 по cosine similarity
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_MODEL = "intfloat/multilingual-e5-large"


def _load_texts_from_parquet(parquet_path: str, n: int, seed: int) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["text"])
    texts = [t for t in df["text"].astype(str).tolist() if t and t.strip()]
    if not texts:
        raise RuntimeError(f"No texts found in {parquet_path}")
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[: max(1, min(n, len(texts)))]


def _load_texts_from_file(path: str, n: int) -> list[str]:
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    out = [ln for ln in lines if ln]
    return out[:n]


def _run_embed_subprocess(texts: list[str], *, model: str, use_int8: bool, max_length: int, batch_size: int) -> tuple[list[list[float]], float]:
    code = r"""
import json, os, time
from rag.e5_embeddings import get_document_embeddings

texts = json.loads(os.environ["CE_TEXTS"])
model = os.environ["RAG_EMBEDDING_MODEL"]
max_length = os.environ.get("RAG_EMBEDDING_MAX_LENGTH", "512")

t0 = time.perf_counter()
emb = get_document_embeddings(texts)
elapsed_ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({"emb": emb, "ms": elapsed_ms}, ensure_ascii=False))
"""
    env = os.environ.copy()
    env["RAG_EMBEDDING_MODEL"] = model
    env["RAG_EMBEDDING_BACKEND"] = "onnx"
    env["RAG_ONNX_USE_INT8"] = "1" if use_int8 else "0"
    env["RAG_EMBEDDING_MAX_LENGTH"] = str(max_length)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CE_TEXTS"] = json.dumps(texts, ensure_ascii=False)

    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1200,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        raise RuntimeError(f"embed subprocess failed (int8={use_int8})\nstderr:\n{p.stderr}")
    payload = json.loads(p.stdout)
    return payload["emb"], float(payload.get("ms", wall_ms))


def _cosine(u: list[float], v: list[float]) -> float:
    import math

    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += float(a) * float(b)
        nu += float(a) * float(a)
        nv += float(b) * float(b)
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def _p(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * q))
    return ys[max(0, min(len(ys) - 1, idx))]


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare ONNX FP32 vs ONNX INT8 embeddings by cosine similarity")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--parquet", default="rag_docs.parquet")
    ap.add_argument("--texts-file", default=None, help="TXT: по 1 тексту на строку")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)

    if args.texts_file:
        texts = _load_texts_from_file(args.texts_file, args.n)
    else:
        texts = _load_texts_from_parquet(args.parquet, args.n, args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"compare_embeddings_fp32_int8_{ts}.json"
    md_path = out_dir / f"compare_embeddings_fp32_int8_{ts}.md"

    emb_fp32, ms_fp32 = _run_embed_subprocess(
        texts, model=args.model, use_int8=False, max_length=args.max_length, batch_size=args.batch_size
    )
    emb_int8, ms_int8 = _run_embed_subprocess(
        texts, model=args.model, use_int8=True, max_length=args.max_length, batch_size=args.batch_size
    )

    cos = [_cosine(u, v) for u, v in zip(emb_fp32, emb_int8)]
    report = {
        "model": args.model,
        "n": len(cos),
        "max_length": args.max_length,
        "fp32_ms": ms_fp32,
        "int8_ms": ms_int8,
        "cosine": {
            "mean": sum(cos) / max(1, len(cos)),
            "p50": _p(cos, 0.50),
            "p05": _p(cos, 0.05),
            "p01": _p(cos, 0.01),
            "min": min(cos) if cos else 0.0,
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append(f"# Compare embeddings FP32 vs INT8 ({ts})")
    md.append("")
    md.append(f"- model: `{args.model}`")
    md.append(f"- n: {report['n']}, max_length: {args.max_length}")
    md.append(f"- fp32 time: {ms_fp32:.1f} ms total")
    md.append(f"- int8 time: {ms_int8:.1f} ms total")
    md.append("")
    md.append("## Cosine similarity (FP32 vs INT8)")
    md.append(
        f"- mean={report['cosine']['mean']:.5f}, p50={report['cosine']['p50']:.5f}, "
        f"p05={report['cosine']['p05']:.5f}, p01={report['cosine']['p01']:.5f}, min={report['cosine']['min']:.5f}"
    )
    md.append("")
    md.append(f"JSON: `{json_path}`")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()

