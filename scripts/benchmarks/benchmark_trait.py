#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.common import (  # noqa: E402
    QuotaExceeded,
    QuotaGuard,
    as_dict,
    balanced_sample,
    call_structured,
    ensure_dir,
    gather_limited,
    hash_payload,
    load_cache,
    load_personas,
    parse_int_choice,
    save_cache,
    save_json,
)
from src.benchmarks.persona_render import render_persona_context  # noqa: E402


class TraitChoiceSchema(BaseModel):
    choice: int = Field(..., ge=1, le=4)
    reasoning: str


class TranslationSchema(BaseModel):
    prompt: str
    option_1: str
    option_2: str
    option_3: str
    option_4: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def translate_item_if_needed(
    item: Dict[str, Any],
    locale: str,
    cache: Dict[str, Any],
    cache_path: Path,
    guard: QuotaGuard,
) -> Dict[str, Any]:
    if locale != "ru":
        return item
    payload = {
        "prompt": item.get("prompt"),
        "option_1": item.get("option_1"),
        "option_2": item.get("option_2"),
        "option_3": item.get("option_3"),
        "option_4": item.get("option_4"),
    }
    key = hash_payload(payload)
    if key in cache:
        translated = dict(item)
        translated.update(cache[key])
        return translated

    prompt = (
        "Переведи вопрос и варианты ответов на русский язык. "
        "Сохрани смысл и нейтральный тон. Верни только JSON.\n\n"
        f"{payload}"
    )
    translated = await call_structured(prompt, TranslationSchema, guard=guard, temperature=0.0)
    cache[key] = as_dict(translated)
    save_cache(cache_path, cache)
    out = dict(item)
    out.update(cache[key])
    return out


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    eval_df = df[df["answer"].notna()].copy()
    if len(eval_df) == 0:
        return {
            "n_rows": int(len(df)),
            "n_labeled": 0,
            "accuracy": None,
            "accuracy_by_subset": {},
            "consistency_std": None,
        }
    eval_df["answer"] = eval_df["answer"].astype(str)
    eval_df["correct"] = (eval_df["answer"] == eval_df["y_pred"].astype(str)).astype(int)
    acc_by_subset = (
        eval_df.groupby("subset")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .reset_index()
    )
    persona_acc = eval_df.groupby("persona_id")["correct"].mean()
    return {
        "n_rows": int(len(df)),
        "n_labeled": int(len(eval_df)),
        "accuracy": float(eval_df["correct"].mean()),
        "accuracy_by_subset": {
            row["subset"]: {"accuracy": float(row["accuracy"]), "n": int(row["n"])}
            for _, row in acc_by_subset.iterrows()
        },
        "consistency_std": float(persona_acc.std(ddof=0)) if len(persona_acc) > 1 else 0.0,
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    trait_path = Path(args.trait_path)
    if not trait_path.exists():
        raise FileNotFoundError(
            f"TRAIT dataset not found: {trait_path}. "
            "Run scripts/benchmarks/prepare_external_data.py first."
        )

    raw_items = pd.DataFrame(read_jsonl(trait_path))
    required = {"item_id", "prompt", "option_1", "option_2", "option_3", "option_4"}
    missing = required - set(raw_items.columns)
    if missing:
        raise ValueError(f"TRAIT file missing columns: {sorted(missing)}")

    sampled_items_df = balanced_sample(raw_items, "subset", args.items_per_persona, args.seed)
    sampled_items = sampled_items_df.to_dict(orient="records")

    personas_df = load_personas(args.synthetic_path, args.persona_sample, args.seed)
    personas = personas_df.to_dict(orient="records")

    guard = QuotaGuard(max_calls=args.max_calls, max_runtime_min=args.max_runtime_min)
    cache_path = Path(args.translation_cache)
    cache = load_cache(cache_path)

    translated_items: List[Dict[str, Any]] = []
    for item in sampled_items:
        translated_items.append(await translate_item_if_needed(item, args.locale, cache, cache_path, guard))

    jobs: List[Tuple[int, int]] = []
    for p_idx in range(len(personas)):
        for i_idx in range(len(translated_items)):
            jobs.append((p_idx, i_idx))

    async def worker(_: int, pair: Tuple[int, int]) -> Dict[str, Any]:
        p_idx, i_idx = pair
        persona = personas[p_idx]
        item = translated_items[i_idx]
        persona_context = render_persona_context(persona, args.locale)
        prompt = (
            f"{persona_context}\n\n"
            "Choose the most likely option for this persona.\n"
            "Return JSON with fields: choice (1..4), reasoning.\n\n"
            f"Question: {item['prompt']}\n"
            f"1) {item['option_1']}\n"
            f"2) {item['option_2']}\n"
            f"3) {item['option_3']}\n"
            f"4) {item['option_4']}"
        )
        try:
            pred = await call_structured(prompt, TraitChoiceSchema, guard=guard, temperature=0.0)
            choice = parse_int_choice(pred.get("choice"), 1, 4, default=1)
            reasoning = str(pred.get("reasoning", ""))
            error = ""
        except QuotaExceeded as exc:
            choice = 1
            reasoning = ""
            error = f"quota_exceeded: {exc}"
        except Exception as exc:
            choice = 1
            reasoning = ""
            error = str(exc)
        return {
            "persona_id": f"persona_{p_idx}",
            "item_id": item.get("item_id", f"item_{i_idx}"),
            "subset": item.get("subset", "default"),
            "answer": item.get("answer"),
            "y_pred": choice,
            "reasoning": reasoning,
            "locale": args.locale,
            "error": error,
        }

    predictions = await gather_limited(jobs, worker, concurrency=args.concurrency)
    pred_df = pd.DataFrame(predictions)

    out_dir = ensure_dir(Path(args.out_dir))
    pred_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(pred_df)
    metrics.update(
        {
            "benchmark": "trait",
            "locale": args.locale,
            "persona_sample": int(args.persona_sample),
            "items_per_persona": int(args.items_per_persona),
            "partial": bool((pred_df["error"] != "").any()),
            "quota": guard.snapshot(),
        }
    )
    save_json(metrics, metrics_path)
    return {"predictions_path": str(pred_path), "metrics_path": str(metrics_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-compatible TRAIT benchmark for synthetic personas.")
    parser.add_argument("--trait-path", default="data/benchmark_external/trait/trait_train.jsonl")
    parser.add_argument("--synthetic-path", default=None, help="Path to synthetic personas CSV.")
    parser.add_argument("--locale", choices=["en", "ru"], default="en")
    parser.add_argument("--persona-sample", type=int, default=15)
    parser.add_argument("--items-per-persona", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--max-runtime-min", type=float, default=180.0)
    parser.add_argument(
        "--translation-cache",
        default="data/benchmark_external/trait/cache_ru_items.json",
        help="Cache for translated TRAIT items.",
    )
    parser.add_argument("--out-dir", default="outputs/benchmarks/suite/latest/trait")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    print(f"Saved predictions: {result['predictions_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

