#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.common import (  # noqa: E402
    QuotaExceeded,
    QuotaGuard,
    as_dict,
    balanced_sample,
    call_text,
    call_structured,
    ensure_dir,
    gather_limited,
    hash_payload,
    load_cache,
    load_personas,
    parse_choice,
    save_cache,
    save_json,
)
from src.benchmarks.persona_render import render_persona_context  # noqa: E402


class TranslationSchema(BaseModel):
    prompt: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _is_likely_english(text: Any) -> bool:
    raw = str(text or "")
    if not raw:
        return False
    ascii_chars = sum(1 for ch in raw if ord(ch) < 128)
    return (ascii_chars / max(1, len(raw))) >= 0.85


def _normalize_label(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    return None


def _prepare_items_pool(rows_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = rows_df.copy()
    df["label"] = df["label"].apply(_normalize_label) if "label" in df.columns else None

    # Prefer rows that are labeled and likely English (to avoid duplicated multilingual rows).
    if args.deduplicate_items and "item_id" in df.columns:
        df["__row_quality"] = (
            df["label"].notna().astype(int) * 10
            + df["prompt"].apply(_is_likely_english).astype(int) * 3
            + df[["option_a", "option_b", "option_c", "option_d"]]
            .astype(str)
            .apply(lambda row: sum(_is_likely_english(x) for x in row), axis=1)
        )
        df = (
            df.sort_values(["item_id", "__row_quality", "prompt"], ascending=[True, False, True])
            .drop_duplicates(subset=["item_id"], keep="first")
            .drop(columns=["__row_quality"])
            .reset_index(drop=True)
        )

    if args.labeled_only:
        df = df[df["label"].notna()].reset_index(drop=True)
    return df


def _pick_balance_column(df: pd.DataFrame, n: int, balance_col: str) -> str:
    if balance_col in {"domain", "taxonomy"} and balance_col in df.columns:
        return balance_col
    if balance_col == "auto":
        taxonomy_unique = int(df["taxonomy"].nunique()) if "taxonomy" in df.columns else 0
        if "taxonomy" in df.columns and 1 < taxonomy_unique <= max(2, n * 2):
            return "taxonomy"
        if "domain" in df.columns:
            return "domain"
    return "domain" if "domain" in df.columns else "taxonomy"


def _build_choice_prompt(
    persona_context: str,
    item: Dict[str, Any],
    strict_output: bool = False,
    prompt_variant: str = "appraisal",
) -> str:
    taxonomy = str(item.get("taxonomy", "default")).lower()
    variant = (prompt_variant or "appraisal").lower()
    extra_rules: List[str] = []
    if variant == "baseline":
        extra_rules.extend(
            [
                "Choose the option that best fits the scenario and available choices.",
                "Keep the decision consistent with the persona profile.",
            ]
        )
    elif variant == "facts_first":
        extra_rules.extend(
            [
                "Base your choice primarily on explicit scenario facts and option text.",
                "Use persona context only as a tie-breaker if two options are equally plausible.",
            ]
        )
    else:
        extra_rules.append(
            "Reason silently in two steps: (1) infer belief/goal/social context, (2) map to the best option."
        )
        if any(key in taxonomy for key in ["false_belief", "perspective_taking", "persona", "strange_story"]):
            extra_rules.append(
                "Use the character's subjective belief and perspective, not objective facts unknown to the character."
            )
    if taxonomy.endswith(":cause"):
        extra_rules.append("Pick the most plausible cause of the emotional reaction.")
    if taxonomy.endswith(":emotion"):
        extra_rules.append("Pick the most plausible emotion expressed in the scenario.")

    strict_line = "Your response must be exactly one character." if strict_output else ""
    rules_block = "\n".join(f"- {rule}" for rule in extra_rules)
    return (
        f"{persona_context}\n\n"
        "You are solving an emotional reasoning multiple-choice item.\n"
        f"Domain: {item.get('domain', 'default')}\n"
        f"Taxonomy: {item.get('taxonomy', 'default')}\n"
        f"{rules_block}\n"
        "Return ONLY one uppercase letter: A, B, C, or D.\n"
        f"{strict_line}\n\n"
        f"Prompt: {item['prompt']}\n"
        f"A) {item['option_a']}\n"
        f"B) {item['option_b']}\n"
        f"C) {item['option_c']}\n"
        f"D) {item['option_d']}"
    ).strip()


def _load_fixed_items(path: Path, prepared_df: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fixed items file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        fixed_df = pd.DataFrame(read_jsonl(path))
    elif path.suffix.lower() == ".csv":
        fixed_df = pd.read_csv(path)
    else:
        raise ValueError("Fixed items must be .jsonl or .csv")

    required = {"item_id", "prompt", "option_a", "option_b", "option_c", "option_d"}
    if required.issubset(set(fixed_df.columns)):
        fixed_df = fixed_df.copy()
        fixed_df["label"] = fixed_df["label"].apply(_normalize_label) if "label" in fixed_df.columns else None
        return fixed_df.reset_index(drop=True)

    if "item_id" not in fixed_df.columns:
        raise ValueError("Fixed items file must contain either full item columns or at least item_id.")

    item_ids = fixed_df["item_id"].astype(str).tolist()
    by_id = prepared_df.copy()
    by_id["item_id"] = by_id["item_id"].astype(str)
    selected = by_id[by_id["item_id"].isin(item_ids)].copy()
    if len(selected) == 0:
        raise ValueError("None of fixed item_ids matched current prepared pool.")
    order_map = {str(item_id): idx for idx, item_id in enumerate(item_ids)}
    selected["__order"] = selected["item_id"].map(order_map).fillna(10**9).astype(int)
    selected = selected.sort_values("__order").drop(columns="__order")
    selected = selected.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    return selected


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
        "option_a": item.get("option_a"),
        "option_b": item.get("option_b"),
        "option_c": item.get("option_c"),
        "option_d": item.get("option_d"),
    }
    key = hash_payload(payload)
    if key in cache:
        out = dict(item)
        out.update(cache[key])
        return out

    prompt = (
        "Переведи задание на русский язык. Сохрани эмоциональные нюансы и смысл. "
        "Верни только JSON.\n\n"
        f"{payload}"
    )
    translated = await call_structured(prompt, TranslationSchema, guard=guard, temperature=0.0)
    cache[key] = as_dict(translated)
    save_cache(cache_path, cache)
    out = dict(item)
    out.update(cache[key])
    return out


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    eval_df = df[df["label"].notna()].copy()
    if len(eval_df) == 0:
        return {
            "n_rows": int(len(df)),
            "n_labeled": 0,
            "n_labeled_clean": 0,
            "accuracy": None,
            "macro_f1": None,
            "accuracy_clean": None,
            "macro_f1_clean": None,
            "accuracy_by_domain": {},
            "accuracy_by_taxonomy": {},
            "error_breakdown": {},
        }
    eval_df["label"] = eval_df["label"].astype(str).str.upper()
    eval_df["y_pred_norm"] = eval_df["y_pred"].fillna("__NONE__").astype(str).str.upper()
    eval_df["correct"] = (eval_df["label"] == eval_df["y_pred_norm"]).astype(int)
    labels = ["A", "B", "C", "D"]
    clean_df = eval_df[eval_df["error"].fillna("") == ""].copy()
    return {
        "n_rows": int(len(df)),
        "n_labeled": int(len(eval_df)),
        "n_labeled_clean": int(len(clean_df)),
        "accuracy": float(eval_df["correct"].mean()),
        "macro_f1": float(
            f1_score(eval_df["label"], eval_df["y_pred_norm"], labels=labels, average="macro", zero_division=0)
        ),
        "accuracy_clean": float(clean_df["correct"].mean()) if len(clean_df) else None,
        "macro_f1_clean": float(
            f1_score(clean_df["label"], clean_df["y_pred_norm"], labels=labels, average="macro", zero_division=0)
        )
        if len(clean_df)
        else None,
        "accuracy_by_domain": {
            k: {"accuracy": float(v["mean"]), "n": int(v["count"])}
            for k, v in (
                eval_df.groupby("domain")["correct"].agg(["mean", "count"]).to_dict(orient="index")
            ).items()
        },
        "accuracy_by_taxonomy": {
            k: {"accuracy": float(v["mean"]), "n": int(v["count"])}
            for k, v in (
                eval_df.groupby("taxonomy")["correct"].agg(["mean", "count"]).to_dict(orient="index")
            ).items()
        },
        "error_breakdown": {
            str(k): int(v)
            for k, v in df["error"].fillna("").replace("", "<empty>").value_counts().to_dict().items()
        },
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = Path(args.emobench_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"EmoBench dataset not found: {data_path}. "
            "Run scripts/benchmarks/prepare_external_data.py first."
        )

    rows_df = pd.DataFrame(read_jsonl(data_path))
    required = {"item_id", "prompt", "option_a", "option_b", "option_c", "option_d"}
    missing = required - set(rows_df.columns)
    if missing:
        raise ValueError(f"EmoBench file missing columns: {sorted(missing)}")

    prepared_df = _prepare_items_pool(rows_df, args)
    if len(prepared_df) == 0:
        raise ValueError("No EmoBench rows left after filtering/dedup. Relax flags or refresh source data.")

    out_dir = ensure_dir(Path(args.out_dir))

    sample_source = "sampled"
    group_col = _pick_balance_column(prepared_df, args.items_per_persona, args.balance_col)
    if args.fixed_items_path:
        sample_source = "fixed_items_path"
        sampled_items_df = _load_fixed_items(Path(args.fixed_items_path), prepared_df)
    else:
        sampled_items_df = balanced_sample(prepared_df, group_col, args.items_per_persona, args.seed)
    sampled_items = sampled_items_df.to_dict(orient="records")
    if len(sampled_items) == 0:
        raise ValueError("No sampled EmoBench items found.")
    sampled_items_path = Path(args.save_sampled_items_path) if args.save_sampled_items_path else out_dir / "sampled_items.jsonl"
    write_jsonl(sampled_items_path, sampled_items)

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
        try:
            choice: Optional[str] = None
            last_text = ""
            attempts = max(1, 1 + int(args.parse_retries))
            for attempt in range(attempts):
                prompt = _build_choice_prompt(
                    persona_context,
                    item,
                    strict_output=(attempt > 0),
                    prompt_variant=args.prompt_variant,
                )
                last_text = await call_text(prompt, guard=guard, temperature=0.0)
                parsed = parse_choice(last_text, allowed=["A", "B", "C", "D"], default="")
                if parsed:
                    choice = parsed
                    break
            reasoning = ""
            error = "" if choice else "unparsed_model_output"
        except QuotaExceeded as exc:
            choice = None
            reasoning = ""
            error = f"quota_exceeded: {exc}"
        except Exception as exc:
            choice = None
            reasoning = ""
            error = str(exc)
        return {
            "persona_id": f"persona_{p_idx}",
            "item_id": item.get("item_id", f"item_{i_idx}"),
            "domain": item.get("domain", "default"),
            "taxonomy": item.get("taxonomy", "default"),
            "label": item.get("label"),
            "y_pred": choice,
            "reasoning": reasoning,
            "locale": args.locale,
            "error": error,
        }

    predictions = await gather_limited(jobs, worker, concurrency=args.concurrency)
    pred_df = pd.DataFrame(predictions)

    pred_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(pred_df)
    metrics.update(
        {
            "benchmark": "emobench",
            "locale": args.locale,
            "persona_sample": int(args.persona_sample),
            "items_per_persona": int(args.items_per_persona),
            "prompt_variant": args.prompt_variant,
            "sample_source": sample_source,
            "sampled_items_path": str(sampled_items_path),
            "balance_column": group_col,
            "partial": bool((pred_df["error"] != "").any()),
            "quota": guard.snapshot(),
        }
    )
    save_json(metrics, metrics_path)
    return {"predictions_path": str(pred_path), "metrics_path": str(metrics_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-compatible EmoBench benchmark for synthetic personas.")
    parser.add_argument("--emobench-path", default="data/benchmark_external/emobench/emobench_test.jsonl")
    parser.add_argument("--synthetic-path", default=None, help="Path to synthetic personas CSV.")
    parser.add_argument("--locale", choices=["en", "ru"], default="en")
    parser.add_argument("--persona-sample", type=int, default=15)
    parser.add_argument("--items-per-persona", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--max-runtime-min", type=float, default=180.0)
    parser.add_argument(
        "--prompt-variant",
        choices=["baseline", "facts_first", "appraisal"],
        default="appraisal",
        help="Prompt template variant for emotional reasoning.",
    )
    parser.add_argument(
        "--balance-col",
        choices=["auto", "domain", "taxonomy"],
        default="auto",
        help="Column used for balanced item sampling.",
    )
    parser.add_argument(
        "--labeled-only",
        dest="labeled_only",
        action="store_true",
        help="Use only rows with valid A/B/C/D labels (default).",
    )
    parser.add_argument(
        "--allow-unlabeled",
        dest="labeled_only",
        action="store_false",
        help="Allow unlabeled rows in sampled items pool.",
    )
    parser.add_argument(
        "--deduplicate-items",
        dest="deduplicate_items",
        action="store_true",
        help="Deduplicate by item_id preferring labeled and likely-English rows (default).",
    )
    parser.add_argument(
        "--no-deduplicate-items",
        dest="deduplicate_items",
        action="store_false",
        help="Disable item_id deduplication.",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=1,
        help="Extra retries if model doesn't return A/B/C/D.",
    )
    parser.add_argument(
        "--fixed-items-path",
        default=None,
        help="Optional JSONL/CSV path with fixed items (or item_id list) to ensure comparable runs.",
    )
    parser.add_argument(
        "--save-sampled-items-path",
        default=None,
        help="Where to save selected item set (JSONL). Defaults to <out_dir>/sampled_items.jsonl.",
    )
    parser.add_argument(
        "--translation-cache",
        default="data/benchmark_external/emobench/cache_ru_items.json",
        help="Cache for translated EmoBench items.",
    )
    parser.add_argument("--out-dir", default="outputs/benchmarks/suite/latest/emobench")
    parser.set_defaults(labeled_only=True, deduplicate_items=True)
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
