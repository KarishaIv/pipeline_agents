#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EmoBench prompt ablation with identical sampled items across variants."
    )
    parser.add_argument("--emobench-path", default="data/benchmark_external/emobench/emobench_test.jsonl")
    parser.add_argument("--synthetic-path", default=None)
    parser.add_argument("--locale", choices=["en", "ru"], default="en")
    parser.add_argument("--persona-sample", type=int, default=12)
    parser.add_argument("--items-per-persona", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--max-runtime-min", type=float, default=90.0)
    parser.add_argument("--balance-col", choices=["auto", "domain", "taxonomy"], default="auto")
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--translation-cache", default="data/benchmark_external/emobench/cache_ru_items.json")
    parser.add_argument(
        "--variants",
        default="baseline,facts_first,appraisal",
        help="Comma-separated variants from: baseline,facts_first,appraisal",
    )
    parser.add_argument(
        "--fixed-items-path",
        default=None,
        help="Optional precomputed sampled items JSONL/CSV. If omitted, first variant will create one.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root dir. Default: outputs/benchmarks/emobench_ablation/<timestamp>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    allowed = {"baseline", "facts_first", "appraisal"}
    invalid = [v for v in variants if v not in allowed]
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}. Allowed: {sorted(allowed)}")
    if not variants:
        raise ValueError("No prompt variants provided.")

    project_root = Path(__file__).resolve().parents[2]
    benchmark_script = project_root / "scripts" / "benchmarks" / "benchmark_emobench.py"
    if not benchmark_script.exists():
        raise FileNotFoundError(f"benchmark_emobench.py not found: {benchmark_script}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else project_root / "outputs" / "benchmarks" / "emobench_ablation" / ts
    ensure_dir(out_root)
    sampled_items_path = Path(args.fixed_items_path) if args.fixed_items_path else out_root / "sampled_items.jsonl"

    summary_rows: List[Dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        variant_out = ensure_dir(out_root / variant)
        cmd: List[str] = [
            sys.executable,
            str(benchmark_script),
            "--emobench-path",
            str(args.emobench_path),
            "--locale",
            str(args.locale),
            "--persona-sample",
            str(args.persona_sample),
            "--items-per-persona",
            str(args.items_per_persona),
            "--seed",
            str(args.seed),
            "--concurrency",
            str(args.concurrency),
            "--max-calls",
            str(args.max_calls),
            "--max-runtime-min",
            str(args.max_runtime_min),
            "--balance-col",
            str(args.balance_col),
            "--parse-retries",
            str(args.parse_retries),
            "--translation-cache",
            str(args.translation_cache),
            "--prompt-variant",
            variant,
            "--out-dir",
            str(variant_out),
        ]
        if args.synthetic_path:
            cmd.extend(["--synthetic-path", str(args.synthetic_path)])

        if idx == 0 and not args.fixed_items_path:
            cmd.extend(["--save-sampled-items-path", str(sampled_items_path)])
        else:
            cmd.extend(["--fixed-items-path", str(sampled_items_path)])

        print(f"[Ablation] Running variant={variant}")
        subprocess.run(cmd, cwd=str(project_root), check=True)

        metrics_path = variant_out / "metrics.json"
        metrics = load_json(metrics_path)
        summary_rows.append(
            {
                "variant": variant,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "accuracy_clean": metrics.get("accuracy_clean"),
                "macro_f1_clean": metrics.get("macro_f1_clean"),
                "partial": metrics.get("partial"),
                "n_rows": metrics.get("n_rows"),
                "n_labeled": metrics.get("n_labeled"),
                "calls_made": (metrics.get("quota") or {}).get("calls_made"),
                "metrics_path": str(metrics_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    rank_col = "macro_f1_clean" if summary_df["macro_f1_clean"].notna().any() else "macro_f1"
    best_idx = summary_df[rank_col].astype(float).idxmax()
    best_variant = summary_df.loc[best_idx, "variant"]

    summary = {
        "variants": variants,
        "seed": int(args.seed),
        "locale": args.locale,
        "sampled_items_path": str(sampled_items_path),
        "rank_metric": rank_col,
        "best_variant": best_variant,
        "rows": summary_rows,
    }

    summary_json = out_root / "summary.json"
    summary_csv = out_root / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary_df.to_csv(summary_csv, index=False)

    print(f"[Ablation] Saved summary: {summary_json}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
