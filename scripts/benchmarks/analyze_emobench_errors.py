#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import f1_score


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].astype(str).str.upper()
    out["y_pred_norm"] = out["y_pred"].fillna("__NONE__").astype(str).str.upper()
    out["correct"] = (out["label"] == out["y_pred_norm"]).astype(int)
    return out


def _summarize_run(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)
    labeled = df[df["label"].notna()].copy()
    labeled = _normalize(labeled) if len(labeled) else labeled
    labels = ["A", "B", "C", "D"]

    item_nunique = (
        labeled.groupby("item_id")["y_pred_norm"].nunique().to_dict()
        if len(labeled)
        else {}
    )
    hardest_taxonomy: List[Dict[str, Any]] = []
    if len(labeled):
        tax = (
            labeled.groupby("taxonomy")["correct"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
            .reset_index()
            .sort_values(["accuracy", "n"], ascending=[True, False])
        )
        hardest_taxonomy = [
            {"taxonomy": row["taxonomy"], "accuracy": float(row["accuracy"]), "n": int(row["n"])}
            for _, row in tax.head(10).iterrows()
        ]

    confusion: Dict[str, Dict[str, int]] = {}
    if len(labeled):
        ctab = pd.crosstab(labeled["label"], labeled["y_pred_norm"])
        for true_label, row in ctab.iterrows():
            confusion[str(true_label)] = {str(col): int(val) for col, val in row.to_dict().items()}

    return {
        "predictions_path": str(path),
        "n_rows": int(len(df)),
        "n_labeled": int(len(labeled)),
        "n_errors": int((df["error"].fillna("") != "").sum()),
        "accuracy": float(labeled["correct"].mean()) if len(labeled) else None,
        "macro_f1": float(
            f1_score(labeled["label"], labeled["y_pred_norm"], labels=labels, average="macro", zero_division=0)
        )
        if len(labeled)
        else None,
        "error_breakdown": {
            str(k): int(v)
            for k, v in df["error"].fillna("").replace("", "<empty>").value_counts().to_dict().items()
        },
        "items_total": int(labeled["item_id"].nunique()) if len(labeled) else 0,
        "items_with_persona_variation": int(sum(1 for v in item_nunique.values() if v > 1)),
        "mean_unique_preds_per_item": float(pd.Series(item_nunique).mean()) if item_nunique else 0.0,
        "hardest_taxonomy_top10": hardest_taxonomy,
        "confusion": confusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze EmoBench predictions and summarize weak spots.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="One or more predictions.csv files from benchmark_emobench.py runs.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path. Default: alongside first predictions as emobench_error_analysis.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_paths = [Path(p) for p in args.predictions]
    summaries = [_summarize_run(path) for path in prediction_paths]

    acc_values = [s["accuracy"] for s in summaries if s["accuracy"] is not None]
    f1_values = [s["macro_f1"] for s in summaries if s["macro_f1"] is not None]
    aggregate = {
        "n_runs": len(summaries),
        "accuracy_mean": float(pd.Series(acc_values).mean()) if acc_values else None,
        "accuracy_std": float(pd.Series(acc_values).std(ddof=1)) if len(acc_values) > 1 else 0.0,
        "macro_f1_mean": float(pd.Series(f1_values).mean()) if f1_values else None,
        "macro_f1_std": float(pd.Series(f1_values).std(ddof=1)) if len(f1_values) > 1 else 0.0,
    }

    out = {
        "aggregate": aggregate,
        "runs": summaries,
    }

    default_out = prediction_paths[0].parent / "emobench_error_analysis.json"
    out_json = Path(args.out_json) if args.out_json else default_out
    ensure_dir(out_json.parent)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved analysis: {out_json}")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
