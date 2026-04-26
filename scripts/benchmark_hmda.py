#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import set_openai_api_key
from src.schemas.decision_schema import DecisionOutcome
from src.utils import robust_llm_call


def _normalize_missing(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.upper() in {"NA", "N/A", "EXEMPT"}:
        return None
    return s


def _parse_age(age_value: Any) -> Optional[int]:
    s = _normalize_missing(age_value)
    if s is None:
        return None
    s = s.replace("+", "").replace("≥", ">=").replace("≤", "<=")
    if s.startswith("<"):
        return 24
    if s.startswith(">="):
        return 80
    if "-" in s:
        parts = s.split("-")
        try:
            lo = int(parts[0])
            hi = int(parts[1])
            return int((lo + hi) / 2)
        except ValueError:
            return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_income(value: Any) -> Optional[int]:
    s = _normalize_missing(value)
    if s is None:
        return None
    try:
        # HMDA income is in thousands of dollars
        return int(float(s) * 1000)
    except ValueError:
        return None


def _parse_ratio(value: Any) -> Optional[float]:
    s = _normalize_missing(value)
    if s is None:
        return None
    s = s.replace("%", "").replace(" ", "")
    if s.startswith("<"):
        try:
            return float(s[1:]) - 5
        except ValueError:
            return None
    if s.startswith(">=") or s.startswith(">"):
        try:
            return float(s.lstrip("=>"))
        except ValueError:
            return None
    if "-" in s:
        try:
            lo, hi = s.split("-", 1)
            lo_v = float(lo)
            hi_v = float(hi.replace("<", ""))
            return (lo_v + hi_v) / 2
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_amount(value: Any) -> Optional[int]:
    s = _normalize_missing(value)
    if s is None:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _map_sex(value: Any) -> Optional[str]:
    s = _normalize_missing(value)
    if s is None:
        return None
    try:
        code = int(float(s))
    except ValueError:
        return None
    if code == 1:
        return "Male"
    if code == 2:
        return "Female"
    return None


def _map_yes_no(value: Any) -> Optional[str]:
    s = _normalize_missing(value)
    if s is None:
        return None
    try:
        code = int(float(s))
    except ValueError:
        return None
    if code == 1:
        return "Yes"
    if code == 2:
        return "No"
    return None


def _action_to_label(value: Any) -> Optional[int]:
    s = _normalize_missing(value)
    if s is None:
        return None
    try:
        code = int(float(s))
    except ValueError:
        return None
    if code == 1:
        return 1
    if code in {2, 3}:
        return 0
    return None


def _build_profile(row: pd.Series) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "age": _parse_age(row.get("applicant_age")),
        "gender": _map_sex(row.get("applicant_sex")),
        "income": _parse_income(row.get("income")),
        "income_unit": "USD",
        "state": _normalize_missing(row.get("state_code")),
        "loan_amount": _parse_amount(row.get("loan_amount")),
        "loan_purpose": _normalize_missing(row.get("loan_purpose")),
        "loan_type": _normalize_missing(row.get("loan_type")),
        "occupancy_type": _normalize_missing(row.get("occupancy_type")),
        "lien_status": _normalize_missing(row.get("lien_status")),
        "loan_term": _parse_amount(row.get("loan_term")),
        "property_value": _parse_amount(row.get("property_value")),
        "interest_rate": _parse_ratio(row.get("interest_rate")),
        "rate_spread": _parse_ratio(row.get("rate_spread")),
        "debt_to_income_ratio": _parse_ratio(row.get("debt_to_income_ratio")),
        "combined_loan_to_value_ratio": _parse_ratio(row.get("combined_loan_to_value_ratio")),
        "total_units": _parse_amount(row.get("total_units")),
        "applicant_age_above_62": _map_yes_no(row.get("applicant_age_above_62")),
        "co_applicant_sex": _map_sex(row.get("co_applicant_sex")),
        "co_applicant_age": _parse_age(row.get("co_applicant_age")),
        "balloon_payment": _map_yes_no(row.get("balloon_payment")),
        "interest_only_payment": _map_yes_no(row.get("interest_only_payment")),
        "reverse_mortgage": _map_yes_no(row.get("reverse_mortgage")),
        "open_end_line_of_credit": _map_yes_no(row.get("open_end_line_of_credit")),
        "preapproval": _map_yes_no(row.get("preapproval")),
        "construction_method": _normalize_missing(row.get("construction_method")),
        "target_audience_name": "hmda_benchmark",
    }
    # Remove None values to keep prompts cleaner
    return {k: v for k, v in profile.items() if v is not None}


def _read_hmda_head(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("|")
    return header


def _load_hmda_sample(
    path: Path,
    sample_size: int,
    seed: int,
    random_sample: bool,
    usecols: List[str],
) -> pd.DataFrame:
    if not random_sample:
        return pd.read_csv(
            path,
            sep="|",
            usecols=usecols,
            nrows=sample_size,
            dtype=str,
            low_memory=False,
        )

    rng = random.Random(seed)
    reservoir: List[pd.Series] = []
    seen = 0
    for chunk in pd.read_csv(
        path,
        sep="|",
        usecols=usecols,
        dtype=str,
        chunksize=100_000,
        low_memory=False,
    ):
        for _, row in chunk.iterrows():
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.randint(0, seen - 1)
                if j < sample_size:
                    reservoir[j] = row
    return pd.DataFrame(reservoir)


async def _run_llm_predictions(
    df: pd.DataFrame,
    concurrency: int,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def run_row(row: pd.Series) -> Dict[str, Any]:
        async with sem:
            profile = _build_profile(row)
            prompt = (
                "Ты кредитный аналитик. На основе анкеты и параметров займа оцени, "
                "будет ли заявка одобрена (action_taken=1). "
                "Верни will_take_credit=true если ожидаешь одобрение, иначе false. "
                "Верни probability_score как вероятность одобрения (0-1)."
                f"\nПрофиль заявки: {profile}"
            )
            decision = await robust_llm_call(prompt, structured_output=DecisionOutcome)
            return {
                "will_take_credit": bool(decision.will_take_credit),
                "probability_score": float(decision.probability_score),
                "reasoning": decision.reasoning,
                "emotional_factors": decision.emotional_factors,
            }

    tasks = [run_row(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    return results


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def _prepare_baseline_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_cols = [
        "income",
        "loan_amount",
        "property_value",
        "interest_rate",
        "rate_spread",
        "debt_to_income_ratio",
        "combined_loan_to_value_ratio",
        "loan_term",
        "total_units",
    ]
    categorical_cols = [
        "loan_type",
        "loan_purpose",
        "occupancy_type",
        "lien_status",
        "preapproval",
        "construction_method",
        "applicant_sex",
        "co_applicant_sex",
        "applicant_age",
        "co_applicant_age",
        "applicant_age_above_62",
        "balloon_payment",
        "interest_only_payment",
        "reverse_mortgage",
        "open_end_line_of_credit",
        "state_code",
    ]

    feats = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        if col not in df.columns:
            feats[col] = np.nan
            continue
        if col in {"interest_rate", "rate_spread", "debt_to_income_ratio", "combined_loan_to_value_ratio"}:
            feats[col] = df[col].apply(_parse_ratio)
        elif col in {"income"}:
            feats[col] = df[col].apply(_parse_income)
        else:
            feats[col] = df[col].apply(_parse_amount)

    for col in categorical_cols:
        if col not in df.columns:
            feats[col] = None
            continue
        if col in {"applicant_sex", "co_applicant_sex"}:
            feats[col] = df[col].apply(_map_sex)
        elif col in {"applicant_age_above_62", "balloon_payment", "interest_only_payment", "reverse_mortgage", "open_end_line_of_credit", "preapproval"}:
            feats[col] = df[col].apply(_map_yes_no)
        else:
            feats[col] = df[col].apply(_normalize_missing)

    y = df["y_true"].astype(int)
    return feats, y


def _run_baseline_model(df: pd.DataFrame, seed: int, threshold: float) -> Dict[str, Any]:
    X, y = _prepare_baseline_features(df)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = _compute_metrics(y_test.to_numpy(), y_pred, y_prob)
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DecisionAgent on HMDA 2024 (Modified LAR).")
    parser.add_argument(
        "--hmda-path",
        default="data/benchmark/hmda/2024_combined_mlar_header.txt",
        help="Path to HMDA 2024 file with headers (pipe-delimited).",
    )
    parser.add_argument("--sample-size", type=int, default=150_000, help="Rows to load from HMDA file.")
    parser.add_argument("--llm-sample", type=int, default=2000, help="Rows to run LLM on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--random-sample", action="store_true", help="Reservoir-sample from full file (slow).")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent LLM calls.")
    parser.add_argument("--api-key", type=str, default=None, help="Yandex API key (optional).")
    parser.add_argument("--folder-id", type=str, default=None, help="Yandex folder_id (optional).")
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="Threshold for positive prediction.")
    parser.add_argument("--baseline", action="store_true", help="Run baseline logistic regression.")
    parser.add_argument("--baseline-sample", type=int, default=None, help="Rows to use for baseline (default: all).")
    parser.add_argument("--out-dir", default="outputs/benchmarks/hmda", help="Output directory.")
    args = parser.parse_args()

    if args.api_key:
        set_openai_api_key(args.api_key, args.folder_id)

    path = Path(args.hmda_path)
    if not path.exists():
        raise FileNotFoundError(f"HMDA file not found: {path}")

    header = _read_hmda_head(path)
    required_cols = [
        "action_taken",
        "applicant_age",
        "applicant_sex",
        "income",
        "state_code",
        "loan_amount",
        "loan_purpose",
        "loan_type",
        "occupancy_type",
        "lien_status",
        "loan_term",
        "property_value",
        "interest_rate",
        "rate_spread",
        "debt_to_income_ratio",
        "combined_loan_to_value_ratio",
        "total_units",
        "applicant_age_above_62",
        "co_applicant_sex",
        "co_applicant_age",
        "balloon_payment",
        "interest_only_payment",
        "reverse_mortgage",
        "open_end_line_of_credit",
        "preapproval",
        "construction_method",
    ]
    usecols = [c for c in required_cols if c in header]
    if "action_taken" not in usecols:
        raise ValueError("HMDA file missing required column: action_taken")

    df = _load_hmda_sample(
        path=path,
        sample_size=args.sample_size,
        seed=args.seed,
        random_sample=args.random_sample,
        usecols=usecols,
    )

    df["y_true"] = df["action_taken"].apply(_action_to_label)
    df = df[df["y_true"].notna()].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No usable rows after filtering action_taken in {1,2,3}.")

    if args.llm_sample < len(df):
        df_llm = df.sample(n=args.llm_sample, random_state=args.seed).reset_index(drop=True)
    else:
        df_llm = df

    predictions = asyncio.run(_run_llm_predictions(df_llm, concurrency=args.concurrency))

    y_true = df_llm["y_true"].astype(int).to_numpy()
    y_prob = np.array([p["probability_score"] for p in predictions], dtype=float)
    y_pred = (y_prob >= args.prob_threshold).astype(int)

    metrics = _compute_metrics(y_true, y_pred, y_prob)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "hmda_predictions.csv"
    out_json = out_dir / "hmda_metrics.json"

    out_df = df_llm.copy()
    out_df["y_pred"] = y_pred
    out_df["probability_score"] = y_prob
    out_df.to_csv(out_csv, index=False)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions: {out_csv}")
    print(f"Saved metrics: {out_json}")
    print("Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.baseline:
        baseline_df = df
        if args.baseline_sample and args.baseline_sample < len(df):
            baseline_df = df.sample(n=args.baseline_sample, random_state=args.seed).reset_index(drop=True)
        baseline_metrics = _run_baseline_model(baseline_df, seed=args.seed, threshold=args.prob_threshold)
        baseline_path = out_dir / "hmda_baseline_metrics.json"
        with baseline_path.open("w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved baseline metrics: {baseline_path}")
        print("Baseline metrics:", json.dumps(baseline_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
