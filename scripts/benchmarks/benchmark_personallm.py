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
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.common import (  # noqa: E402
    QuotaExceeded,
    QuotaGuard,
    call_structured,
    call_text,
    ensure_dir,
    gather_limited,
    load_json,
    load_personas,
    parse_int_choice,
    save_json,
)
from src.benchmarks.judge import judge_behavior_consistency  # noqa: E402
from src.benchmarks.persona_render import render_answer_style_hint, render_persona_context  # noqa: E402


class BfiAnswerSchema(BaseModel):
    score: int = Field(..., ge=1, le=5)
    reasoning: str


BEHAVIOR_SCENARIOS = {
    "en": [
        ("risk", "A loan offer has hidden fee clauses. Do you still proceed, delay, or reject?"),
        ("debt", "You can buy now on credit, but monthly payments reduce your comfort budget. What do you do?"),
        ("social_pressure", "Family pushes you to take a loan for status consumption. How do you respond?"),
        ("planning", "You choose between a short expensive loan and a longer cheaper one. Explain your choice."),
        ("stress_coping", "Income uncertainty appears right before signing a loan. What is your next action?"),
    ],
    "ru": [
        ("risk", "В кредитном предложении есть скрытые комиссии. Вы продолжаете, откладываете или отказываетесь?"),
        ("debt", "Можно купить сейчас в кредит, но ежемесячный платеж заметно снизит комфортный бюджет. Что вы делаете?"),
        ("social_pressure", "Семья давит и убеждает взять кредит ради статусной покупки. Как вы реагируете?"),
        ("planning", "Вы выбираете между коротким дорогим кредитом и длинным более дешевым. Объясните выбор."),
        ("stress_coping", "Перед подписанием кредита возникла неопределенность с доходом. Ваш следующий шаг?"),
    ],
}


OCEAN_TARGET_MAP: Dict[str, Dict[str, float]] = {
    "openness": {
        "умеренно открыт новому": 3.0,
        "открыт новому опыту и идеям": 4.0,
    },
    "conscientiousness": {
        "обычно ответственен и организован": 3.0,
        "дисциплинирован, пунктуален, ответственен": 4.0,
        "очень организован, планирует всё до деталей": 5.0,
    },
    "extraversion": {
        "склонен к одиночеству, избегает социума": 2.0,
        "умеренно общителен, зависит от настроения": 3.0,
        "общительный, энергичный, легко находит друзей": 4.0,
        "очень общителен, постоянно ищет социального взаимодействия": 5.0,
    },
    "agreeableness": {
        "альтруист, эмпатичен, готов помочь": 4.0,
        "очень эмпатичен, ставит нужды других выше своих": 5.0,
    },
    "neuroticism": {
        "обычно спокоен, но может реагировать на сложности": 2.0,
    },
}


def _target_trait_score(persona: Dict[str, Any], trait: str) -> float:
    raw = str(persona.get(trait, "")).strip()
    return OCEAN_TARGET_MAP.get(trait, {}).get(raw, 3.0)


def _score_traits(
    bfi_df: pd.DataFrame,
    scoring_key: Dict[str, Any],
) -> pd.DataFrame:
    traits_cfg = scoring_key["traits"]
    records: List[Dict[str, Any]] = []
    for persona_id, part in bfi_df.groupby("persona_id"):
        by_item = {int(row["item_id"]): float(row["score"]) for _, row in part.iterrows()}
        for trait, cfg in traits_cfg.items():
            vals: List[float] = []
            reverse = set(int(x) for x in cfg.get("reverse_items", []))
            for item_id in cfg.get("items", []):
                score = by_item.get(int(item_id))
                if score is None:
                    continue
                if int(item_id) in reverse:
                    score = 6.0 - score
                vals.append(score)
            records.append(
                {
                    "persona_id": persona_id,
                    "trait": trait,
                    "pred_trait_score": float(np.mean(vals)) if vals else np.nan,
                    "n_items": int(len(vals)),
                }
            )
    return pd.DataFrame(records)


def _compute_metrics(
    trait_scores_df: pd.DataFrame,
    target_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
) -> Dict[str, Any]:
    merged = trait_scores_df.merge(target_df, on=["persona_id", "trait"], how="inner")
    merged["abs_err"] = (merged["pred_trait_score"] - merged["target_trait_score"]).abs()
    trait_mae = float(merged["abs_err"].mean()) if len(merged) else None
    spearman = None
    if len(merged) >= 3:
        corr = spearmanr(merged["pred_trait_score"], merged["target_trait_score"], nan_policy="omit")
        spearman = float(corr.correlation) if corr.correlation == corr.correlation else None

    behavior_mean = float(behavior_df["combined_score"].mean()) if len(behavior_df) else None
    trait_component = 0.5 if trait_mae is None else max(0.0, 1.0 - (trait_mae / 4.0))
    spearman_component = 0.5 if spearman is None else (spearman + 1.0) / 2.0
    behavior_component = 0.5 if behavior_mean is None else behavior_mean / 5.0
    composite = float(np.mean([trait_component, spearman_component, behavior_component]) * 100.0)

    by_trait = (
        merged.groupby("trait")["abs_err"].agg(["mean", "count"]).reset_index()
        if len(merged)
        else pd.DataFrame(columns=["trait", "mean", "count"])
    )

    return {
        "n_trait_rows": int(len(merged)),
        "trait_mae": trait_mae,
        "trait_spearman": spearman,
        "behavior_consistency_mean": behavior_mean,
        "personallm_composite_score": composite,
        "trait_mae_by_trait": {
            row["trait"]: {"mae": float(row["mean"]), "n": int(row["count"])}
            for _, row in by_trait.iterrows()
        },
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    bfi_items_path = Path(args.bfi_items_path)
    bfi_scoring_path = Path(args.bfi_scoring_path)
    if not bfi_items_path.exists():
        raise FileNotFoundError(f"BFI items file not found: {bfi_items_path}")
    if not bfi_scoring_path.exists():
        raise FileNotFoundError(f"BFI scoring key file not found: {bfi_scoring_path}")

    bfi_items = load_json(bfi_items_path, default=[])
    scoring_key = load_json(bfi_scoring_path, default={})
    if not isinstance(bfi_items, list) or not bfi_items:
        raise ValueError("BFI items file is empty or invalid.")
    if "traits" not in scoring_key:
        raise ValueError("BFI scoring key is invalid.")

    personas_df = load_personas(args.synthetic_path, args.persona_sample, args.seed)
    personas = personas_df.to_dict(orient="records")
    guard = QuotaGuard(max_calls=args.max_calls, max_runtime_min=args.max_runtime_min)

    bfi_jobs: List[Tuple[int, Dict[str, Any]]] = []
    for p_idx in range(len(personas)):
        for item in bfi_items:
            bfi_jobs.append((p_idx, item))

    async def bfi_worker(_: int, job: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        p_idx, item = job
        persona = personas[p_idx]
        persona_context = render_persona_context(persona, args.locale)
        text = item["text_ru"] if args.locale == "ru" else item["text_en"]
        prompt = (
            f"{persona_context}\n\n"
            "Rate this statement on a Likert scale 1..5 for this persona:\n"
            "1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree.\n"
            "Return JSON with fields: score, reasoning.\n\n"
            f"Statement: {text}"
        )
        try:
            pred = await call_structured(prompt, BfiAnswerSchema, guard=guard, temperature=0.0)
            score = parse_int_choice(pred.get("score"), 1, 5, default=3)
            reasoning = str(pred.get("reasoning", ""))
            error = ""
        except QuotaExceeded as exc:
            score = 3
            reasoning = ""
            error = f"quota_exceeded: {exc}"
        except Exception as exc:
            score = 3
            reasoning = ""
            error = str(exc)
        return {
            "row_type": "bfi_response",
            "persona_id": f"persona_{p_idx}",
            "item_id": int(item["id"]),
            "trait": item.get("trait"),
            "reverse": bool(item.get("reverse", False)),
            "score": score,
            "reasoning": reasoning,
            "locale": args.locale,
            "error": error,
        }

    bfi_rows = await gather_limited(bfi_jobs, bfi_worker, concurrency=args.concurrency)
    bfi_df = pd.DataFrame(bfi_rows)

    # Behavior scenarios (5 per persona in pilot).
    behavior_jobs = [(p_idx, scenario) for p_idx in range(len(personas)) for scenario in BEHAVIOR_SCENARIOS[args.locale]]

    async def behavior_worker(_: int, job: Tuple[int, Tuple[str, str]]) -> Dict[str, Any]:
        p_idx, (scenario_id, scenario_text) = job
        persona = personas[p_idx]
        persona_context = render_persona_context(persona, args.locale)
        prompt = (
            f"{persona_context}\n\nScenario:\n{scenario_text}\n\n"
            f"{render_answer_style_hint(args.locale)}"
        )
        try:
            response = await call_text(prompt, guard=guard, temperature=0.2)
            judge = await judge_behavior_consistency(
                persona_context=persona_context,
                scenario=scenario_text,
                response_text=response,
                locale=args.locale,
                guard=guard,
            )
            combined = float(judge["combined_score"])
            error = ""
        except QuotaExceeded as exc:
            response = ""
            judge = {"llm_score": 1.0, "rule_score": 1.0, "combined_score": 1.0, "reasoning": ""}
            combined = 1.0
            error = f"quota_exceeded: {exc}"
        except Exception as exc:
            response = ""
            judge = {"llm_score": 1.0, "rule_score": 1.0, "combined_score": 1.0, "reasoning": ""}
            combined = 1.0
            error = str(exc)

        return {
            "row_type": "behavior_response",
            "persona_id": f"persona_{p_idx}",
            "scenario_id": scenario_id,
            "scenario_text": scenario_text,
            "response_text": response,
            "llm_score": judge["llm_score"],
            "rule_score": judge["rule_score"],
            "combined_score": combined,
            "judge_reasoning": judge.get("reasoning", ""),
            "locale": args.locale,
            "error": error,
        }

    behavior_rows = await gather_limited(behavior_jobs, behavior_worker, concurrency=args.concurrency)
    behavior_df = pd.DataFrame(behavior_rows)

    # Trait targets from synthetic OCEAN text labels.
    target_rows: List[Dict[str, Any]] = []
    for p_idx, persona in enumerate(personas):
        pid = f"persona_{p_idx}"
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            target_rows.append(
                {"persona_id": pid, "trait": trait, "target_trait_score": _target_trait_score(persona, trait)}
            )
    target_df = pd.DataFrame(target_rows)

    trait_scores_df = _score_traits(bfi_df, scoring_key)
    metrics = _compute_metrics(trait_scores_df, target_df, behavior_df)
    metrics.update(
        {
            "benchmark": "personallm_protocol",
            "locale": args.locale,
            "persona_sample": int(args.persona_sample),
            "n_bfi_items": int(len(bfi_items)),
            "n_behavior_scenarios": int(len(BEHAVIOR_SCENARIOS[args.locale])),
            "partial": bool((bfi_df["error"] != "").any() or (behavior_df["error"] != "").any()),
            "quota": guard.snapshot(),
        }
    )

    out_df = pd.concat([bfi_df, behavior_df], ignore_index=True, sort=False)
    out_dir = ensure_dir(Path(args.out_dir))
    responses_path = out_dir / "responses.csv"
    metrics_path = out_dir / "metrics.json"
    trait_scores_path = out_dir / "trait_scores.csv"
    out_df.to_csv(responses_path, index=False)
    trait_scores_df.to_csv(trait_scores_path, index=False)
    save_json(metrics, metrics_path)

    return {
        "responses_path": str(responses_path),
        "trait_scores_path": str(trait_scores_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-compatible PersonaLLM benchmark.")
    parser.add_argument("--bfi-items-path", default="data/benchmark_external/personallm/bfi44_items_en_ru.json")
    parser.add_argument("--bfi-scoring-path", default="data/benchmark_external/personallm/bfi44_scoring_key.json")
    parser.add_argument("--synthetic-path", default=None, help="Path to synthetic personas CSV.")
    parser.add_argument("--locale", choices=["en", "ru"], default="en")
    parser.add_argument("--persona-sample", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--max-runtime-min", type=float, default=180.0)
    parser.add_argument("--out-dir", default="outputs/benchmarks/suite/latest/personallm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    print(f"Saved responses: {result['responses_path']}")
    print(f"Saved trait scores: {result['trait_scores_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

