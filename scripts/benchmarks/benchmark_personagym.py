#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.common import (  # noqa: E402
    QuotaExceeded,
    QuotaGuard,
    call_text,
    ensure_dir,
    gather_limited,
    load_personas,
    save_json,
)
from src.benchmarks.judge import judge_personagym_response  # noqa: E402
from src.benchmarks.persona_render import render_answer_style_hint, render_persona_context  # noqa: E402


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {
            "n_rows": 0,
            "persona_consistency": None,
            "behavior_plausibility": None,
            "cross_turn_coherence": None,
            "persona_score": None,
            "persona_score_by_domain": {},
        }

    for key in ["persona_consistency", "behavior_plausibility", "cross_turn_coherence"]:
        df[key] = pd.to_numeric(df[key], errors="coerce")

    persona_score = (
        df[["persona_consistency", "behavior_plausibility", "cross_turn_coherence"]]
        .mean(axis=1, skipna=True)
        .mul(20.0)
    )
    by_domain = (
        pd.DataFrame({"domain": df["domain"], "score": persona_score})
        .groupby("domain")["score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return {
        "n_rows": int(len(df)),
        "persona_consistency": float(df["persona_consistency"].mean()),
        "behavior_plausibility": float(df["behavior_plausibility"].mean()),
        "cross_turn_coherence": float(df["cross_turn_coherence"].mean()),
        "persona_score": float(persona_score.mean()),
        "persona_score_by_domain": {
            row["domain"]: {"score": float(row["mean"]), "n": int(row["count"])}
            for _, row in by_domain.iterrows()
        },
    }


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    scenarios_path = Path(args.scenarios_path)
    if not scenarios_path.exists():
        raise FileNotFoundError(f"PersonaGym scenarios file not found: {scenarios_path}")
    with scenarios_path.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios_en_ru.json is empty or invalid.")

    personas_df = load_personas(args.synthetic_path, args.persona_sample, args.seed)
    personas = personas_df.to_dict(orient="records")
    scenarios = scenarios[: args.scenarios_per_persona]

    guard = QuotaGuard(max_calls=args.max_calls, max_runtime_min=args.max_runtime_min)

    async def worker(persona_idx: int, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        persona_tag = f"persona_{persona_idx}"
        persona_context = render_persona_context(persona, args.locale)
        history: List[str] = []
        rows: List[Dict[str, Any]] = []

        for scenario in scenarios:
            prompt_text = scenario["prompt_ru"] if args.locale == "ru" else scenario["prompt_en"]
            response_prompt = (
                f"{persona_context}\n\n"
                "Scenario:\n"
                f"{prompt_text}\n\n"
                f"{render_answer_style_hint(args.locale)}"
            )
            try:
                response_text = await call_text(response_prompt, guard=guard, temperature=0.2)
                judge = await judge_personagym_response(
                    persona_context=persona_context,
                    scenario=prompt_text,
                    response_text=response_text,
                    response_history=history,
                    locale=args.locale,
                    guard=guard,
                )
                scores = judge["combined_scores"]
                error = ""
            except QuotaExceeded as exc:
                response_text = ""
                scores = {
                    "persona_consistency": 1.0,
                    "behavior_plausibility": 1.0,
                    "cross_turn_coherence": 1.0,
                }
                judge = {"llm_scores": {}, "rule_score": 1.0, "reasoning": ""}
                error = f"quota_exceeded: {exc}"
            except Exception as exc:
                response_text = ""
                scores = {
                    "persona_consistency": 1.0,
                    "behavior_plausibility": 1.0,
                    "cross_turn_coherence": 1.0,
                }
                judge = {"llm_scores": {}, "rule_score": 1.0, "reasoning": ""}
                error = str(exc)

            row = {
                "persona_id": persona_tag,
                "scenario_id": scenario.get("id"),
                "domain": scenario.get("domain", "default"),
                "response_text": response_text,
                "persona_consistency": scores["persona_consistency"],
                "behavior_plausibility": scores["behavior_plausibility"],
                "cross_turn_coherence": scores["cross_turn_coherence"],
                "llm_reasoning": judge.get("reasoning", ""),
                "llm_scores": judge.get("llm_scores", {}),
                "rule_score": judge.get("rule_score", 1.0),
                "locale": args.locale,
                "error": error,
            }
            rows.append(row)
            history.append(response_text)
        return rows

    persona_rows = await gather_limited(personas, worker, concurrency=args.concurrency)
    flat_rows = [item for part in persona_rows for item in part]
    out_df = pd.DataFrame(flat_rows)

    out_dir = ensure_dir(Path(args.out_dir))
    csv_path = out_dir / "judgments.csv"
    metrics_path = out_dir / "metrics.json"
    out_df.to_csv(csv_path, index=False)

    metrics = compute_metrics(out_df)
    metrics.update(
        {
            "benchmark": "personagym_protocol_compatible",
            "locale": args.locale,
            "persona_sample": int(args.persona_sample),
            "scenarios_per_persona": int(args.scenarios_per_persona),
            "partial": bool((out_df["error"] != "").any()),
            "protocol_note": "Protocol-compatible PersonaGym, not official leaderboard run.",
            "quota": guard.snapshot(),
        }
    )
    save_json(metrics, metrics_path)
    return {"judgments_path": str(csv_path), "metrics_path": str(metrics_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-compatible PersonaGym benchmark.")
    parser.add_argument(
        "--scenarios-path",
        default="data/benchmark_external/personagym/scenarios_en_ru.json",
        help="Path to localized PersonaGym scenarios.",
    )
    parser.add_argument("--synthetic-path", default=None, help="Path to synthetic personas CSV.")
    parser.add_argument("--locale", choices=["en", "ru"], default="en")
    parser.add_argument("--persona-sample", type=int, default=15)
    parser.add_argument("--scenarios-per-persona", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--max-runtime-min", type=float, default=180.0)
    parser.add_argument("--out-dir", default="outputs/benchmarks/suite/latest/personagym")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    print(f"Saved judgments: {result['judgments_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

