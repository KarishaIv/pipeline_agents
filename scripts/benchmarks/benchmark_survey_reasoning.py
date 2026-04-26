#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.agents.structured_survey_reasoner import StructuredSurveyReasoner
from src.agents.survey_agent import MultiAgentReasoner
from src.benchmarks.common import QuotaGuard
from src.benchmarks.judge import judge_survey_reasoning
def _ensure_llm_env() -> None:
    api_key = os.getenv("YANDEX_API_KEY") or os.getenv("OPENAI_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if api_key and folder_id:
        return
    raise SystemExit(
        "Missing LLM credentials. Set YANDEX_API_KEY (preferred) or OPENAI_API_KEY, and YANDEX_FOLDER_ID in the shell or in .env before running benchmark_survey_reasoning.py."
    )


def _load_news_context(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_profiles(pattern: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No profile files matched pattern: {pattern}")
    profiles: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_profile_path"] = str(path)
        profiles.append(payload)
    return profiles


def _sample_items(items: Sequence[Any], n: int, seed: int) -> List[Any]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    chosen = list(items)
    rng.shuffle(chosen)
    return chosen[:n]


def _round_robin_sample(items: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[(str(item.get("target_audience_name") or "unknown"), item.get("question_index"))].append(dict(item))

    keys = list(grouped.keys())
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(grouped[key])

    selected: List[Dict[str, Any]] = []
    active = list(keys)
    while active and len(selected) < n:
        next_active: List[Tuple[str, Any]] = []
        for key in active:
            if len(selected) >= n:
                break
            bucket = grouped[key]
            if bucket:
                selected.append(bucket.pop())
            if bucket:
                next_active.append(key)
        active = next_active
    return selected[:n]


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _profile_label(profile: Dict[str, Any]) -> str:
    return (
        str(profile.get("name") or "").strip()
        or str(profile.get("persona_id") or "").strip()
        or Path(str(profile.get("_profile_path") or "profile_unknown")).stem
    )


def _flatten_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(row)
    flat["voice_stances"] = _dump_json(flat.get("voice_stances"))
    flat["trace_voices"] = _dump_json(flat.get("trace_voices"))
    flat["score_breakdown"] = _dump_json(flat.get("score_breakdown"))
    return flat


def _write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil((len(ordered) - 1) * q))))
    return float(ordered[idx])


def _extract_legacy_voice_stances(result: Dict[str, Any]) -> Dict[str, str]:
    voice_stances: Dict[str, str] = {}
    for voice_name in ["emotional", "rational", "social", "ideological"]:
        history = result.get(f"{voice_name}_history", [])
        if history:
            voice_stances[voice_name] = "present"
    return voice_stances


def _extract_prediction(
    mode: str,
    profile: Dict[str, Any],
    question: str,
    question_index: int,
    repeat_idx: int,
    result: Dict[str, Any],
    latency_seconds: float,
) -> Dict[str, Any]:
    final_decision = result.get("final_decision") or {}
    trace = result.get("trace") or {}
    reasoning = str(final_decision.get("reasoning") or "")
    decision = bool(final_decision.get("decision", False))
    confidence = float(final_decision.get("confidence", 0.0) or 0.0)
    prompt_chars = int(trace.get("prompt_chars") or 0)
    generation_count = int(result.get("generation_count") or 0)
    llm_calls = int(trace.get("llm_calls") or 0)
    if not llm_calls:
        llm_calls = generation_count + (1 if final_decision else 0)
    voice_stances = final_decision.get("voice_stances") or {}
    if not voice_stances and mode == "legacy":
        voice_stances = _extract_legacy_voice_stances(result)

    return {
        "survey_mode": mode,
        "profile_name": _profile_label(profile),
        "persona_id": _profile_label(profile),
        "target_audience_name": profile.get("target_audience_name"),
        "question_index": question_index,
        "question": question,
        "repeat_idx": repeat_idx,
        "decision": decision,
        "confidence": round(confidence, 4),
        "reasoning": reasoning,
        "reasoning_chars": len(reasoning),
        "generation_count": generation_count,
        "llm_calls": llm_calls,
        "prompt_chars": prompt_chars,
        "latency_seconds": round(latency_seconds, 4),
        "conflict_summary": final_decision.get("conflict_summary"),
        "voice_stances": voice_stances,
        "trace_voices": result.get("trace_voices") or {},
        "score_breakdown": final_decision.get("score_breakdown") or {},
        "news_context_used": bool(final_decision.get("news_context_used") or trace.get("world_context_used")),
        "question_topic": (final_decision.get("score_breakdown") or {}).get("question_topic") or trace.get("question_topic"),
        "resolver_used": bool((final_decision.get("score_breakdown") or {}).get("resolver_used") or trace.get("resolver_used")),
        "profile_path": profile.get("_profile_path"),
    }


def _error_prediction(
    survey_mode: str,
    profile: Dict[str, Any],
    question: str,
    question_index: int,
    repeat_idx: int,
    error: Exception,
    latency_seconds: float,
) -> Dict[str, Any]:
    return {
        "survey_mode": survey_mode,
        "profile_name": _profile_label(profile),
        "persona_id": _profile_label(profile),
        "target_audience_name": profile.get("target_audience_name"),
        "question_index": question_index,
        "question": question,
        "repeat_idx": repeat_idx,
        "decision": False,
        "confidence": 0.0,
        "reasoning": "",
        "reasoning_chars": 0,
        "generation_count": 0,
        "llm_calls": 0,
        "prompt_chars": 0,
        "latency_seconds": round(latency_seconds, 4),
        "conflict_summary": None,
        "voice_stances": {},
        "trace_voices": {},
        "score_breakdown": {},
        "news_context_used": False,
        "question_topic": None,
        "resolver_used": False,
        "profile_path": profile.get("_profile_path"),
        "error": f"{type(error).__name__}: {error}",
    }


async def _run_single(
    survey_mode: str,
    profile: Dict[str, Any],
    question: str,
    question_index: int,
    repeat_idx: int,
    news_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    persona_id = _profile_label(profile)
    started = time.perf_counter()
    if survey_mode == "structured":
        reasoner = StructuredSurveyReasoner(profile, world_context=news_context)
    elif survey_mode == "legacy":
        reasoner = MultiAgentReasoner(profile)
    else:
        raise ValueError(f"Unsupported survey_mode: {survey_mode}")
    result = await reasoner.run(
        scenario=question,
        persona_id=persona_id,
    )
    latency = time.perf_counter() - started
    return _extract_prediction(
        mode=survey_mode,
        profile=profile,
        question=question,
        question_index=question_index,
        repeat_idx=repeat_idx,
        result=result,
        latency_seconds=latency,
    )


async def _run_predictions(
    survey_modes: Sequence[str],
    profiles: Sequence[Dict[str, Any]],
    questions: Sequence[str],
    repeats: int,
    concurrency: int,
    news_context: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    jobs: List[Tuple[str, Dict[str, Any], str, int, int]] = []
    for survey_mode in survey_modes:
        for profile in profiles:
            for question_index, question in enumerate(questions):
                for repeat_idx in range(repeats):
                    jobs.append((survey_mode, profile, question, question_index, repeat_idx))

    results: List[Optional[Dict[str, Any]]] = [None] * len(jobs)

    async def worker(job_index: int, job: Tuple[str, Dict[str, Any], str, int, int]) -> None:
        survey_mode, profile, question, question_index, repeat_idx = job
        async with sem:
            started = time.perf_counter()
            try:
                results[job_index] = await _run_single(
                    survey_mode=survey_mode,
                    profile=profile,
                    question=question,
                    question_index=question_index,
                    repeat_idx=repeat_idx,
                    news_context=news_context,
                )
            except Exception as exc:
                results[job_index] = _error_prediction(
                    survey_mode=survey_mode,
                    profile=profile,
                    question=question,
                    question_index=question_index,
                    repeat_idx=repeat_idx,
                    error=exc,
                    latency_seconds=time.perf_counter() - started,
                )

    tasks = [asyncio.create_task(worker(job_index, job)) for job_index, job in enumerate(jobs)]
    for task in asyncio.as_completed(tasks):
        await task
    return [row for row in results if row is not None]


def _group_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return row["survey_mode"], row["persona_id"], int(row["question_index"])


async def _run_judging(
    profiles: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    judge_sample: int,
    judge_seed: int,
    locale: str,
) -> List[Dict[str, Any]]:
    if judge_sample <= 0:
        return []

    profile_by_id = {_profile_label(profile): profile for profile in profiles}
    grouped_candidates: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in predictions:
        if row.get("error"):
            continue
        key = (row["persona_id"], int(row["question_index"]))
        existing = grouped_candidates.get(key)
        if existing is None or int(row["repeat_idx"]) < int(existing["repeat_idx"]):
            grouped_candidates[key] = row

    selected_cases = _round_robin_sample(list(grouped_candidates.values()), min(judge_sample, len(grouped_candidates)), judge_seed)
    selected_keys = {(row["persona_id"], int(row["question_index"])) for row in selected_cases}

    judge_inputs: List[Dict[str, Any]] = []
    for row in predictions:
        if row.get("error"):
            continue
        key = (row["persona_id"], int(row["question_index"]))
        if key not in selected_keys:
            continue
        existing = next((item for item in judge_inputs if item["survey_mode"] == row["survey_mode"] and item["persona_id"] == row["persona_id"] and int(item["question_index"]) == int(row["question_index"])), None)
        if existing is None or int(row["repeat_idx"]) < int(existing["repeat_idx"]):
            if existing is not None:
                judge_inputs.remove(existing)
            judge_inputs.append(row)

    guard = QuotaGuard(max_calls=max(1, len(judge_inputs)), max_runtime_min=60.0)
    results: List[Dict[str, Any]] = []
    for row in judge_inputs:
        profile = profile_by_id.get(row["persona_id"])
        if not profile:
            continue
        answer = {
            "decision": row.get("decision"),
            "confidence": row.get("confidence"),
            "reasoning": row.get("reasoning", ""),
            "conflict_summary": row.get("conflict_summary"),
            "voice_stances": row.get("voice_stances"),
            "question_topic": row.get("question_topic"),
            "resolver_used": row.get("resolver_used"),
            "score_breakdown": row.get("score_breakdown"),
        }
        judgement = await judge_survey_reasoning(
            profile=profile,
            question=str(row["question"]),
            answer=answer,
            locale=locale,
            guard=guard,
        )
        combined = judgement["combined_scores"]
        results.append(
            {
                "survey_mode": row["survey_mode"],
                "persona_id": row["persona_id"],
                "target_audience_name": row.get("target_audience_name"),
                "question_index": row["question_index"],
                "question": row["question"],
                "question_topic": row.get("question_topic"),
                "persona_alignment": combined["persona_alignment"],
                "reasoning_nuance": combined["reasoning_nuance"],
                "decision_coherence": combined["decision_coherence"],
                "judge_reasoning": judgement.get("reasoning", ""),
                "raw_judgement": json.dumps(judgement, ensure_ascii=False, sort_keys=True),
            }
        )
    return results


def _mode_metrics(rows: Sequence[Dict[str, Any]], judge_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    successful_rows = [row for row in rows if not row.get("error")]
    error_count = len(rows) - len(successful_rows)
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in successful_rows:
        grouped[_group_key(row)].append(row)

    stability = []
    confidence_stds = []
    for group_rows in grouped.values():
        decisions = [bool(row["decision"]) for row in group_rows]
        confidences = [float(row["confidence"]) for row in group_rows]
        stability.append(1.0 if len(set(decisions)) == 1 else 0.0)
        confidence_stds.append(_std(confidences))

    prompt_values = [int(row["prompt_chars"]) for row in successful_rows if int(row.get("prompt_chars") or 0) > 0]
    latency_values = [float(row["latency_seconds"]) for row in successful_rows]

    mixed_stance_cases = 0
    all_uncertain_cases = 0
    resolver_used_cases = 0
    for row in successful_rows:
        stances = row.get("voice_stances") or {}
        if isinstance(stances, str):
            try:
                stances = json.loads(stances)
            except json.JSONDecodeError:
                stances = {}
        stance_values = list(stances.values())
        if stance_values:
            if all(stance == "uncertain" for stance in stance_values):
                all_uncertain_cases += 1
            elif len(set(stance_values)) > 1:
                mixed_stance_cases += 1
        if bool(row.get("resolver_used")):
            resolver_used_cases += 1

    metrics = {
        "total_predictions": len(rows),
        "success_count": len(successful_rows),
        "error_count": error_count,
        "unique_profile_question_pairs": len(grouped),
        "decision_rate": round(_mean([1.0 if row["decision"] else 0.0 for row in successful_rows]), 4),
        "stability_rate": round(_mean(stability), 4),
        "mean_confidence_std": round(_mean(confidence_stds), 5),
        "avg_confidence": round(_mean([float(row["confidence"]) for row in successful_rows]), 4),
        "avg_reasoning_chars": round(_mean([float(row["reasoning_chars"]) for row in successful_rows]), 2),
        "avg_generation_count": round(_mean([float(row["generation_count"]) for row in successful_rows]), 2),
        "avg_llm_calls": round(_mean([float(row["llm_calls"]) for row in successful_rows]), 2),
        "avg_prompt_chars": round(_mean(prompt_values), 2) if prompt_values else 0.0,
        "prompt_chars_coverage": round(len(prompt_values) / len(successful_rows), 4) if successful_rows else 0.0,
        "avg_latency_seconds": round(_mean(latency_values), 4),
        "median_latency_seconds": round(_percentile(latency_values, 0.5), 4),
        "p95_latency_seconds": round(_percentile(latency_values, 0.95), 4),
        "conflict_rate": round(
            _mean([1.0 if str(row.get("conflict_summary") or "").strip() else 0.0 for row in successful_rows]),
            4,
        ),
        "mixed_stance_rate": round(mixed_stance_cases / len(successful_rows), 4) if successful_rows else 0.0,
        "all_uncertain_rate": round(all_uncertain_cases / len(successful_rows), 4) if successful_rows else 0.0,
        "resolver_used_rate": round(resolver_used_cases / len(successful_rows), 4) if successful_rows else 0.0,
        "news_context_used": any(bool(row.get("news_context_used")) for row in successful_rows),
    }
    question_topics = sorted({row.get("question_topic") for row in rows if row.get("question_topic")})
    if question_topics:
        metrics["question_topics"] = question_topics
    if judge_rows:
        metrics.update(
            {
                "judge_sample_cases": len(judge_rows),
                "persona_alignment_mean": round(_mean([float(row["persona_alignment"]) for row in judge_rows]), 4),
                "reasoning_nuance_mean": round(_mean([float(row["reasoning_nuance"]) for row in judge_rows]), 4),
                "decision_coherence_mean": round(_mean([float(row["decision_coherence"]) for row in judge_rows]), 4),
            }
        )
    return metrics


def _build_comparison(metrics_by_mode: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if "legacy" not in metrics_by_mode or "structured" not in metrics_by_mode:
        return {}
    legacy = metrics_by_mode["legacy"]
    structured = metrics_by_mode["structured"]
    keys = [
        "decision_rate",
        "stability_rate",
        "mean_confidence_std",
        "avg_confidence",
        "avg_reasoning_chars",
        "avg_generation_count",
        "avg_llm_calls",
        "avg_prompt_chars",
        "avg_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "conflict_rate",
        "mixed_stance_rate",
        "all_uncertain_rate",
        "resolver_used_rate",
        "persona_alignment_mean",
        "reasoning_nuance_mean",
        "decision_coherence_mean",
    ]
    comparison = {}
    for key in keys:
        if key not in legacy or key not in structured:
            continue
        comparison[key] = {
            "legacy": legacy[key],
            "structured": structured[key],
            "delta_structured_minus_legacy": round(structured[key] - legacy[key], 5),
        }
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark legacy vs structured survey reasoning on fixed personas and questions.",
    )
    parser.add_argument(
        "--profiles-glob",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "profile_*.json"),
        help="Glob pattern for profile JSON files.",
    )
    parser.add_argument(
        "--questions-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "survey_questions.json"),
        help="Path to survey questions JSON. Defaults to project survey data.",
    )
    parser.add_argument(
        "--profile-sample",
        type=int,
        default=3,
        help="How many profiles to sample. 0 means use all matched profiles.",
    )
    parser.add_argument(
        "--question-sample",
        type=int,
        default=5,
        help="How many questions to sample. 0 means use all survey questions.",
    )
    parser.add_argument(
        "--survey-modes",
        nargs="+",
        default=["legacy", "structured"],
        choices=["legacy", "structured"],
        help="Which survey modes to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=2, help="How many repeats per profile/question/mode.")
    parser.add_argument("--concurrency", type=int, default=2, help="Async concurrency for benchmark jobs.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--judge-sample", type=int, default=0, help="How many unique persona/question cases to judge.")
    parser.add_argument("--judge-seed", type=int, default=42, help="Seed for judge case sampling.")
    parser.add_argument("--locale", type=str, default="ru", help="Judge locale.")
    parser.add_argument("--news-context-path", type=str, default=None, help="Optional news-context JSON for structured mode.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "benchmarks" / "survey_reasoning" / "default_run"),
        help="Directory where benchmark artifacts will be written.",
    )
    return parser.parse_args()


def _load_questions(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"Survey questions file is empty: {path}")
    return questions


async def _main() -> None:
    args = parse_args()
    _ensure_llm_env()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = _load_profiles(args.profiles_glob)
    questions = _load_questions(Path(args.questions_path))
    profiles = _sample_items(profiles, args.profile_sample, args.seed)
    questions = _sample_items(questions, args.question_sample, args.seed + 1)
    news_context = _load_news_context(Path(args.news_context_path)) if args.news_context_path else None

    manifest = {
        "profiles_glob": args.profiles_glob,
        "selected_profiles": [
            {
                "persona_id": profile.get("persona_id"),
                "name": _profile_label(profile),
                "target_audience_name": profile.get("target_audience_name"),
                "profile_path": profile.get("_profile_path"),
            }
            for profile in profiles
        ],
        "selected_questions": [{"index": idx, "question": question} for idx, question in enumerate(questions)],
        "survey_modes": args.survey_modes,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "news_context_path": args.news_context_path,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions = await _run_predictions(
        survey_modes=args.survey_modes,
        profiles=profiles,
        questions=questions,
        repeats=args.repeats,
        concurrency=args.concurrency,
        news_context=news_context,
    )
    _write_csv((_flatten_prediction(row) for row in predictions), out_dir / "predictions.csv")

    judge_rows = await _run_judging(
        profiles=profiles,
        predictions=predictions,
        judge_sample=args.judge_sample,
        judge_seed=args.judge_seed,
        locale=args.locale,
    )
    _write_csv(judge_rows, out_dir / "judge_results.csv")

    metrics_by_mode: Dict[str, Dict[str, Any]] = {}
    for survey_mode in args.survey_modes:
        mode_rows = [row for row in predictions if row["survey_mode"] == survey_mode]
        mode_judge_rows = [row for row in judge_rows if row["survey_mode"] == survey_mode]
        metrics = _mode_metrics(mode_rows, mode_judge_rows)
        metrics.update(
            {
                "survey_mode": survey_mode,
                "profile_sample": len(profiles),
                "question_sample": len(questions),
                "repeats": args.repeats,
                "news_context_used": bool(news_context) and survey_mode == "structured",
            }
        )
        metrics_by_mode[survey_mode] = metrics

    metrics_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics_by_mode": metrics_by_mode,
        "comparison": _build_comparison(metrics_by_mode),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
