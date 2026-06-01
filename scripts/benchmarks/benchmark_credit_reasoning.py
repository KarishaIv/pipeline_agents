#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.agents.credit_reasoning_agent import CreditReasoningAgent
from src.agents.decision_agent import DecisionAgent
from src.benchmarks.common import QuotaGuard
from src.benchmarks.judge import judge_credit_reasoning
from src.schemas.deliberation_schema import DecisionPacket


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _ensure_llm_env() -> None:
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if api_key and folder_id:
        return
    raise SystemExit(
        "Missing LLM credentials. Set YANDEX_API_KEY and YANDEX_FOLDER_ID in the shell or in .env before running benchmark_credit_reasoning.py."
    )


def _load_packets(path: Path) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packet = DecisionPacket(**json.loads(line))
            packets.append(_dump_model(packet))
    if not packets:
        raise ValueError(f"No packets found in {path}")
    return packets


def _load_news_context(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _packet_group(packet: Dict[str, Any]) -> Tuple[str, str]:
    profile = packet.get("profile", {})
    goal = packet.get("goal", {})
    audience = str(profile.get("target_audience_name") or "unknown")
    goal_description = str(goal.get("goal_description") or "unknown")
    return audience, goal_description


def _round_robin_sample(items: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(items):
        return list(items)

    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[_packet_group(item)].append(dict(item))

    keys = list(groups.keys())
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    selected: List[Dict[str, Any]] = []
    active = list(keys)
    while active and len(selected) < n:
        next_active: List[Tuple[str, str]] = []
        for key in active:
            if len(selected) >= n:
                break
            bucket = groups[key]
            if bucket:
                selected.append(bucket.pop())
            if bucket:
                next_active.append(key)
        active = next_active

    if len(selected) < n:
        remainder: List[Dict[str, Any]] = []
        for bucket in groups.values():
            remainder.extend(bucket)
        rng.shuffle(remainder)
        selected.extend(remainder[: max(0, n - len(selected))])

    return selected[:n]


def _flatten_prediction(record: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(record)
    row["voice_stances"] = json.dumps(row.get("voice_stances"), ensure_ascii=False, sort_keys=True)
    row["prompt_char_counts"] = json.dumps(row.get("prompt_char_counts"), ensure_ascii=False, sort_keys=True)
    row["trace_voices"] = json.dumps(row.get("trace_voices"), ensure_ascii=False, sort_keys=True)
    row["guardrails_applied"] = json.dumps(row.get("guardrails_applied"), ensure_ascii=False, sort_keys=True)
    row["context_summary"] = json.dumps(row.get("context_summary"), ensure_ascii=False, sort_keys=True)
    row["score_breakdown"] = json.dumps(row.get("score_breakdown"), ensure_ascii=False, sort_keys=True)
    row["news_context_summary"] = json.dumps(row.get("news_context_summary"), ensure_ascii=False, sort_keys=True)
    row["news_signal_summary"] = json.dumps(row.get("news_signal_summary"), ensure_ascii=False, sort_keys=True)
    return row


def _write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


async def _run_predictions(
    packets: Sequence[Dict[str, Any]],
    decision_mode: str,
    repeats: int,
    concurrency: int,
    news_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if decision_mode == "compact_debate":
        agent = CreditReasoningAgent()
    elif decision_mode == "direct":
        agent = DecisionAgent()
    else:
        raise ValueError(f"Unsupported decision mode: {decision_mode}")

    sem = asyncio.Semaphore(max(1, concurrency))
    jobs: List[Tuple[int, Dict[str, Any], int]] = []
    for packet_idx, packet in enumerate(packets):
        for repeat_idx in range(repeats):
            jobs.append((packet_idx, packet, repeat_idx))

    results: List[Dict[str, Any]] = [None] * len(jobs)

    async def worker(job_idx: int, packet_idx: int, packet: Dict[str, Any], repeat_idx: int) -> None:
        async with sem:
            started = time.perf_counter()
            if decision_mode == "compact_debate":
                decision, trace = await agent.make_final_decision_with_trace(
                    profile=packet["profile"],
                    persona_history=packet.get("session_history", []),
                    emotional_state=packet.get("final_emotional_state", {}),
                    push_info=packet.get("push_info", {}),
                    goal=packet.get("goal"),
                    reaction=packet.get("reaction"),
                    news_context=news_context,
                )
            else:
                decision, trace = await agent.make_final_decision_with_trace(
                    profile=packet["profile"],
                    persona_history=packet.get("session_history", []),
                    emotional_state=packet.get("final_emotional_state", {}),
                    push_info=packet.get("push_info", {}),
                    goal=packet.get("goal"),
                    reaction=packet.get("reaction"),
                )
            latency = time.perf_counter() - started
            decision_data = _dump_model(decision)
            results[job_idx] = {
                "packet_id": packet["packet_id"],
                "packet_index": packet_idx,
                "repeat_idx": repeat_idx,
                "source_run_path": packet["source_run_path"],
                "target_audience_name": packet.get("profile", {}).get("target_audience_name"),
                "goal_description": packet.get("goal", {}).get("goal_description"),
                "decision_mode": decision_data.get("decision_mode"),
                "will_take_credit": bool(decision_data.get("will_take_credit")),
                "probability_score": float(decision_data.get("probability_score", 0.0)),
                "reasoning": decision_data.get("reasoning", ""),
                "emotional_factors": decision_data.get("emotional_factors", ""),
                "voice_stances": decision_data.get("voice_stances"),
                "conflict_summary": decision_data.get("conflict_summary"),
                "prompt_chars": int(trace.get("prompt_chars", 0)),
                "prompt_char_counts": trace.get("prompt_char_counts", {}),
                "llm_calls": int(trace.get("llm_calls", 0)),
                "latency_seconds": round(latency, 4),
                "goal_intent": (trace.get("context_summary") or {}).get("goal_intent"),
                "news_snapshot_id": (trace.get("context_summary") or {}).get("news_snapshot_id"),
                "narrative_mode": trace.get("narrative_mode"),
                "context_summary": trace.get("context_summary", {}),
                "score_breakdown": trace.get("score_breakdown", {}),
                "guardrails_applied": trace.get("guardrails_applied", []),
                "news_context_summary": trace.get("news_context", {}),
                "news_signal_summary": trace.get("news_signals", {}),
                "trace_voices": trace.get("voices"),
                "baseline_decision": packet.get("baseline_decision", {}).get("will_take_credit"),
            }

    tasks = [
        asyncio.create_task(worker(job_idx, packet_idx, packet, repeat_idx))
        for job_idx, (packet_idx, packet, repeat_idx) in enumerate(jobs)
    ]
    for task in asyncio.as_completed(tasks):
        await task
    return results


async def _run_judging(
    packets: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    judge_sample: int,
    judge_seed: int,
    locale: str,
) -> List[Dict[str, Any]]:
    if judge_sample <= 0:
        return []

    selected_packets = _round_robin_sample(list(packets), min(judge_sample, len(packets)), judge_seed)
    prediction_by_packet = {}
    for row in predictions:
        if row["packet_id"] not in prediction_by_packet or row["repeat_idx"] < prediction_by_packet[row["packet_id"]]["repeat_idx"]:
            prediction_by_packet[row["packet_id"]] = row

    guard = QuotaGuard(max_calls=max(1, len(selected_packets)), max_runtime_min=60.0)
    results: List[Dict[str, Any]] = []
    for packet in selected_packets:
        decision_row = prediction_by_packet.get(packet["packet_id"])
        if not decision_row:
            continue
        decision_payload = {
            "decision_mode": decision_row.get("decision_mode"),
            "will_take_credit": decision_row.get("will_take_credit"),
            "probability_score": decision_row.get("probability_score"),
            "reasoning": decision_row.get("reasoning"),
            "emotional_factors": decision_row.get("emotional_factors"),
            "voice_stances": decision_row.get("voice_stances"),
            "conflict_summary": decision_row.get("conflict_summary"),
        }
        judgement = await judge_credit_reasoning(packet=packet, decision=decision_payload, locale=locale, guard=guard)
        combined = judgement["combined_scores"]
        results.append(
            {
                "packet_id": packet["packet_id"],
                "target_audience_name": packet.get("profile", {}).get("target_audience_name"),
                "goal_description": packet.get("goal", {}).get("goal_description"),
                "decision_mode": decision_row.get("decision_mode"),
                "persona_alignment": combined["persona_alignment"],
                "emotional_nuance": combined["emotional_nuance"],
                "decision_coherence": combined["decision_coherence"],
                "judge_reasoning": judgement.get("reasoning", ""),
                "raw_judgement": json.dumps(judgement, ensure_ascii=False, sort_keys=True),
            }
        )
    return results


def _aggregate_metrics(
    packets: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    mode_runtime_seconds: float,
    judge_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    grouped_decisions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped_decisions[row["packet_id"]].append(row)

    stability_flags: List[float] = []
    probability_stds: List[float] = []
    for packet in packets:
        rows = sorted(grouped_decisions.get(packet["packet_id"], []), key=lambda item: item["repeat_idx"])
        if not rows:
            continue
        decision_values = [bool(row["will_take_credit"]) for row in rows]
        probability_values = [float(row["probability_score"]) for row in rows]
        stability_flags.append(1.0 if len(set(decision_values)) == 1 else 0.0)
        probability_stds.append(_std(probability_values))

    metrics = {
        "n_packets": len(packets),
        "n_predictions": len(predictions),
        "decision_mode": predictions[0]["decision_mode"] if predictions else None,
        "news_context_used": bool(predictions and predictions[0].get("news_snapshot_id")),
        "decision_rate": _mean([1.0 if row["will_take_credit"] else 0.0 for row in predictions]),
        "stability_rate": _mean(stability_flags),
        "mean_probability_std": _mean(probability_stds),
        "avg_reasoning_chars": _mean([float(len(str(row["reasoning"]))) for row in predictions]),
        "avg_prompt_chars": _mean([float(row["prompt_chars"]) for row in predictions]),
        "avg_calls_per_packet": _mean([float(row["llm_calls"]) for row in predictions]),
        "llm_narrative_rate": _mean([1.0 if row.get("narrative_mode") == "llm" else 0.0 for row in predictions]),
        "avg_latency_seconds": _mean([float(row["latency_seconds"]) for row in predictions]),
        "mode_runtime_seconds": round(mode_runtime_seconds, 3),
    }

    if judge_rows:
        metrics.update(
            {
                "judge_sample_packets": len(judge_rows),
                "persona_alignment_mean": _mean([float(row["persona_alignment"]) for row in judge_rows]),
                "emotional_nuance_mean": _mean([float(row["emotional_nuance"]) for row in judge_rows]),
                "decision_coherence_mean": _mean([float(row["decision_coherence"]) for row in judge_rows]),
            }
        )

    return metrics


async def async_main(args: argparse.Namespace) -> int:
    _ensure_llm_env()
    packets_path = PROJECT_ROOT / args.decision_packets
    out_dir = PROJECT_ROOT / args.out_dir
    if out_dir.name == "latest":
        out_dir = out_dir / args.decision_mode
    packets = _load_packets(packets_path)
    news_context = _load_news_context(PROJECT_ROOT / args.news_context_path) if args.news_context_path else None

    started = time.perf_counter()
    predictions = await _run_predictions(
        packets=packets,
        decision_mode=args.decision_mode,
        repeats=args.repeats,
        concurrency=args.concurrency,
        news_context=news_context,
    )
    mode_runtime_seconds = time.perf_counter() - started

    judge_rows = await _run_judging(
        packets=packets,
        predictions=predictions,
        judge_sample=args.judge_sample,
        judge_seed=args.judge_seed,
        locale=args.locale,
    )

    flat_predictions = [_flatten_prediction(row) for row in predictions]
    predictions_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    judge_path = out_dir / "judge_results.csv"
    _write_csv(flat_predictions, predictions_path)
    _write_csv(judge_rows, judge_path)

    metrics = _aggregate_metrics(packets, predictions, mode_runtime_seconds, judge_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark direct vs compact debate reasoning on frozen credit packets.")
    parser.add_argument(
        "--decision-packets",
        default="outputs/benchmarks/credit_reasoning_packets/decision_packets.jsonl",
        help="Path to decision_packets.jsonl",
    )
    parser.add_argument("--decision-mode", choices=["direct", "compact_debate"], default="direct")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs per packet.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent LLM calls.")
    parser.add_argument("--judge-sample", type=int, default=30, help="How many packets to judge with the quality judge.")
    parser.add_argument("--judge-seed", type=int, default=42, help="Seed for judge packet sampling.")
    parser.add_argument("--locale", default="ru", help="Judge locale.")
    parser.add_argument(
        "--out-dir",
        default="outputs/benchmarks/credit_reasoning/latest",
        help="Directory for predictions.csv, metrics.json and judge_results.csv",
    )
    parser.add_argument(
        "--news-context-path",
        default=None,
        help="Optional path to a news-context JSON snapshot to inject into compact_debate reasoning.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
