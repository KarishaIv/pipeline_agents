from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
for path in (str(CURRENT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from credit_reasoning_agent import CreditReasoningAgent


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_packets(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_predictions(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "packet_id",
        "packet_index",
        "repeat_idx",
        "source_run_path",
        "target_audience_name",
        "goal_description",
        "decision_mode",
        "will_take_credit",
        "probability_score",
        "reasoning",
        "emotional_factors",
        "voice_stances",
        "conflict_summary",
        "prompt_chars",
        "prompt_char_counts",
        "llm_calls",
        "latency_seconds",
        "goal_intent",
        "narrative_mode",
        "context_summary",
        "score_breakdown",
        "guardrails_applied",
        "trace_voices",
        "baseline_decision",
        "news_snapshot_id",
        "news_context_summary",
        "news_signal_summary",
    ]
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(fields))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*fields, *extra_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_cell(row.get(field, "")) for field in writer.fieldnames or []})


def _metrics(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["packet_id"])].append(row)

    decision_rates = [1.0 if row.get("will_take_credit") else 0.0 for row in rows]
    probabilities = [float(row.get("probability_score") or 0.0) for row in rows]
    stability_values: List[float] = []
    probability_stds: List[float] = []
    for packet_rows in grouped.values():
        decisions = [bool(row.get("will_take_credit")) for row in packet_rows]
        majority = max(set(decisions), key=decisions.count)
        stability_values.append(sum(1 for value in decisions if value == majority) / len(decisions))
        packet_probs = [float(row.get("probability_score") or 0.0) for row in packet_rows]
        probability_stds.append(pstdev(packet_probs) if len(packet_probs) > 1 else 0.0)

    return {
        "decision_mode": args.decision_mode,
        "narrative_mode": args.narrative_mode,
        "news_context_used": bool(args.news_context_path),
        "n_packets": len(grouped),
        "n_predictions": len(rows),
        "decision_rate": round(mean(decision_rates), 4) if decision_rates else 0.0,
        "avg_probability_score": round(mean(probabilities), 4) if probabilities else 0.0,
        "stability_rate": round(mean(stability_values), 4) if stability_values else 0.0,
        "mean_probability_std": round(mean(probability_stds), 4) if probability_stds else 0.0,
        "avg_calls_per_prediction": round(mean(float(row.get("llm_calls") or 0.0) for row in rows), 4) if rows else 0.0,
        "avg_latency_seconds": round(mean(float(row.get("latency_seconds") or 0.0) for row in rows), 4) if rows else 0.0,
        "avg_prompt_chars": round(mean(float(row.get("prompt_chars") or 0.0) for row in rows), 2) if rows else 0.0,
        "avg_reasoning_chars": round(mean(len(str(row.get("reasoning") or "")) for row in rows), 2) if rows else 0.0,
        "guardrails_applied": sorted(
            {
                guardrail
                for row in rows
                for guardrail in (row.get("guardrails_applied") or [])
            }
        ),
    }


async def _run(args: argparse.Namespace) -> None:
    packets = _load_packets(Path(args.decision_packets))
    if args.packet_sample:
        packets = packets[: args.packet_sample]
    news_context = _load_json(args.news_context_path)
    agent = CreditReasoningAgent(
        decision_mode=args.decision_mode,
        narrative_mode=args.narrative_mode,
        news_context=news_context,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(packet_index: int, repeat_idx: int, packet: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            result = await agent.decide(packet)
        result.update(
            {
                "packet_id": packet.get("packet_id"),
                "packet_index": packet_index,
                "repeat_idx": repeat_idx,
                "source_run_path": packet.get("source_run_path"),
                "target_audience_name": (packet.get("profile") or {}).get("target_audience_name"),
                "goal_description": (packet.get("goal") or {}).get("goal_description"),
            }
        )
        return result

    tasks = [
        run_one(packet_index, repeat_idx, packet)
        for packet_index, packet in enumerate(packets)
        for repeat_idx in range(args.repeats)
    ]
    rows = await asyncio.gather(*tasks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(out_dir / "predictions.csv", rows)
    metrics = _metrics(rows, args)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reconstructed credit reasoning benchmark.")
    parser.add_argument("--decision-packets", default="research/credit_reasoning/decision_packets/decision_packets.jsonl")
    parser.add_argument("--decision-mode", choices=["direct", "compact_debate"], default="compact_debate")
    parser.add_argument("--narrative-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--news-context-path")
    parser.add_argument("--packet-sample", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out-dir", default="research/credit_reasoning/results/reconstructed_smoke")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

