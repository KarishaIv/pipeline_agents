#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.deliberation_schema import DecisionPacket


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _packet_group(packet: Dict[str, Any]) -> Tuple[str, str]:
    profile = packet.get("profile", {})
    goal = packet.get("goal", {})
    audience = str(profile.get("target_audience_name") or "unknown")
    goal_description = str(goal.get("goal_description") or "unknown")
    return audience, goal_description


def _packet_id(source_path: Path) -> str:
    rel = _relative_path(source_path)
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{source_path.parent.name}_{digest}"


def _load_full_run(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_packet(run_data: Dict[str, Any], path: Path) -> Dict[str, Any] | None:
    required = [
        "profile",
        "goal",
        "session_history",
        "final_emotional_state",
        "push_info",
        "reaction",
        "decision",
    ]
    if any(key not in run_data for key in required):
        return None

    decision = run_data.get("decision", {})
    if not isinstance(decision, dict) or not isinstance(decision.get("will_take_credit"), bool):
        return None

    try:
        packet = DecisionPacket(
            packet_id=_packet_id(path),
            source_run_path=_relative_path(path),
            profile=run_data.get("profile", {}),
            goal=run_data.get("goal", {}),
            session_history=run_data.get("session_history", []),
            final_emotional_state=run_data.get("final_emotional_state", {}),
            push_info=run_data.get("push_info", {}),
            reaction=run_data.get("reaction", {}),
            baseline_decision=decision,
        )
    except Exception:
        return None
    return _dump_model(packet)


def _round_robin_sample(items: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(items):
        return list(items)

    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[_packet_group(item)].append(item)

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    for key in group_keys:
        rng.shuffle(groups[key])

    selected: List[Dict[str, Any]] = []
    active_keys = list(group_keys)
    while active_keys and len(selected) < n:
        next_keys: List[Tuple[str, str]] = []
        for key in active_keys:
            if len(selected) >= n:
                break
            bucket = groups[key]
            if bucket:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        active_keys = next_keys

    if len(selected) < n:
        remainder: List[Dict[str, Any]] = []
        for bucket in groups.values():
            remainder.extend(bucket)
        rng.shuffle(remainder)
        selected.extend(remainder[: max(0, n - len(selected))])

    return selected[:n]


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a frozen packet corpus for credit decision reasoning.")
    parser.add_argument("--input-glob", default="outputs/**/full_run.json", help="Glob for full_run.json files.")
    parser.add_argument("--sample-size", type=int, default=90, help="Number of packets to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--out-dir",
        default="outputs/benchmarks/credit_reasoning_packets",
        help="Output directory for decision_packets.jsonl and summary.json",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    paths = sorted(PROJECT_ROOT.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {args.input_glob}")

    valid_packets: List[Dict[str, Any]] = []
    invalid_count = 0
    for path in paths:
        try:
            run_data = _load_full_run(path)
        except Exception:
            invalid_count += 1
            continue
        packet = _validate_packet(run_data, path)
        if packet is None:
            invalid_count += 1
            continue
        valid_packets.append(packet)

    if not valid_packets:
        raise ValueError("No valid credit decision packets found.")

    sampled = _round_robin_sample(valid_packets, args.sample_size, args.seed)
    audience_counts = Counter(packet.get("profile", {}).get("target_audience_name") or "unknown" for packet in sampled)
    goal_counts = Counter(packet.get("goal", {}).get("goal_description") or "unknown" for packet in sampled)

    packets_path = out_dir / "decision_packets.jsonl"
    summary_path = out_dir / "decision_packets_summary.json"
    _write_jsonl(sampled, packets_path)
    summary = {
        "input_glob": args.input_glob,
        "seed": args.seed,
        "sample_size_requested": args.sample_size,
        "files_scanned": len(paths),
        "valid_packets": len(valid_packets),
        "invalid_or_skipped": invalid_count,
        "sampled_packets": len(sampled),
        "audience_counts": dict(sorted(audience_counts.items())),
        "goal_counts": dict(sorted(goal_counts.items())),
        "output_files": {
            "decision_packets": str(packets_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
