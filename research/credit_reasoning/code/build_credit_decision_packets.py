from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _packet_id(path: Path, packet: Dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(packet, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    run_name = path.parent.name or path.stem
    return f"{run_name}_{digest}"


def _first_present(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _extract_packet(path: Path, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    profile = _first_present(data, ["profile", "persona", "user_profile", "agent_profile"])
    goal = _first_present(data, ["goal", "user_goal", "task_goal"])
    session_history = _first_present(data, ["session_history", "history", "steps"])
    final_emotional_state = _first_present(data, ["final_emotional_state", "emotional_state"])
    push_info = _first_present(data, ["push_info", "push", "notification"])
    reaction = _first_present(data, ["reaction", "agent_reaction"])
    baseline_decision = _first_present(data, ["baseline_decision", "decision", "final_decision"])

    if not isinstance(profile, dict) or not isinstance(goal, dict):
        return None
    if not isinstance(session_history, list):
        session_history = []
    if not isinstance(final_emotional_state, dict):
        final_emotional_state = {}
    if not isinstance(push_info, dict):
        push_info = {}
    if not isinstance(reaction, dict):
        reaction = {}
    if not isinstance(baseline_decision, dict):
        baseline_decision = {}

    packet = {
        "source_run_path": str(path),
        "profile": profile,
        "goal": goal,
        "session_history": session_history,
        "final_emotional_state": final_emotional_state,
        "push_info": push_info,
        "reaction": reaction,
        "baseline_decision": baseline_decision,
    }
    packet["packet_id"] = _packet_id(path, packet)
    return packet


def build_packets(input_glob: str, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for raw_path in sorted(glob.glob(input_glob, recursive=True)):
        path = Path(raw_path)
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        packet = _extract_packet(path, data)
        if packet:
            candidates.append(packet)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if sample_size > 0:
        candidates = candidates[:sample_size]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen credit decision packets from old full_run.json files.")
    parser.add_argument("--input-glob", default="outputs/**/full_run.json")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="research/credit_reasoning/decision_packets")
    args = parser.parse_args()

    packets = build_packets(args.input_glob, args.sample_size, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "decision_packets.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for packet in packets:
            f.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "input_glob": args.input_glob,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "n_packets": len(packets),
        "target_audience_counts": {},
    }
    for packet in packets:
        audience = str((packet.get("profile") or {}).get("target_audience_name") or "unknown")
        summary["target_audience_counts"][audience] = summary["target_audience_counts"].get(audience, 0) + 1
    with (out_dir / "decision_packets_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

