#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.common import (  # noqa: E402
    find_latest_synthetic_csv,
    git_commit_hash,
    load_json,
    save_json,
)


BENCHMARK_TO_SCRIPT = {
    "trait": "benchmark_trait.py",
    "personallm": "benchmark_personallm.py",
    "personagym": "benchmark_personagym.py",
    "emobench": "benchmark_emobench.py",
}


def _metric_to_score(benchmark: str, metrics: Dict[str, Any]) -> float:
    if benchmark == "trait":
        acc = metrics.get("accuracy")
        return float(acc * 100.0) if isinstance(acc, (int, float)) else 0.0
    if benchmark == "personallm":
        return float(metrics.get("personallm_composite_score", 0.0) or 0.0)
    if benchmark == "personagym":
        return float(metrics.get("persona_score", 0.0) or 0.0)
    if benchmark == "emobench":
        acc = metrics.get("accuracy", 0.0) or 0.0
        f1 = metrics.get("macro_f1", 0.0) or 0.0
        return float(((acc * 100.0) + (f1 * 100.0)) / 2.0)
    return 0.0


def _to_locales(raw: str) -> List[str]:
    if raw == "both":
        return ["en", "ru"]
    return [raw]


def _write_report(
    summary: Dict[str, Any],
    path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Synthetic Persona Benchmark Suite Report")
    lines.append("")
    lines.append("Protocol-compatible evaluation (not official leaderboard runs).")
    lines.append("")
    lines.append("## Run context")
    lines.append(f"- Timestamp (UTC): {summary['run_config']['timestamp_utc']}")
    lines.append(f"- Synthetic source: `{summary['run_config']['synthetic_path']}`")
    lines.append(f"- Benchmarks: {', '.join(summary['run_config']['benchmarks'])}")
    lines.append(f"- Locales: {', '.join(summary['run_config']['locales'])}")
    lines.append(f"- Partial run: {summary['partial']}")
    lines.append("")

    lines.append("## Scores by locale")
    for locale, payload in summary["scores"]["by_locale"].items():
        lines.append(f"### {locale}")
        lines.append("")
        lines.append("| Benchmark | Score |")
        lines.append("|---|---:|")
        for bench, score in payload["benchmark_scores"].items():
            lines.append(f"| {bench} | {score:.2f} |")
        lines.append(f"| combined | **{payload['combined_score']:.2f}** |")
        lines.append("")

    lines.append("## Combined")
    lines.append(f"- Final combined score: **{summary['scores']['combined_score']:.2f}**")
    lines.append("- Weights: TRAIT 25%, PersonaLLM 25%, PersonaGym 25%, EmoBench 25%.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("- Protocol-compatible adapters were used instead of official benchmark runners.")
    lines.append("- Open-ended scoring uses hybrid LLM-as-judge + rule checks.")
    lines.append("- Results depend on current LLM backend and prompt templates.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic persona benchmark suite.")
    parser.add_argument("--synthetic-path", default=None, help="Path to synthetic personas CSV.")
    parser.add_argument("--benchmarks", default="trait,personallm,personagym,emobench")
    parser.add_argument("--locale", choices=["en", "ru", "both"], default="both")
    parser.add_argument("--persona-sample", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=2500)
    parser.add_argument("--max-runtime-min", type=float, default=180.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--trait-items", type=int, default=20)
    parser.add_argument("--personagym-scenarios", type=int, default=8)
    parser.add_argument("--emobench-items", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/benchmarks/suite/{run_ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_path = Path(args.synthetic_path) if args.synthetic_path else find_latest_synthetic_csv()
    benchmarks = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    unknown = [b for b in benchmarks if b not in BENCHMARK_TO_SCRIPT]
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}")
    locales = _to_locales(args.locale)

    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_path": str(synthetic_path),
        "benchmarks": benchmarks,
        "locales": locales,
        "persona_sample": int(args.persona_sample),
        "seed": int(args.seed),
        "concurrency": int(args.concurrency),
        "max_calls_suite": int(args.max_calls),
        "max_runtime_min_suite": float(args.max_runtime_min),
        "git_commit": git_commit_hash(PROJECT_ROOT),
    }
    save_json(run_config, out_dir / "run_config.json")

    py = sys.executable
    scripts_dir = Path(__file__).resolve().parent
    partial = False
    results: Dict[str, Dict[str, Any]] = {}
    total_runs = len(locales) * len(benchmarks)
    completed_runs = 0
    remaining_calls = int(args.max_calls)
    suite_started_at = datetime.now(timezone.utc)

    for locale in locales:
        results[locale] = {}
        for benchmark in benchmarks:
            completed_runs += 1
            runs_left_including_current = max(1, total_runs - completed_runs + 1)
            elapsed_min = (datetime.now(timezone.utc) - suite_started_at).total_seconds() / 60.0
            remaining_runtime = max(0.0, float(args.max_runtime_min) - elapsed_min)
            run_call_budget = max(1, remaining_calls // runs_left_including_current)
            run_time_budget = max(1.0, remaining_runtime / runs_left_including_current)

            bench_out_dir = out_dir / locale / benchmark
            bench_out_dir.mkdir(parents=True, exist_ok=True)
            script_path = scripts_dir / BENCHMARK_TO_SCRIPT[benchmark]
            if remaining_calls <= 0 or remaining_runtime <= 0:
                partial = True
                results[locale][benchmark] = {
                    "command": [],
                    "return_code": -1,
                    "stdout_tail": "",
                    "stderr_tail": "Skipped due to suite budget exhaustion.",
                    "metrics_path": str(bench_out_dir / "metrics.json"),
                    "metrics": {},
                    "score": 0.0,
                }
                continue

            cmd = [
                py,
                str(script_path),
                "--synthetic-path",
                str(synthetic_path),
                "--locale",
                locale,
                "--persona-sample",
                str(args.persona_sample),
                "--seed",
                str(args.seed),
                "--concurrency",
                str(args.concurrency),
                "--max-calls",
                str(run_call_budget),
                "--max-runtime-min",
                str(run_time_budget),
                "--out-dir",
                str(bench_out_dir),
            ]
            if benchmark == "trait":
                cmd.extend(["--items-per-persona", str(args.trait_items)])
            elif benchmark == "personagym":
                cmd.extend(["--scenarios-per-persona", str(args.personagym_scenarios)])
            elif benchmark == "emobench":
                cmd.extend(["--items-per-persona", str(args.emobench_items)])

            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            metrics_path = bench_out_dir / "metrics.json"
            payload: Dict[str, Any] = {
                "command": cmd,
                "return_code": int(proc.returncode),
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "metrics_path": str(metrics_path),
            }
            if proc.returncode == 0 and metrics_path.exists():
                metrics = load_json(metrics_path, default={}) or {}
                payload["metrics"] = metrics
                payload["score"] = _metric_to_score(benchmark, metrics)
                used_calls = int(metrics.get("quota", {}).get("calls_made", run_call_budget))
            else:
                partial = True
                payload["metrics"] = {}
                payload["score"] = 0.0
                used_calls = run_call_budget
            if payload.get("metrics", {}).get("partial"):
                partial = True
            results[locale][benchmark] = payload
            remaining_calls = max(0, remaining_calls - max(0, used_calls))

    score_by_locale: Dict[str, Any] = {}
    locale_scores: List[float] = []
    for locale in locales:
        bench_scores = {bench: float(results[locale].get(bench, {}).get("score", 0.0)) for bench in benchmarks}
        combined = float(sum(bench_scores.values()) / len(bench_scores)) if bench_scores else 0.0
        locale_scores.append(combined)
        score_by_locale[locale] = {"benchmark_scores": bench_scores, "combined_score": combined}

    summary = {
        "run_config": run_config,
        "partial": partial,
        "results": results,
        "scores": {
            "by_locale": score_by_locale,
            "combined_score": float(sum(locale_scores) / len(locale_scores)) if locale_scores else 0.0,
            "weights": {"trait": 0.25, "personallm": 0.25, "personagym": 0.25, "emobench": 0.25},
        },
    }

    summary_path = out_dir / "suite_summary.json"
    report_path = out_dir / "suite_report.md"
    save_json(summary, summary_path)
    _write_report(summary, report_path)

    print(f"Saved run config: {out_dir / 'run_config.json'}")
    print(f"Saved suite summary: {summary_path}")
    print(f"Saved suite report: {report_path}")
    print(json.dumps(summary["scores"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
