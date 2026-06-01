"""CLI entrypoint for the qualitative meta-agent benchmark.

Usage examples:
  python -m src.meta_agent.benchmark.main run --suite command_following --output benchmark_reports/cmd
  python -m src.meta_agent.benchmark.main run --suite all --review interactive

  # Score previously saved results (interactive)
  python -m src.meta_agent.benchmark.main score --input benchmark_reports/run1 --suite all
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import BenchmarkCase, BenchmarkResult, CaseScore
from .report import generate_report
from .runner import BenchmarkRunner
from .suites import (
    get_command_following_suite,
    get_data_extraction_suite,
    get_analysis_correctness_suite,
    get_graph_artifact_quality_suite,
    get_session_context_behavior_suite,
)

from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger("benchmark.main")

SUITE_CHOICES = ["all", "command_following", "data_extraction", "analysis_correctness", "graph_artifact_quality", "session_context_behavior"]


def get_all_suites() -> dict[str, list[BenchmarkCase]]:
    return {
        "command_following": get_command_following_suite(),
        "data_extraction": get_data_extraction_suite(),
        "analysis_correctness": get_analysis_correctness_suite(),
        "graph_artifact_quality": get_graph_artifact_quality_suite(),
        "session_context_behavior": get_session_context_behavior_suite(),
    }


def save_results(results: list[Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ if hasattr(r, "__dict__") else r for r in results], f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {json_path}")


def _interactive_score(result: Any, case: BenchmarkCase | None = None) -> CaseScore:
    """Prompt user for 0.0-1.0 score and optional comment."""
    print("\n" + "=" * 60)
    print(f"Case: {result.case_id}")
    print(f"Prompt: {result.prompt}")
    if case:
        print(f"Section: {case.section}")
        print(f"Expected: {case.expected_answer[:200]}...")
        print(f"Success criteria: {case.success_criteria}")
    print(f"Latency: {result.latency_ms:.1f} ms | Error: {result.error}")
    print(f"Outputs: {len(result.outputs)} | Artifacts: {result.artifact_count}")

    if result.outputs:
        last = result.outputs[-1]
        print(f"Last output preview: {str(last)[:1000]}...")
    print("=" * 60)

    while True:
        try:
            score_str = input("Enter score (0.0 - 1.0): ").strip()
            score = float(score_str)
            if 0.0 <= score <= 1.0:
                break
        except ValueError:
            pass
        print("Invalid. Please enter a number between 0.0 and 1.0.")

    comment = input("Optional comment (press enter to skip): ").strip() or None
    return CaseScore(
        case_id=result.case_id,
        score=score,
        comment=comment,
        scored_by="human",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _flatten_cases_run(suites: dict[str, list[BenchmarkCase]], suite_name: str) -> list[BenchmarkCase]:
    """Cases that correspond to benchmark results ordering (suite order preserved)."""
    if suite_name == "all":
        return [c for suite_cases in suites.values() for c in suite_cases]
    return suites.get(suite_name, [])


def score_saved_results(input_dir: str, suite_name: str | None = None, output: str | None = None) -> None:
    """Load saved benchmark_results.json and run interactive scoring + report generation."""
    in_path = Path(input_dir)
    results_path = in_path / "benchmark_results.json"
    if not results_path.exists():
        print(f"No benchmark_results.json found in {in_path}")
        return

    raw = json.loads(results_path.read_text(encoding="utf-8"))
    results = [BenchmarkResult(**r) for r in raw]

    out_path = Path(output) if output else in_path
    out_path.mkdir(parents=True, exist_ok=True)

    # Build case context if suite provided
    case_map: dict[str, BenchmarkCase] = {}
    section_map: dict[str, str] = {}
    if suite_name:
        suites = get_all_suites()
        for s in suites.values():
            for c in s:
                case_map[c.id] = c
        if suite_name != "all":
            section_map = {c.id: c.section for c in suites.get(suite_name, [])}
        else:
            section_map = {c.id: c.section for c in _flatten_cases_run(suites, "all")}

    scores: list[CaseScore] = []
    for res in results:
        case = case_map.get(res.case_id)
        sc = _interactive_score(res, case)
        scores.append(sc)

    # Save scores
    scores_path = out_path / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump([s.__dict__ for s in scores], f, indent=2, ensure_ascii=False)
    print(f"\nScores saved to {scores_path}")
    avg = sum(s.score for s in scores) / max(1, len(scores))
    print(f"Overall average score: {avg:.3f}")

    summary = generate_report(results, scores=scores, output_dir=out_path, section_map=section_map)
    print(
        f"Report saved to {out_path / 'benchmark_report.json'} "
        f"and {out_path / 'benchmark_report.md'} "
        f"(overall score {summary['overall_score']:.3f})"
    )


async def run_benchmark(suite_name: str, output: str, review: str) -> None:
    suites = get_all_suites()
    runner = BenchmarkRunner()

    if suite_name == "all":
        results = await runner.run_all(suites)
    else:
        cases = suites.get(suite_name, [])
        if not cases:
            print(f"Unknown suite: {suite_name}")
            return
        results = await runner.run_suite(cases)

    out_path = Path(output)
    cases_run = _flatten_cases_run(suites, suite_name)
    section_map = {c.id: c.section for c in cases_run}
    save_results(results, out_path)

    scores: list[CaseScore] = []
    if review == "interactive":
        # Map case_id to case for context
        case_map: dict[str, BenchmarkCase] = {}
        for s in suites.values():
            for c in s:
                case_map[c.id] = c

        for res in results:
            case = case_map.get(res.case_id)
            if res.error:
                print(f"\n[ERROR] {res.case_id}: {res.error}")
                scores.append(CaseScore(case_id=res.case_id, score=0.0, comment=f"Error: {res.error}", scored_by="auto"))
                continue
            sc = _interactive_score(res, case)
            scores.append(sc)

        # Save scores
        scores_path = out_path / "scores.json"
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump([s.__dict__ for s in scores], f, indent=2, ensure_ascii=False)
        print(f"\nScores saved to {scores_path}")
        avg = sum(s.score for s in scores) / max(1, len(scores))
        print(f"Overall average score: {avg:.3f}")

        summary = generate_report(results, scores=scores, output_dir=out_path, section_map=section_map)
        print(
            f"Report saved to {out_path / 'benchmark_report.json'} "
            f"and {out_path / 'benchmark_report.md'} "
            f"(overall score {summary['overall_score']:.3f})"
        )
    else:
        print("Run completed (unscored results saved). Use --review interactive for manual scoring.")

        summary = generate_report(results, scores=[], output_dir=out_path, section_map=section_map)
        print(
            f"Report saved to {out_path / 'benchmark_report.json'} "
            f"and {out_path / 'benchmark_report.md'} "
            f"(success rate {summary['success_rate']:.0%}; scores not applied — run with --review interactive to score)"
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Silence noisy third-party loggers (chat completions, internal debug, etc.)
    for noisy in ("httpx", "httpcore", "openai", "langchain", "langsmith", "yandex", "sgr_agent_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Meta-Agent Qualitative Benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Execute benchmark suites")
    run_p.add_argument("--suite", default="all", choices=SUITE_CHOICES)
    run_p.add_argument("--output", default="benchmark_reports/latest")
    run_p.add_argument("--review", default="none", choices=["none", "interactive"])

    score_p = sub.add_parser("score", help="Interactively score saved benchmark_results.json")
    score_p.add_argument("--input", required=True, help="Directory with benchmark_results.json")
    score_p.add_argument("--suite", default=None, choices=SUITE_CHOICES, help="Provide case context (expected answers, criteria)")
    score_p.add_argument("--output", default=None, help="Where to write scores.json + report (default: same as --input)")

    args = parser.parse_args()
    if args.cmd == "run":
        asyncio.run(run_benchmark(args.suite, args.output, args.review))
    elif args.cmd == "score":
        score_saved_results(args.input, args.suite, args.output)


if __name__ == "__main__":
    main()
