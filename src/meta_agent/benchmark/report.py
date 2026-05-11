"""Report generation for qualitative benchmark runs (with CaseScore support)."""

import json
from pathlib import Path
from typing import Any

from .cases import BenchmarkResult, CaseScore


def generate_report(
    results: list[BenchmarkResult],
    scores: list[CaseScore] | dict[str, float] | None = None,
    output_dir: Path | str = "benchmark_reports",
    section_map: dict[str, str] | None = None,  # case_id -> section
) -> dict[str, Any]:
    """Create rich summary report (JSON + MD) with scores, sections and performance."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if scores is None:
        scores = []
    if isinstance(scores, dict):
        scores = [CaseScore(case_id=k, score=v) for k, v in scores.items()]

    score_dict = {s.case_id: s.score for s in scores}
    comment_dict = {s.case_id: s.comment for s in scores}

    total = len(results)
    successes = sum(1 for r in results if r.error is None)
    avg_latency = sum(r.latency_ms for r in results) / max(1, total)
    overall_score = sum(s.score for s in scores) / max(1, len(scores)) if scores else 0.0

    # Group by section if possible
    section_scores: dict[str, list[float]] = {}
    for r in results:
        sec = (section_map or {}).get(r.case_id, "unknown")
        sc = score_dict.get(r.case_id, 0.0)
        section_scores.setdefault(sec, []).append(sc)

    section_avgs = {sec: sum(vs) / len(vs) for sec, vs in section_scores.items()}

    failures = [
        {"id": r.case_id, "prompt": r.prompt, "error": r.error}
        for r in results
        if r.error
    ]

    summary = {
        "total_cases": total,
        "success_rate": successes / max(1, total),
        "avg_latency_ms": avg_latency,
        "overall_score": overall_score,
        "section_averages": section_avgs,
        "per_case_scores": score_dict,
        "failures": failures,
        "performance": {
            "total_latency_ms": sum(r.latency_ms for r in results),
            "avg_iterations": sum(getattr(r, "iterations", 0) for r in results) / max(1, total),
        },
    }

    # JSON
    json_path = out / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Markdown
    md_path = out / "benchmark_report.md"
    md_lines = [
        "# Meta-Agent Qualitative Benchmark Report",
        "",
        f"**Total cases:** {total}",
        f"**Success rate:** {summary['success_rate']:.2%}",
        f"**Average latency:** {avg_latency:.1f} ms",
        f"**Overall score (0-1):** {overall_score:.3f}",
        "",
        "## Section Averages",
    ]
    for sec, avg in section_avgs.items():
        md_lines.append(f"- {sec}: {avg:.3f}")

    md_lines.append("\n## Per-Case Scores")
    for r in results:
        sc = score_dict.get(r.case_id, "N/A")
        comment = comment_dict.get(r.case_id) or ""
        status = "OK" if not r.error else "ERR"
        md_lines.append(f"- {r.case_id} [{status}] ({r.latency_ms:.0f}ms): {sc} {comment}")

    if failures:
        md_lines.append("\n## Failures")
        for f in failures:
            md_lines.append(f"- {f['id']}: {f['error']}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return summary
