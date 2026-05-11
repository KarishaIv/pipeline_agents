# Meta-Agent Qualitative Benchmark

Redesigned benchmark: no gold oracles or deterministic scorers.  
Cases include Russian prompts + qualitative expectations (description, expected_answer, success_criteria, rubric).  
Human (or future LLM) assigns 0.0–1.0 scores after execution. Runner captures performance (latency, iterations, artifact counts).

## Thematic Sections
- command_following (OOD, force commands)
- data_extraction (filters, collections)
- analysis_correctness (distributions, top-k)
- graph_artifact_quality (charts)
- session_context_behavior (multi-turn, follow-up)

## Quick Start (CLI)
```bash
# Run all suites and score interactively
python -m src.meta_agent.benchmark.main run --suite all --review interactive --output benchmark_reports/run1

# Run specific section (unscored results only)
python -m src.meta_agent.benchmark.main run --suite analysis_correctness --output benchmark_reports/analysis
```

## Python API
```python
import asyncio
from src.meta_agent.benchmark import (
    get_command_following_suite, BenchmarkRunner, generate_report
)

cases = get_command_following_suite()
runner = BenchmarkRunner()
results = asyncio.run(runner.run_suite(cases))
# ... collect CaseScore list manually or via CLI ...
report = generate_report(results, scores=..., output_dir="benchmark_reports")
```

## Output
- `benchmark_results.json` — raw execution + performance
- `scores.json` — per-case 0-1 scores + comments
- `benchmark_report.json` + `.md` — summary with section averages, overall score, failures

See `suites.py`, `cases.py`, `__main__.py` for details.
