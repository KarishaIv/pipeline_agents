"""Benchmark runner that invokes the meta-agent graph or API."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.meta_agent.graph import MetaAgentGraphManager
from src.meta_agent.output_models import AgentOutput

from .cases import BenchmarkCase, BenchmarkResult

logger = logging.getLogger("benchmark.runner")

# Visual separators for terminal logs (72 cols — readable in narrow terminals)
_SUITE_LINE = "=" * 72
_CASE_LINE = "-" * 72


class BenchmarkRunner:
    """Executes benchmark cases against the meta-agent with performance tracking and session policy support."""

    def __init__(self, checkpoint_db_path: Path | None = None):
        self.manager = MetaAgentGraphManager(checkpoint_db_path=checkpoint_db_path)
        self._current_thread: str | None = None

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def run_case(self, case: BenchmarkCase, thread_id: str | None = None) -> BenchmarkResult:
        """Run one case, respect thread_policy, capture performance metrics."""
        start_perf = time.perf_counter()
        started_at = self._now_iso()

        # Determine thread for this case
        effective_thread = thread_id
        if case.thread_policy == "followup" and self._current_thread is not None:
            effective_thread = self._current_thread
        elif case.thread_policy == "fixed" and thread_id is None:
            effective_thread = self._current_thread

        try:
            result = await self.manager.invoke_graph_session(case.prompt, effective_thread)
            latency = (time.perf_counter() - start_perf) * 1000
            finished_at = self._now_iso()

            outputs = []
            for o in getattr(result, "outputs", []):
                if isinstance(o, AgentOutput):
                    outputs.append(o.model_dump())
                elif isinstance(o, dict):
                    outputs.append(o)

            artifacts = []
            if hasattr(result, "artifacts"):
                for a in result.artifacts:
                    artifacts.append(a.model_dump() if hasattr(a, "model_dump") else a)

            # Compute counts for performance / review
            type_counts: dict[str, int] = {}
            for o in outputs:
                t = o.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            # Update current thread for followup
            self._current_thread = result.thread_id

            return BenchmarkResult(
                case_id=case.id,
                thread_id=result.thread_id,
                prompt=case.prompt,
                outputs=outputs,
                artifacts=artifacts,
                latency_ms=latency,
                error=None,
                iterations=getattr(result, "iterations", 0),
                started_at=started_at,
                finished_at=finished_at,
                output_type_counts=type_counts,
                artifact_count=len(artifacts),
                metadata={"thread_policy": case.thread_policy},
            )
        except Exception as exc:
            latency = (time.perf_counter() - start_perf) * 1000
            finished_at = self._now_iso()
            return BenchmarkResult(
                case_id=case.id,
                thread_id=effective_thread or "error",
                prompt=case.prompt,
                latency_ms=latency,
                error=str(exc),
                started_at=started_at,
                finished_at=finished_at,
                output_type_counts={},
                artifact_count=0,
            )

    async def run_suite(self, cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
        """Run list of cases sequentially. Resets thread state per suite for determinism."""
        self._current_thread = None
        results: list[BenchmarkResult] = []
        logger.info(_SUITE_LINE)
        logger.info("Suite start — %d case(s)", len(cases))
        logger.info(_SUITE_LINE)
        for i, case in enumerate(cases, 1):
            if i > 1:
                logger.info(_CASE_LINE)
            logger.info("Running case %d/%d: %s", i, len(cases), case.id)
            res = await self.run_case(case)
            if res.error:
                logger.warning("Case %s failed: %s", case.id, res.error)
            else:
                logger.info("Completed %s (%.0f ms, %d outputs)", case.id, res.latency_ms, len(res.outputs))
            results.append(res)
        logger.info(_CASE_LINE)
        logger.info(_SUITE_LINE)
        logger.info("Suite finished — %d result(s)", len(results))
        logger.info(_SUITE_LINE)
        return results

    async def run_all(self, suites: dict[str, list[BenchmarkCase]]) -> list[BenchmarkResult]:
        """Run multiple named suites, preserving per-suite thread state."""
        all_results: list[BenchmarkResult] = []
        total_suites = len(suites)
        for s_idx, (name, cases) in enumerate(suites.items(), 1):
            if s_idx > 1:
                logger.info(_SUITE_LINE)
                logger.info(_SUITE_LINE)

            logger.info(_SUITE_LINE)
            logger.info("Suite %d/%d: %s — %d case(s)", s_idx, total_suites, name, len(cases))
            logger.info(_SUITE_LINE)

            self._current_thread = None  # fresh context per suite
            for i, case in enumerate(cases, 1):
                if i > 1:
                    logger.info(_CASE_LINE)
                logger.info("[%s] Case %d/%d: %s", name, i, len(cases), case.id)

                res = await self.run_case(case)
                res.metadata = {**res.metadata, "suite": name}

                if res.error:
                    logger.warning("[%s] %s error: %s", name, case.id, res.error)
                else:
                    logger.info("[%s] %s done (%.0f ms)", name, case.id, res.latency_ms)
                all_results.append(res)

            logger.info(_CASE_LINE)
            logger.info(_SUITE_LINE)
            logger.info("[%s] Suite done — %d result(s)", name, len(cases))
            logger.info(_SUITE_LINE)

        logger.info(_SUITE_LINE)
        logger.info("All suites finished — %d result(s) total", len(all_results))
        logger.info(_SUITE_LINE)
        return all_results
