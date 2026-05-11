"""Focused tests for the new qualitative benchmark (no gold)."""


from src.meta_agent.benchmark.cases import BenchmarkCase, CaseScore, BenchmarkResult
from src.meta_agent.benchmark.suites import get_command_following_suite, get_analysis_correctness_suite


def test_benchmark_case_has_qualitative_fields():
    case = BenchmarkCase(
        id="test1",
        prompt="Test prompt",
        section="command_following",
        description="desc",
        expected_answer="good answer",
    )
    assert case.section == "command_following"
    assert "good answer" in case.expected_answer
    d = case.to_dict()
    assert "rubric" in d and "success_criteria" in d


def test_case_score_roundtrip():
    sc = CaseScore(case_id="c1", score=0.85, comment="Solid", scored_by="human")
    assert 0.0 <= sc.score <= 1.0
    assert sc.scored_by == "human"


def test_suites_have_sections():
    cases = get_command_following_suite() + get_analysis_correctness_suite()
    sections = {c.section for c in cases}
    assert "command_following" in sections
    assert "analysis_correctness" in sections
    assert all(c.prompt for c in cases)


def test_benchmark_result_performance_fields():
    res = BenchmarkResult(
        case_id="r1",
        thread_id="t1",
        prompt="p",
        latency_ms=123.4,
        output_type_counts={"text": 1},
        artifact_count=0,
        started_at="2026-01-01T00:00:00Z",
    )
    assert res.latency_ms > 0
    assert "started_at" in res.__dataclass_fields__
