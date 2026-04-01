import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from config import NEWS_SYSTEM_PATH, PIPELINE_PATH, YANDEX_API_KEY
from llm import extract_demographics_with_llm

log = logging.getLogger(__name__)

_abs_pipeline = os.path.abspath(PIPELINE_PATH)
if _abs_pipeline not in sys.path:
    sys.path.insert(0, _abs_pipeline)

_news_enricher: Optional[object] = None
_pipeline_initialized = False


def _init_pipeline():
    global _pipeline_initialized
    if _pipeline_initialized:
        return

    # pipeline_agents/config.py должен быть загружен как 'config',
    # а не tg_bot/config.py, иначе все `from config import` внутри пайплайна сломаются
    import importlib.util
    _pipeline_config_path = os.path.join(_abs_pipeline, "config.py")
    spec = importlib.util.spec_from_file_location("config", _pipeline_config_path)
    pipeline_config_mod = importlib.util.module_from_spec(spec)
    sys.modules["config"] = pipeline_config_mod
    spec.loader.exec_module(pipeline_config_mod)

    from config_pipeline import set_yandex_config
    set_yandex_config(
        os.getenv("YANDEX_FOLDER_ID", ""),
        os.getenv("YANDEX_API_KEY", ""),
    )
    _pipeline_initialized = True


async def _get_news_enricher():
    global _news_enricher
    if _news_enricher is None:
        from src.news_enricher import NewsContextEnricher
        _news_enricher = NewsContextEnricher(
            news_system_path=os.path.abspath(NEWS_SYSTEM_PATH),
            iam_token=YANDEX_API_KEY,
        )
        await _news_enricher.initialize()
    return _news_enricher


def distribute_personas(total: int, ratios: List[int]) -> List[int]:
    """Распределяет total персон пропорционально ratios."""
    total_ratio = sum(ratios)
    counts = [max(1, round(total * r / total_ratio)) for r in ratios]
    diff = total - sum(counts)
    counts[0] += diff
    return counts


async def run_full_analysis(
    audiences: List[str],
    counts: List[int],
    question: str,
) -> dict:
    _init_pipeline()
    from src.orchestration import PipelineRunner

    news_contexts: Dict[str, dict] = {}
    evidence_list = []

    enricher = None
    try:
        enricher = await _get_news_enricher()
    except Exception as exc:
        log.warning("news enricher недоступен: %s", exc)

    for ta, count in zip(audiences, counts):
        news_ctx = {}
        if enricher:
            try:
                news_ctx = await enricher.get_news_context(question, ta)
                log.info("Новостной контекст для '%s' получен", ta)
            except Exception as exc:
                log.warning("Новостной контекст для '%s': %s", ta, exc)
        news_contexts[ta] = news_ctx

        demographics = await extract_demographics_with_llm(ta)
        log.info("Демография '%s': %s", ta, demographics)

        evidence_list.append({
            "target_audience_name": ta,
            "synthetic_size": count,
            "news_question": question,
            **demographics,
        })

    out_dir = os.path.join(_abs_pipeline, "outputs", "bot_runs")
    pipeline_config = {
        "evidence": evidence_list,
        "survey_questions": [question],
        "nemo_size": 1000,
        "output": out_dir,
        "simulation_steps": 1,
        "concurrency": 2,
        "timeout": 90.0,
        "ta_concurrency": 1,
        "agent_mode": "survey",
        "use_pgm": True,
        "ocean_flag": True,
    }

    runner = PipelineRunner(pipeline_config, news_enricher=_news_enricher)
    results = await runner.run()

    return {
        "results":       results,
        "news_contexts": news_contexts,
        "question":      question,
        "audiences":     audiences,
        "counts":        counts,
        "out_dir":       out_dir,
    }
