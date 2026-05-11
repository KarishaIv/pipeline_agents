import asyncio
import logging
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

try:
    from src.agents.multi_agent import MultiAgentSystem
except ImportError:
    MultiAgentSystem = None
from src.agents.survey_agent import MultiAgentReasoner
from src.agents.structured_survey_reasoner import StructuredSurveyReasoner
from src.core.storage import StorageManager
from src.core.visualization import EmotionalVisualizer
from config import *

logger = logging.getLogger(__name__)


def _json_safe_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Values safe for json.dumps (parquet row dicts may contain numpy ndarrays)."""
    safe: Dict[str, Any] = {}
    for key, value in profile.items():
        if key == "embedding":
            continue
        if isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


class SimulationManager:
    """
    Менеджер для параллельных симуляций
    """

    def __init__(self,
                 out_dir: Path = Path("outputs"),
                 concurrency: int = 4,
                 timeout: float = 120.0,
                 run_retries: int = 1,
                 executor_workers: int = 4,
                 agent_mode: str = "credit",
                 decision_mode: str = "direct",
                 survey_mode: str = "legacy",
                 news_context: Optional[Dict[str, Any]] = None,
                 survey_questions: List[str] = None,
                 visualize_sample: int = 0,
                 summary_visualize: bool = True):
        self.out_dir = out_dir
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        # legacy flag kept for backward compatibility
        self.visualize = False
        self.run_retries = run_retries
        self.executor = ThreadPoolExecutor(max_workers=executor_workers)
        self.agent_mode = agent_mode
        self.decision_mode = decision_mode
        self.survey_mode = survey_mode
        self.news_context = news_context
        self.survey_questions = survey_questions or []
        self.visualize_sample = max(0, int(visualize_sample))
        self.summary_visualize = bool(summary_visualize)

    async def _run_single(self, profile: Dict[str, Any], steps: int, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Запускает одну симуляцию с таймаутом и повторными попытками
        """
        last_exc = None
        persona_name = profile.get('name', 'unknown')
        
        for attempt in range(1, self.run_retries + 2):
            try:
                if self.agent_mode == "survey":
                    logger.debug(f"[Persona:{persona_name}] Запуск опросного режима")
                    if self.survey_mode == "structured":
                        reasoner = StructuredSurveyReasoner(
                            profile,
                            world_context=self.news_context,
                        )
                    else:
                        reasoner = MultiAgentReasoner(profile)
                    survey_results = await reasoner.answer_survey_questions(self.survey_questions)
                    return {
                        "profile": profile,
                        "survey_responses": survey_results, 
                        "mode": "survey",
                        "survey_mode": self.survey_mode,
                        "total_questions": len(self.survey_questions),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    logger.debug(f"[Persona:{persona_name}] Запуск кредитного режима, шагов: {steps}")
                    mas = MultiAgentSystem(
                        profile,
                        steps=steps,
                        decision_mode=self.decision_mode,
                        news_context=self.news_context,
                    )
                    coro = mas.run_simulation()
                    result = await asyncio.wait_for(coro, timeout=self.timeout)
                    return result
            except Exception as e:
                logger.warning(f"[Persona:{persona_name}] Сбой симуляции, попытка {attempt}: {e}")
                last_exc = e
                await asyncio.sleep(0.5 * attempt)

        logger.error(f"[Persona:{persona_name}] Все попытки симуляции провалились")
        raise last_exc

    async def _worker(self, i: int, profile: Dict[str, Any], steps: int, out_dir: Path):
        name = profile.get("name", "unknown")
        run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{name}_{run_ts}_{i}"
        
        async with self._sem:
            started = time.time()
            logger.info(f"[Run:{run_id}] START")
            
            try:
                result = await self._run_single(profile, steps)

                if self.agent_mode == "survey":
                    logger.debug(f"[Run:{run_id}] Сохранение результатов опроса")
                    await StorageManager.save_survey_results(
                        result, out_dir, run_id, self.survey_questions
                    )
                else:
                    logger.debug(f"[Run:{run_id}] Сохранение результатов кредитной симуляции")
                    await StorageManager.save_result_stream(result, out_dir, run_id)
                    # keep run_id for optional post-run visualizations
                    result["_run_id"] = run_id
                
                elapsed = time.time() - started
                logger.info(f"[Run:{run_id}] END за {elapsed:.1f}s")
                return result
                
            except Exception as e:
                logger.error(f"[Run:{run_id}] FAILED: {e}")
                failure = {
                    "run_id": run_id,
                    "profile": _json_safe_profile(profile),
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await StorageManager.save_json_async(failure, out_dir / f"{run_id}_error.json")
                return failure

    async def run_many(self, profiles: Iterable[Dict[str, Any]], steps: int = 3, out_subdir: str = "results") -> List[Dict[str, Any]]:
        out_dir = self.out_dir / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        profiles_list = list(profiles)
        logger.info(f"Запуск {len(profiles_list)} симуляций, параллелизм: {self.concurrency}")

        tasks = []
        results = []

        for i, p in enumerate(profiles_list):
            task = asyncio.create_task(self._worker(i, p, steps, out_dir))
            tasks.append(task)

        completed = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            completed += 1
            

            if completed % max(1, len(profiles_list) // 2) == 0 or completed % 2 == 0:
                logger.info(f"Прогресс симуляций: {completed}/{len(profiles_list)} ({completed/len(profiles_list)*100:.1f}%)")

            await StorageManager.append_line_async(
                out_dir / "summary_stream.log", 
                str({"timestamp": datetime.utcnow().isoformat(), "completed": completed, "total": len(profiles_list)})
            )
        
        if self.agent_mode == "survey":
            await self._generate_survey_summary(results, out_dir)
        else:
            await self._generate_visualizations(results, out_dir)
        
        successful = len([r for r in results if 'error' not in r])
        logger.info(f"Симуляции завершены: {successful}/{len(results)} успешных")
            
        return results

    async def _generate_survey_summary(self, results: List[Dict[str, Any]], out_dir: Path):
        """Генерирует сводную статистику по опросу"""
        logger.info("Генерация сводной статистики опроса")
        
        question_stats = {}
        
        for result in results:
            responses = result.get("survey_responses", [])
            for response in responses:
                question_text = response.get("question", "Unknown")
                full_state = response.get("full_state", {})
                
                final_decision = full_state.get("final_decision", {})
                if isinstance(final_decision, dict):
                    agreement = final_decision.get("decision", False)
                else:
                    agreement = getattr(final_decision, "decision", False)
                
                if question_text not in question_stats:
                    question_stats[question_text] = {
                        "question": question_text,
                        "agree": 0, 
                        "disagree": 0, 
                        "total": 0,
                        "agree_percent": 0.0,
                        "disagree_percent": 0.0,
                        "confidence_sum": 0.0, 
                        "confidence_avg": 0.0
                    }
                
                if agreement:
                    question_stats[question_text]["agree"] += 1
                else:
                    question_stats[question_text]["disagree"] += 1
                
                question_stats[question_text]["total"] += 1
                
                if isinstance(final_decision, dict):
                    confidence = final_decision.get("confidence", 0.5)
                else:
                    confidence = getattr(final_decision, "confidence", 0.5)
                
                question_stats[question_text]["confidence_sum"] += confidence
        
        for stats in question_stats.values():
            total = stats["total"]
            if total > 0:
                stats["agree_percent"] = round((stats["agree"] / total) * 100, 2)
                stats["disagree_percent"] = round((stats["disagree"] / total) * 100, 2)
                stats["confidence_avg"] = round(stats["confidence_sum"] / total, 3)
        
        question_stats_list = list(question_stats.values())
        question_stats_list.sort(key=lambda x: x["agree_percent"], reverse=True)
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_respondents": len(results),
            "total_questions": len(self.survey_questions),
            "question_statistics": question_stats_list,
            "agent_mode": "survey",
            "survey_mode": self.survey_mode,
            "overall_agreement_percent": round(
                sum(stats["agree"] for stats in question_stats_list) / 
                (sum(stats["total"] for stats in question_stats_list) or 1) * 100, 2
            )
        }
        
        await StorageManager.save_json_async(summary, out_dir / "survey_summary.json")
        
        simplified_stats = {}
        for item in question_stats_list:
            simplified_stats[item["question"]] = {
                "agree": item["agree"],
                "disagree": item["disagree"], 
                "agree_percent": item["agree_percent"],
                "disagree_percent": item["disagree_percent"],
                "confidence_avg": item["confidence_avg"]
            }
        
        simplified_summary = {
            "timestamp": summary["timestamp"],
            "total_respondents": summary["total_respondents"],
            "overall_agreement_percent": summary["overall_agreement_percent"],
            "question_statistics": simplified_stats,
            "agent_mode": "survey",
            "survey_mode": self.survey_mode,
        }
        
        await StorageManager.save_json_async(
            simplified_summary, 
            out_dir / "survey_summary_simplified.json"
        )
        
        logger.info(f"СВОДКА ОПРОСА - {len(results)} респондентов")
        logger.info("=" * 80)
        for i, item in enumerate(question_stats_list, 1):
            logger.info(
                f"{i:2d}. {item['agree_percent']:5.1f}% согласны "
                f"({item['agree']:3d}/{item['total']:3d}) "
                f"[доверие: {item['confidence_avg']:.3f}] - "
                f"{item['question'][:60]}..."
            )
        
        logger.info(f"Общее согласие: {summary['overall_agreement_percent']:.1f}%")
        logger.info(f"Сводка опроса сохранена: {len(results)} респондентов, {len(question_stats)} вопросов")

    async def _generate_visualizations(self, results: List[Dict[str, Any]], out_dir: Path) -> None:
        if not self.visualize or self.agent_mode != "credit":
            return

        successful = [r for r in results if isinstance(r, dict) and r.get("session_history")]
        if not successful:
            logger.warning("Нет успешных результатов для визуализации")
            return

        if self.summary_visualize:
            summary_viz = self._build_summary_viz_data(successful)
            if summary_viz:
                await EmotionalVisualizer.plot_async(summary_viz, out_dir / "summary_emotions.html")

        if self.visualize_sample > 0:
            sample = successful[: self.visualize_sample]
            for idx, result in enumerate(sample, 1):
                run_id = result.get("_run_id") or result.get("run_id") or f"sample_{idx}"
                viz_data = self._build_person_viz_data(result)
                if not viz_data:
                    continue
                save_path = out_dir / run_id / f"{run_id}_emotions.html"
                await EmotionalVisualizer.plot_async(viz_data, save_path)

    @staticmethod
    def _build_person_viz_data(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        history = result.get("session_history", [])
        profile_name = result.get("profile", {}).get("name", "Persona")

        viz_data = []
        for record in history:
            step = record.get("step")
            emotions = record.get("emotional_state", {}) or {}
            for emotion, value in emotions.items():
                try:
                    intensity = float(value)
                except (TypeError, ValueError):
                    continue
                viz_data.append(
                    {
                        "agent": profile_name,
                        "emotion": emotion,
                        "step": step,
                        "intensity": intensity,
                    }
                )
        return viz_data

    @staticmethod
    def _build_summary_viz_data(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bucket: Dict[tuple, List[float]] = {}
        for result in results:
            for record in result.get("session_history", []):
                step = record.get("step")
                emotions = record.get("emotional_state", {}) or {}
                for emotion, value in emotions.items():
                    try:
                        intensity = float(value)
                    except (TypeError, ValueError):
                        continue
                    bucket.setdefault((emotion, step), []).append(intensity)

        summary = []
        for (emotion, step), values in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                continue
            mean = float(np.nanmean(arr))
            p25 = float(np.nanpercentile(arr, 25))
            p75 = float(np.nanpercentile(arr, 75))
            summary.extend(
                [
                    {"agent": "mean", "emotion": emotion, "step": step, "intensity": mean},
                    {"agent": "p25", "emotion": emotion, "step": step, "intensity": p25},
                    {"agent": "p75", "emotion": emotion, "step": step, "intensity": p75},
                ]
            )
        return summary
