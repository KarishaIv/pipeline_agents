import asyncio
import logging
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.agents.survey_agent import MultiAgentReasoner
from src.core.storage import StorageManager
from config import *

logger = logging.getLogger(__name__)

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
                 survey_questions: List[str] = None,
                 world_contexts: Dict[str, dict] = None):
        self.out_dir = out_dir
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.run_retries = run_retries
        self.executor = ThreadPoolExecutor(max_workers=executor_workers)
        self.survey_questions = survey_questions or []
        self.world_contexts = world_contexts or {}  # {ta_name → news_context}

    async def _run_single(self, profile: Dict[str, Any], steps: int, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Запускает одну симуляцию с таймаутом и повторными попытками
        """
        last_exc = None
        persona_name = profile.get('name', 'unknown')
        
        for attempt in range(1, self.run_retries + 2):
            try:
                logger.debug(f"[Persona:{persona_name}] Запуск опросного режима")

                ta_name = profile.get("target_audience_name", "")
                world_ctx = self.world_contexts.get(ta_name, {})
                reasoner = MultiAgentReasoner(profile, world_context=world_ctx)
                survey_results = await reasoner.answer_survey_questions(self.survey_questions)
                return {
                    "profile": profile,
                    "survey_responses": survey_results, 
                    "total_questions": len(self.survey_questions),
                    "timestamp": datetime.utcnow().isoformat()
                }
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
                result['run_id'] = run_id
                
                elapsed = time.time() - started
                logger.info(f"[Run:{run_id}] END за {elapsed:.1f}s")
                return result
                
            except Exception as e:
                logger.error(f"[Run:{run_id}] FAILED: {e}")
                failure = {
                    "run_id": run_id, 
                    "profile": profile, 
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
        
        successful = len([r for r in results if 'error' not in r])
        logger.info(f"Симуляции завершены: {successful}/{len(results)} успешных")
            
        return results
