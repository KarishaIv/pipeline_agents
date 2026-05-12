import json
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Any, Dict, Iterable, Optional, List
import logging
import pandas as pd
from pandas import DataFrame
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Асинхронный менеджер сохранения результатов.
    - save_json_async: сохраняет dict -> файл (в executor)
    - save_stream: дозапись логов/стримов
    - save_batch: сохраняет набор результатов агрегированно
    - append_parquet_async: добавляет данные в parquet файл
    """

    @staticmethod
    async def save_json_async(obj: Any, path: Path, ensure_ascii: bool = False, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        data = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
        await loop.run_in_executor(None, StorageManager._write_text_sync, path, data, "w", "utf-8")
        logger.debug(f"Saved JSON -> {path}")

    @staticmethod
    def _write_text_sync(path: Path, data: str, mode: str = "w", encoding: str = "utf-8"):
        with open(path, mode, encoding=encoding) as f:
            f.write(data)

    @staticmethod
    async def append_line_async(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, StorageManager._append_line_sync, path, line)

    @staticmethod
    def _append_line_sync(path: Path, line: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    async def save_batch(results: Iterable[Dict], out_dir: Path, prefix: str = "batch", per_file: int = 100):
        out_dir.mkdir(parents=True, exist_ok=True)
        results = list(results)
        total = len(results)
        for i in range(0, total, per_file):
            batch = results[i: i + per_file]
            filename = out_dir / f"{prefix}_{i // per_file + 1}.json"
            await StorageManager.save_json_async(batch, filename)
            logger.info(f"Saved batch {i // per_file + 1} ({len(batch)} items) -> {filename}")

    async def save_result_stream(result: Dict, out_dir: Path, run_id: str):
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        await StorageManager.save_json_async(result, run_dir / "full_run.json")

        summary = {
            "run_id": run_id,
            "profile_name": result.get("profile", {}).get("name"),
            "final_decision": result.get("decision", {}).get("will_take_credit") if isinstance(result.get("decision"), dict) else None,
            "decision_mode": result.get("decision_mode") or result.get("decision", {}).get("decision_mode"),
            "news_snapshot_id": result.get("news_context", {}).get("snapshot_id") if isinstance(result.get("news_context"), dict) else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        await StorageManager.append_line_async(out_dir / "runs_index.log", json.dumps(summary, ensure_ascii=False))


    @staticmethod
    async def save_survey_results(result: Dict, out_dir: Path, run_id: str, survey_questions: List[str]):
        run_dir = out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        profile = result.get("profile", {})
        survey_responses = result.get("survey_responses", [])
        
        await StorageManager.save_json_async(profile, run_dir / "profile.json")
        
        question_agreement = {}
        agreement_count = 0
        disagreement_count = 0
        
        for i, response in enumerate(survey_responses):
            question = response.get("question", f"question_{i}")
            question_index = response.get("question_index", i)
            full_state = response.get("full_state", {})
            
            safe_question_name = f"question_{question_index:03d}_{question[:30].replace(' ', '_').replace('?', '')}"
            
            question_file = run_dir / f"{safe_question_name}_full.json"
            await StorageManager.save_json_async({
                "question": question,
                "question_index": question_index,
                "profile": profile,
                "full_state": full_state,
                "timestamp": response.get("timestamp")
            }, question_file)
            
            final_decision = full_state.get("final_decision", {})
            if isinstance(final_decision, dict):
                agreement = final_decision.get("decision", False)
            else:
                agreement = getattr(final_decision, "decision", False)
            
            simplified_state = {
                "question": question,
                "question_index": question_index,
                "final_decision": final_decision,
                "agent_histories": {
                    "emotional": full_state.get("emotional_history", []),
                    "rational": full_state.get("rational_history", []),
                    "social": full_state.get("social_history", []),
                    "ideological": full_state.get("ideological_history", [])
                },
                "generation_count": full_state.get("generation_count", 0),
                "timestamp": response.get("timestamp")
            }
            
            simplified_file = run_dir / f"{safe_question_name}_simplified.json"
            await StorageManager.save_json_async(simplified_state, simplified_file)
            
            question_agreement[question] = "AGREE" if agreement else "DISAGREE"
            if agreement:
                agreement_count += 1
            else:
                disagreement_count += 1
        
        summary = {
            "run_id": run_id,
            "profile_name": profile.get("name", "unknown"),
            "persona_id": profile.get("persona_id", "unknown"),
            "survey_mode": result.get("survey_mode", "legacy"),
            "total_questions": len(survey_questions),
            "agreement_count": agreement_count,
            "disagreement_count": disagreement_count,
            "agreement_percent": round((agreement_count / len(survey_questions) * 100), 2) if survey_questions else 0,
            "question_agreement": question_agreement,
            "questions_by_index": [{"index": i, "question": q, "agreement": question_agreement.get(q, "UNKNOWN")} 
                                  for i, q in enumerate(survey_questions)],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await StorageManager.save_json_async(summary, run_dir / "survey_summary.json")
        
        await StorageManager.append_line_async(
            out_dir / "survey_index.log",
            json.dumps({
                "run_id": run_id,
                "profile_name": summary["profile_name"],
                "persona_id": summary["persona_id"],
                "survey_mode": summary["survey_mode"],
                "agreement_count": agreement_count,
                "disagreement_count": disagreement_count,
                "agreement_percent": summary["agreement_percent"],
                "timestamp": summary["timestamp"]
            }, ensure_ascii=False)
        )

    @staticmethod
    async def append_parquet_async(data: List[Dict] | pd.DataFrame, path: Path, check_columns: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, StorageManager._append_parquet_sync, data, path, check_columns)
        logger.debug(f'Saved Parquet -> {path}')

    @staticmethod
    def _append_parquet_sync(new_data: List[Dict] | pd.DataFrame, path: Path, check_columns: bool = True):
        if isinstance(new_data, list):
            if not new_data:
                return
            if not isinstance(new_data[0], dict):
                raise TypeError(f'Expected list[dict], got list[{type(new_data[0]).__name__}]')
            new_data = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            if new_data.empty:
                return
        else:
            raise TypeError(f'Expected list[dict] or DataFrame, got {type(new_data).__name__}')

        if path.exists():
            existing_data = pd.read_parquet(path)
            if check_columns:
                missing = set(existing_data.columns) ^ set(new_data.columns)
                assert not missing, (
                    f'Column mismatch: existing={sorted(existing_data.columns)}, '
                    f'new={sorted(new_data.columns)}, diff={sorted(missing)}'
                )
            new_data = pd.concat([existing_data, new_data], ignore_index=True)

        # Deduplicate by UUID, keeping the last occurrence (most recent write wins)
        if "UUID" in new_data.columns:
            before = len(new_data)
            new_data = new_data.drop_duplicates(subset=["UUID"], keep="last")
            dropped = before - len(new_data)
            if dropped:
                logger.debug("Dropped %d duplicate UUID(s) in %s", dropped, path)

        # Сохранение данных
        new_data.to_parquet(path, index=False)
