"""Standalone script to load personas from parquet, run simulations (with optional world_context),
and save ALL data: questions.parquet, target_audiences.parquet, world_contexts.parquet, simulations.parquet.

Usage:
  python -m src.scripts.run_simulations --personas-parquet data_4_qdrant/personas.parquet \
      --evidence ./data/evidence.json --news-context-path ./context.json --output ./data_4_qdrant

Reuses SimulationManager + builders from orchestration.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from src.orchestration import (
    _build_question_rows,
    _build_target_audience_rows,
    _build_world_context_rows,
    _build_simulation_row,
)
from src.core.storage import StorageManager
from src.core.simulation_manager import SimulationManager, split_news_context_file_payload
from src.data_loading import load_evidence_from_json, load_survey_data
from config import set_yandex_credentials

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


async def run_simulations_from_parquet(
    personas_parquet: Path,
    evidence_path: Path,
    output_dir: Path,
    survey_questions: Optional[List[str]] = None,
    news_context_path: Optional[Path] = None,
    use_news_enricher: bool = False,
    simulation_steps: int = 1,
    concurrency: int = 4,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> List[Dict]:
    """Main entrypoint: load personas, run sims, save all 5 parquet files (questions, TAs, world_contexts, sims)."""
    if api_key:
        set_yandex_credentials(api_key, folder_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    data_4_qdrant = output_dir  # save parquets here for simplicity (matches test expectation)

    # Load evidence first (for synthetic_size + TA name filtering)
    evidence = load_evidence_from_json(str(evidence_path))

    # Load personas (drop embedding), then filter by target_audience_name + per-TA synthetic_size
    personas_df = pd.read_parquet(personas_parquet)
    if "embedding" in personas_df.columns:
        personas_df = personas_df.drop(columns=["embedding"])

    if evidence and "target_audience_name" in personas_df.columns:
        filtered_parts = []
        for item in evidence:
            ta_name = item.get("target_audience_name")
            size = item.get("synthetic_size", 0)
            if ta_name and size > 0:
                ta_df = personas_df[personas_df["target_audience_name"] == ta_name].head(size)
                if not ta_df.empty:
                    filtered_parts.append(ta_df)
        if filtered_parts:
            personas_df = pd.concat(filtered_parts, ignore_index=True)
        else:
            # fallback
            total = sum(item.get("synthetic_size", 0) for item in evidence)
            if total > 0:
                personas_df = personas_df.head(total)
    else:
        total = sum(item.get("synthetic_size", 0) for item in evidence)
        if total > 0:
            personas_df = personas_df.head(total)

    personas = [personas_df.iloc[i].to_dict() for i in range(len(personas_df))]
    logger.info(f"Loaded {len(personas)} personas from {personas_parquet} (filtered by evidence TA names + sizes)")
    if survey_questions is None:
        try:
            survey_questions = load_survey_data() or ["Default question for test?"]
        except Exception:
            survey_questions = ["Default question for test?"]

    # Load or enrich world_contexts using shared helper
    world_contexts: Dict[str, dict] = {}
    news_context = None
    if news_context_path and news_context_path.exists():
        with open(news_context_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            world_contexts, news_context = split_news_context_file_payload(loaded, evidence)
    # (enricher path omitted for minimal impl; can be added)

    # Run simulations
    manager = SimulationManager(
        out_dir=output_dir,
        concurrency=concurrency,
        timeout=timeout,
        run_retries=1,
        agent_mode="survey",
        survey_mode="structured",
        survey_questions=survey_questions,
        world_contexts=world_contexts or {},
        news_context=news_context,
    )

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    results = await manager.run_many(personas, steps=simulation_steps, out_subdir=f"sim_{timestamp}")

    # === SAVE ALL DATA (as required) ===
    questions_path = data_4_qdrant / "questions.parquet"
    personas_path = data_4_qdrant / "personas.parquet"  # already exists
    ta_path = data_4_qdrant / "target_audiences.parquet"
    wc_path = data_4_qdrant / "world_contexts.parquet"
    sims_path = data_4_qdrant / "simulations.parquet"

    # Questions
    questions_uuids = {q: __import__("src.utils").utils.get_uuid("questions", q) for q in survey_questions}
    q_rows = _build_question_rows(survey_questions)
    await StorageManager.append_parquet_async(q_rows, questions_path)

    # Target audiences (re-link to loaded personas)
    ta_rows = _build_target_audience_rows(evidence, personas_df)
    await StorageManager.append_parquet_async(ta_rows, ta_path, check_columns=False)

    # World contexts
    wc_rows = _build_world_context_rows(world_contexts)
    if wc_rows:
        await StorageManager.append_parquet_async(wc_rows, wc_path, check_columns=False)

    # Simulations (with world_context nullable support)
    wc_uuid_map = {r["ta_name"]: r["UUID"] for r in wc_rows} if wc_rows else {}
    sim_rows = []
    for result in results:
        if "survey_responses" not in result:
            continue
        for response in result["survey_responses"]:
            state = response["full_state"]
            sim_rows.append(_build_simulation_row(result, response, state, questions_uuids, wc_uuid_map))
    if sim_rows:
        await StorageManager.append_parquet_async(sim_rows, sims_path, check_columns=False)

    logger.info(f"✅ Saved questions, TAs, world_contexts, simulations to {data_4_qdrant}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulations from existing personas parquet and save all data")
    parser.add_argument('--personas-parquet', type=str, default='data_4_qdrant/personas.parquet')
    parser.add_argument('--evidence', type=str, default='./data/evidence.json')
    parser.add_argument('--output', type=str, default='data_4_qdrant/')
    parser.add_argument('--news-context-path', type=str, default=None)
    parser.add_argument('--use-news-enricher', action='store_true', default=False)
    parser.add_argument('--simulation-steps', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--survey-questions-path', type=str, default='./data/survey_questions.json', help='Path to survey questions JSON file (with "questions" array)')
    parser.add_argument('--api_key', type=str, default=os.getenv('YANDEX_API_KEY'))
    parser.add_argument('--folder_id', type=str, default=os.getenv('YANDEX_FOLDER_ID'))
    return parser.parse_args()


async def main_async(args):
    survey_qs = None
    if args.survey_questions_path:
        try:
            with open(args.survey_questions_path, 'r', encoding='utf-8') as f:
                survey_data = json.load(f)
                survey_qs = survey_data.get('questions', []) if isinstance(survey_data, dict) else None
        except Exception:
            survey_qs = None

    await run_simulations_from_parquet(
        personas_parquet=Path(args.personas_parquet),
        evidence_path=Path(args.evidence),
        output_dir=Path(args.output),
        survey_questions=survey_qs,
        news_context_path=Path(args.news_context_path) if args.news_context_path else None,
        use_news_enricher=args.use_news_enricher,
        simulation_steps=args.simulation_steps,
        concurrency=args.concurrency,
        timeout=args.timeout,
        api_key=args.api_key,
        folder_id=args.folder_id,
    )


def main():
    setup_logging()
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()