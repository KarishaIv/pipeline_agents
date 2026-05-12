import warnings
warnings.filterwarnings('ignore')

import argparse
import asyncio
import json
import logging
import os
from typing import Dict

from dotenv import load_dotenv
load_dotenv(override=True)

from config import *
from src.data_loading import load_evidence_from_json
from src.orchestration import PipelineRunner
from src.core.simulation_manager import split_news_context_file_payload

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log', encoding='utf-8')
        ]
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def parse_arguments():
    """Парсинг аргументов командной строки"""
    # Load .env if present so defaults can be pulled from env vars
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='Survey Pipeline with Multi-Agent Simulation'
    )
    
    parser.add_argument('--evidence', type=str, default='./data/evidence.json',
                        help='Path to JSON file with evidence data')
    parser.add_argument('--api_key', type=str, default=os.getenv("YANDEX_API_KEY"),
                        help='Yandex GPT API key (or set YANDEX_API_KEY in .env)')
    parser.add_argument('--folder_id', type=str, default=os.getenv("YANDEX_FOLDER_ID"),
                        help='Yandex Cloud Folder ID (or set YANDEX_FOLDER_ID in .env)')
    parser.add_argument('--nemo_size', type=int, default=DEFAULT_NEMO_SIZE,
                        help='Size of Nemotron dataset to use')
    parser.add_argument('--output', type=str, default='outputs/',
                        help='Output directory for results')
    parser.add_argument('--simulation_steps', type=int, default=1,
                        help='Number of steps for simulation')
    parser.add_argument('--concurrency', type=int, default=15,
                        help='Number of parallel persona simulations')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Timeout per persona simulation (seconds)')
    parser.add_argument('--ta_concurrency', type=int, default=1,
                        help='Number of parallel target audience processing')
    parser.add_argument('--agent_mode', type=str, default='credit', 
                        choices=['credit', 'survey'],
                        help="Agent simulation mode")
    parser.add_argument('--decision-mode', type=str, default='direct',
                        choices=['direct', 'compact_debate'],
                        help='Decision mode for credit agent reasoning')
    parser.add_argument('--survey-mode', type=str, default='legacy',
                        choices=['legacy', 'structured'],
                        help='Survey reasoning mode inside agent_mode=survey')
    parser.add_argument('--news-context-path', type=str, default=None,
                        help='Optional path to a news-context JSON snapshot for compact_debate or structured survey runtime')
    parser.add_argument('--visualize-sample', type=int, default=5,
                        help='Number of per-person emotion charts to save (credit mode). 0 disables.')
    parser.add_argument('--no-summary-visualize', dest='summary_visualize', action='store_false',
                        help='Disable summary emotions chart (credit mode)')
    parser.set_defaults(summary_visualize=True)
    parser.add_argument('--use_pgm', action='store_true',  default=True,
                        help='Use PGM for synthetic data generation (default: True)')
    parser.add_argument('--no-pgm', action='store_false', dest='use_pgm',
                        help='Skip PGM, use real Russian data filtered by evidence')
    parser.add_argument("--oceanflag", action="store_true",
                        help="Enable OCEAN calculation (default: enabled if flag is present)")
    parser.add_argument("--no-oceanflag", dest="oceanflag", action="store_false", help="Disable OCEAN calculation")
    parser.set_defaults(oceanflag=True)
    
    return parser.parse_args()

def main():
    """Основная функция запуска пайплайна"""
    args = parse_arguments()
    
    setup_logging()
    api_key = args.api_key or os.getenv("YANDEX_API_KEY")
    folder_id = args.folder_id or os.getenv("YANDEX_FOLDER_ID")
    if not api_key:
        raise SystemExit("API key not provided. Use --api_key or set YANDEX_API_KEY in .env.")

    # Prepare world_contexts from --news-context-path using shared helper
    # so that world_contexts.parquet gets populated on save.
    world_contexts: Dict[str, dict] = {}
    if args.news_context_path:
        try:
            with open(args.news_context_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            evidence_data = load_evidence_from_json(args.evidence) if not isinstance(args.evidence, list) else args.evidence
            world_contexts, _ = split_news_context_file_payload(loaded, evidence_data)
        except Exception as e:
            logging.warning(f"Failed to load news_context from {args.news_context_path}: {e}")

    pipeline_config = {
        'evidence': args.evidence,
        'nemo_size': args.nemo_size,
        'output': args.output,
        'simulation_steps': args.simulation_steps,
        'concurrency': args.concurrency,
        'timeout': args.timeout,
        'ta_concurrency': args.ta_concurrency,
        'agent_mode': args.agent_mode,
        'decision_mode': args.decision_mode,
        'survey_mode': args.survey_mode,
        'news_context_path': args.news_context_path,
        'world_contexts': world_contexts,
        'use_pgm': args.use_pgm,
        "ocean_flag": args.oceanflag,
        "visualize_sample": args.visualize_sample,
        "summary_visualize": args.summary_visualize,
    }
    
    runner = PipelineRunner(pipeline_config)
    asyncio.run(runner.run())

if __name__ == "__main__":
    main()
