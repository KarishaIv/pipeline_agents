import warnings
warnings.filterwarnings('ignore')

import argparse
import asyncio
import logging
from pathlib import Path

from src.orchestration import PipelineRunner
from config import *

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

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Credit Decision Prediction Pipeline with Multi-Agent Simulation'
    )
    
    parser.add_argument('--evidence', type=str, default='./data/evidence.json',
                        help='Path to JSON file with evidence data')
    parser.add_argument('--api_key', type=str, required=True, 
                        help='Yandex GPT API key')
    parser.add_argument('--folder_id', type=str, default=None,
                        help='Yandex Cloud Folder ID (if not provided, will use api_key)')
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
    set_openai_api_key(args.api_key, args.folder_id)
    
    pipeline_config = {
        'evidence': args.evidence,
        'nemo_size': args.nemo_size,
        'output': args.output,
        'simulation_steps': args.simulation_steps,
        'concurrency': args.concurrency,
        'timeout': args.timeout,
        'ta_concurrency': args.ta_concurrency,
        'agent_mode': args.agent_mode,
        'use_pgm': args.use_pgm,
        "ocean_flag": args.oceanflag,
    }
    
    runner = PipelineRunner(pipeline_config)
    asyncio.run(runner.run())

if __name__ == "__main__":
    main()