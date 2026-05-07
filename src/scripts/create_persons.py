"""Standalone script to generate ~100 persons (or as specified in evidence) and save only personas + target_audiences to parquet.

Usage:
  python -m src.scripts.create_persons --evidence ./data/evidence_100.json --output ./data_4_qdrant

Reuses logic from orchestration without modifying core files.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from src.data_loading import load_synthetic_data, load_american_data, preprocess_pgm_data
from src.pgm_model import create_pgm_model, train_pgm_model
from src.utils import get_uuid, get_clear_personas
from src.orchestration import (
    generate_personas_via_pgm,
    filter_personas_from_real_data,
    _build_persona_rows,
    _build_target_audience_rows,
)
from src.core.storage import StorageManager
from config import DEFAULT_NEMO_SIZE, set_yandex_credentials

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


async def run_create_persons(
    evidence: List[Dict],
    output_dir: Path,
    use_pgm: bool = True,
    nemo_size: int = DEFAULT_NEMO_SIZE,
    ocean_flag: bool = True,
    api_key: Optional[str] = None,
    folder_id: Optional[str] = None,
) -> pd.DataFrame:
    """Core function to generate personas and save personas.parquet + target_audiences.parquet.

    This is the entrypoint used by tests and CLI.
    """
    if api_key:
        set_yandex_credentials(api_key, folder_id)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load minimal datasets needed for persona generation
    russian_data = load_synthetic_data()
    russian_preprocessed = preprocess_pgm_data(russian_data)
    nemo = load_american_data(nemo_size)

    pgm_model = None
    if use_pgm:
        logger.info("Training PGM model for persona generation...")
        df_prep = preprocess_pgm_data(russian_data)
        model = create_pgm_model()
        pgm_model = train_pgm_model(model, df_prep)

    if use_pgm:
        all_personas, _ = await generate_personas_via_pgm(
            evidence_list=evidence,
            model=pgm_model,
            df_base=russian_data,
            nemo=nemo,
            output_dir=output_dir,
            ta_concurrency=1,
            ocean_flag=ocean_flag,
        )
    else:
        all_personas, _ = await filter_personas_from_real_data(
            evidence_list=evidence,
            df_russian_preprocessed=russian_preprocessed,
            nemo=nemo,
            output_dir=output_dir,
            ta_concurrency=1,
            ocean_flag=ocean_flag,
        )

    # Add UUIDs and age formatting (same as PipelineRunner)
    if 'age_group' in all_personas.columns:
        all_personas['age_group'] = all_personas['age_group'].apply(
            lambda x: f'{int(x)*5}-{int(x)*5+4}' if isinstance(x, (int, float)) else str(x)
        )

    all_personas['UUID'] = all_personas.apply(
        lambda row: get_uuid("personas", get_clear_personas(row).to_string()), axis=1
    )

    # Save personas
    personas_path = output_dir / "personas.parquet"
    persona_rows = _build_persona_rows(all_personas)
    await StorageManager.append_parquet_async(persona_rows, personas_path)

    # Save target audiences (linked to personas)
    ta_path = output_dir / "target_audiences.parquet"
    ta_rows = _build_target_audience_rows(evidence, all_personas)
    await StorageManager.append_parquet_async(ta_rows, ta_path, check_columns=False)

    logger.info(f"✅ Created {len(all_personas)} personas -> {personas_path}")
    logger.info(f"✅ Saved {len(ta_rows)} target audiences -> {ta_path}")

    return all_personas


def parse_args():
    parser = argparse.ArgumentParser(description="Generate persons and save to parquet (no simulations)")
    parser.add_argument('--evidence', type=str, default='./data/evidence.json', help='Path to evidence JSON')
    parser.add_argument('--output', type=str, default='data_4_qdrant/', help='Output dir for parquets')
    parser.add_argument('--nemo-size', type=int, default=DEFAULT_NEMO_SIZE)
    parser.add_argument('--use_pgm', action='store_true', default=True)
    parser.add_argument('--no-pgm', action='store_false', dest='use_pgm')
    parser.add_argument('--oceanflag', action='store_true', default=True)
    parser.add_argument('--no-oceanflag', action='store_false', dest='oceanflag')
    parser.add_argument('--api_key', type=str, default=os.getenv('YANDEX_API_KEY'))
    parser.add_argument('--folder_id', type=str, default=os.getenv('YANDEX_FOLDER_ID'))
    return parser.parse_args()


async def main_async(args):
    from src.data_loading import load_evidence_from_json
    evidence = load_evidence_from_json(args.evidence)
    await run_create_persons(
        evidence=evidence,
        output_dir=Path(args.output),
        use_pgm=args.use_pgm,
        nemo_size=args.nemo_size,
        ocean_flag=args.oceanflag,
        api_key=args.api_key,
        folder_id=args.folder_id,
    )


def main():
    setup_logging()
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()