import asyncio
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import logging

from data_loading import load_synthetic_data, load_american_data, preprocess_pgm_data, load_survey_data, load_evidence_from_json
from pgm_model import create_pgm_model, train_pgm_model, generate_synthetic_data
from utils import normalize_features, normalize_evidence, filter_real_russian_data, translate_ocean_to_readable, get_income_range, get_uuid, get_embedding, get_clear_personas
from clustering import replicate_personas_with_gmm
from core.simulation_manager import SimulationManager
from core.storage import StorageManager
from news_enricher import NewsContextEnricher

from config import *
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=4)

logger = logging.getLogger(__name__)

class PipelineRunner:
    """Основной класс для запуска пайплайна генерации и симуляции персон"""
    
    def __init__(self, config: dict, news_enricher: Optional[NewsContextEnricher] = None):
        self.config = config
        self.output_dir = Path(config['output'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_pgm = config.get('use_pgm', True)
        self.news_enricher = news_enricher
        
    async def run(self):
        """Запуск полного пайплайна"""
        logger.info("🚀 Starting pipeline execution")
        logger.info(f"Mode: {'PGM (synthetic)' if self.use_pgm else 'Real data (filtered)'}")
        
        # 1. Загрузка данных
        datasets = await self._load_datasets()
        
        # 2. Обучение PGM модели (только если используем PGM)
        pgm_model = await self._train_pgm_model(datasets['russian']) if self.use_pgm else None
        
        # 3. Генерация/фильтрация персон для всех целевых аудиторий
        all_personas = await self._generate_or_filter_personas(datasets, pgm_model)

        # 3.5 Получение новостного контекста среды (опционально)
        world_contexts: Dict[str, dict] = {}
        if self.news_enricher:
            world_contexts = await self._enrich_with_news_context(datasets['evidence'])

        # 4. Запуск симуляций
        results = await self._run_simulations(all_personas, datasets, world_contexts)
        
        # 5. Сохранение результатов
        await self._save_results(all_personas, results, datasets)
        
        logger.info("✅ Pipeline completed successfully")
        return results
    
    async def _load_datasets(self) -> Dict[str, any]:
        """Загрузка всех необходимых datasets"""
        logger.info("Loading datasets...")

        if isinstance(self.config['evidence'], list):
            evidence_data = self.config['evidence']
        else:
            evidence_data = load_evidence_from_json(self.config['evidence'])
        nemo_data = load_american_data(self.config['nemo_size'])
        russian_data = load_synthetic_data()
        
        # Предобрабатываем российские данные для единообразия
        russian_data_preprocessed = preprocess_pgm_data(russian_data)
        
        datasets = {
            'evidence': evidence_data,
            'nemo': nemo_data,
            'russian': russian_data,
            'russian_preprocessed': russian_data_preprocessed
        }
        
        # Загрузка вопросов опроса
        if self.config.get('survey_questions'):
            datasets['survey_questions'] = self.config['survey_questions']
            logger.info(f"Loaded {len(datasets['survey_questions'])} survey questions (inline)")
        else:
            try:
                datasets['survey_questions'] = load_survey_data()
                logger.info(f"Loaded {len(datasets['survey_questions'])} survey questions")
            except Exception as e:
                logger.warning(f"Failed to load survey questions: {e}")
                datasets['survey_questions'] = []
        
        logger.info(f"  ✓ Evidence: {len(evidence_data)} target audiences")
        logger.info(f"  ✓ Russian data: {len(russian_data)} personas")
        logger.info(f"  ✓ American data: {len(nemo_data)} personas")
        logger.info(f"  ✓ Using {'PGM generation' if self.use_pgm else 'real data filtering'}")
        
        return datasets
    
    async def _train_pgm_model(self, russian_data: pd.DataFrame):
        """Обучение PGM модели (только при использовании PGM)"""
        if not self.use_pgm:
            logger.info("⏭️  Skipping PGM training (use_pgm=False)")
            return None
            
        logger.info("🧠 Training PGM model...")
        
        df_prep = preprocess_pgm_data(russian_data)
        model = create_pgm_model()
        trained_model = train_pgm_model(model, df_prep)
        
        logger.info(f"  ✓ Model trained: {len(trained_model.nodes())} nodes, {len(trained_model.edges())} edges")
        return trained_model
    
    async def _enrich_with_news_context(
        self, evidence_list: List[Dict]
    ) -> Dict[str, dict]:
        """
        Для каждой целевой аудитории делает один вызов новостного оркестратора.
        Возвращает словарь {ta_name → news_context} — контекст среды, отдельно от персон.
        """
        logger.info("📰 Получение новостного контекста...")

        await self.news_enricher.initialize()

        ta_context_map: Dict[str, dict] = {}
        for evidence in evidence_list:
            ta_name = evidence.get('target_audience_name', '')
            question = evidence.get('news_question', ta_name)
            if ta_name:
                context = await self.news_enricher.get_news_context(question, ta_name)
                ta_context_map[ta_name] = context

        enriched = sum(1 for v in ta_context_map.values() if v)
        logger.info(f"  ✓ Новостной контекст получен для {enriched}/{len(ta_context_map)} ЦА")
        return ta_context_map

    async def _generate_or_filter_personas(self, datasets: Dict, pgm_model) -> pd.DataFrame:
        """Генерация или фильтрация персон для всех целевых аудиторий"""
        if self.use_pgm:
            logger.info("👥 Generating synthetic personas via PGM...")
            all_personas, ta_summary = await generate_personas_via_pgm(
                evidence_list=datasets['evidence'],
                model=pgm_model,
                df_base=datasets['russian'],
                nemo=datasets['nemo'],
                output_dir=self.output_dir,
                ta_concurrency=self.config['ta_concurrency'],
                ocean_flag=self.config['ocean_flag']
            )
        else:
            logger.info("🔍 Filtering real Russian personas by evidence...")
            all_personas, ta_summary = await filter_personas_from_real_data(
                evidence_list=datasets['evidence'],
                df_russian_preprocessed=datasets['russian_preprocessed'],
                nemo=datasets['nemo'],
                output_dir=self.output_dir,
                ta_concurrency=self.config['ta_concurrency'],
                ocean_flag=self.config['ocean_flag']
            )

        if 'age_group' in all_personas.columns:
            all_personas['age_group'] = all_personas['age_group'].apply(
                lambda x: f'{int(x)*5}-{int(x)*5+4}' if isinstance(x, (int, float)) else str(x)
            )

        all_personas['UUID'] = all_personas.apply(
            lambda row: get_uuid("personas", get_clear_personas(row).to_string()), axis=1
        )

        
        logger.info(f"  ✓ Processed {len(all_personas)} personas across {len(datasets['evidence'])} target audiences")
        return all_personas
    
    async def _run_simulations(self, all_personas: pd.DataFrame, datasets, world_contexts: Dict[str, dict] = None):
        """Запуск мульти-агентных симуляций"""
        logger.info("🤖 Running multi-agent simulations...")

        personas = [all_personas.iloc[i].to_dict() for i in range(len(all_personas))]
        news_context = None
        news_context_path = self.config.get('news_context_path')
        if news_context_path:
            with open(news_context_path, 'r', encoding='utf-8') as f:
                news_context = json.load(f)
        
        manager = SimulationManager(
            out_dir=self.output_dir,
            concurrency=self.config['concurrency'],
            timeout=self.config['timeout'],
            run_retries=1,
            agent_mode=self.config['agent_mode'],
            decision_mode=self.config.get('decision_mode', 'direct'),
            survey_mode=self.config.get('survey_mode', 'legacy'),
            news_context=news_context,
            survey_questions=datasets.get('survey_questions', []),
            visualize_sample=self.config.get('visualize_sample', 0),
            summary_visualize=self.config.get('summary_visualize', True)
        )
        
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        results = await manager.run_many(
            personas, 
            steps=self.config['simulation_steps'], 
            out_subdir=f"sim_{timestamp}"
        )
        
        logger.info(f"  ✓ Completed {len(results)} simulations")
        return results
    
    async def _save_results(self, all_personas: pd.DataFrame, results: List, datasets: Dict):
        """Сохранение всех результатов пайплайна"""
        logger.info("💾 Saving pipeline results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        questions_path = self.output_dir.parent / "data_4_qdrant/questions.parquet"
        personas_path = self.output_dir.parent / "data_4_qdrant/personas.parquet"
        target_audiences_path = self.output_dir.parent / "data_4_qdrant/target_audiences.parquet"
        simulations_path = self.output_dir.parent / "data_4_qdrant/simulations.parquet"

        # Сохранение вопросов опроса
        logger.info(f" Сохранение вопросов опроса")
        questions_uuids = {
            question : get_uuid("questions", question)
            for question in datasets['survey_questions']
        }
        await StorageManager.append_parquet_async(
            [
                {
                    "UUID": questions_uuids[question],
                    "embedding": get_embedding(question, query=False),
                    "question": question,
                }
                for question in datasets['survey_questions']
            ], 
            questions_path,
        )
        
        # Сохранение персон
        logger.info(f" Сохранение персон")
        await StorageManager.append_parquet_async(
            [
                {
                    "UUID": persona['UUID'],
                    "embedding": get_embedding(get_clear_personas(persona).to_string(), query=False),
                    **persona.to_dict(),
                }
                for _, persona in all_personas.iterrows()
            ], 
            personas_path,
        )

        # Сохранение целевых аудиторий
        logger.info(f" Сохранение целевых аудиторий")
        await StorageManager.append_parquet_async(
            [
                {
                    "UUID": get_uuid("target_audiences", str(group)),
                    "embedding": get_embedding(str(group), query=False),
                    **group,
                    "personas_uuids": [
                        persona['UUID']
                        for _, persona in all_personas[
                            all_personas['target_audience_name'] == group['target_audience_name']
                        ].iterrows()
                    ],

                }
                for group in datasets['evidence']
            ], 
            target_audiences_path,
            check_columns=False
        )

        # Сохранение результатов симуляций
        logger.info(f" Сохранение результатов симуляций")
        await StorageManager.append_parquet_async(
            [
                {
                    "UUID": get_uuid("simulations"),
                    **get_sim_embeddings(state),
                    **get_sim_reasonings(state),
                    **get_sim_last_reactions(state),
                    "persona_UUID": result['profile']['UUID'],
                    "question_UUID": questions_uuids[state['scenario']],
                    "decision_reasoning": get_decision(state['final_decision'])['reasoning'],
                    "decision": get_decision(state['final_decision'])['decision'],
                    "decision_confidence": get_decision(state['final_decision'])['confidence'],
                    "generation_count": state['generation_count'],
                    "max_generations": state['max_generations'],
                    "timestamp": state['timestamp'],
                    # TODO: добавить контекст агентов
                } 
                for result in results if 'survey_responses' in result
                for response in result['survey_responses']
                for state in [response['full_state']]   # чтобы не писать на каждой строке ['full_state']
            ],
            simulations_path,
            check_columns=False
        )
        
async def generate_personas_via_pgm(
    evidence_list: List[Dict],
    model,
    df_base: pd.DataFrame,
    nemo: pd.DataFrame,
    output_dir: Path,
    ta_concurrency: int = 2,
    ocean_flag: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Генерация персон через PGM модель"""
    
    async def data_fetcher(evidence: Dict, ta_index: int) -> Tuple[pd.DataFrame, str, int, str]:
        """Фетчер данных для PGM режима"""
        ta_name = evidence.get('target_audience_name', f'TA_{ta_index}')
        synthetic_size = evidence.get('synthetic_size', 10)
        
        logger.info(f"[TA:{ta_name}] Генерация {synthetic_size} синтетических персон через PGM")
        
        normalized_evidence = normalize_evidence(evidence)
        synthetic_data = generate_synthetic_data(
            model, evidence=normalized_evidence, size=synthetic_size
        )
        return synthetic_data, ta_name, synthetic_size, 'pgm_synthetic'
    
    return await _process_target_audiences_generic(
        evidence_list=evidence_list,
        data_fetcher=data_fetcher,
        nemo=nemo,
        output_dir=output_dir,
        ta_concurrency=ta_concurrency,
        ocean_flag=ocean_flag
    )


async def filter_personas_from_real_data(
    evidence_list: List[Dict],
    df_russian_preprocessed: pd.DataFrame,
    nemo: pd.DataFrame,
    output_dir: Path,
    ta_concurrency: int = 2,
    ocean_flag: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Фильтрация реальных российских персон по evidence"""
    
    async def data_fetcher(evidence: Dict, ta_index: int) -> Tuple[pd.DataFrame, str, int, str]:
        """Фетчер данных для режима реальных данных"""
        ta_name = evidence.get('target_audience_name', f'TA_{ta_index}')
        sample_size = evidence.get('synthetic_size', 10)
        
        logger.info(f"[TA:{ta_name}] Фильтрация реальных персон")
        
        filtered_data = filter_real_russian_data(evidence, df_russian_preprocessed, sample_size)
        
        if len(filtered_data) == 0:
            logger.warning(f"[TA:{ta_name}] Не найдено реальных персон с заданными критериями")
            empty_data = pd.DataFrame(columns=df_russian_preprocessed.columns)
            return empty_data, ta_name, sample_size, 'real_filtered'
        
        return filtered_data, ta_name, len(filtered_data), 'real_filtered'
    
    return await _process_target_audiences_generic(
        evidence_list=evidence_list,
        data_fetcher=data_fetcher,
        nemo=nemo,
        output_dir=output_dir,
        ta_concurrency=ta_concurrency,
        ocean_flag=ocean_flag
    )


async def _process_target_audiences_generic(
    evidence_list: List[Dict],
    data_fetcher: Callable,
    nemo: pd.DataFrame,
    output_dir: Path,
    ta_concurrency: int = 2,
    ocean_flag: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Общая функция для обработки целевых аудиторий
    
    Args:
        evidence_list: Список целевых аудиторий
        data_fetcher: Функция для получения данных (PGM или реальных)
        nemo: Американские данные
        output_dir: Директория для сохранения
        ta_concurrency: Параллелизм обработки ЦА
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (все персоны, статистика по ЦА)
    """
    
    async def process_target_audience(ta_index: int, evidence: Dict):
        """Обработка одной целевой аудитории - общая логика"""
        # 1. Получаем данные
        russian_data, ta_name, original_size, data_source = await data_fetcher(evidence, ta_index)
        
        # 2. Проверяем, есть ли данные для обработки
        if len(russian_data) == 0:
            logger.warning(f"[TA:{ta_name}] Нет данных для обработки")
            return
        
        # 3. Нормализация features 
        russian_norm, american_norm = normalize_features(russian_data, nemo)

        if ocean_flag:
        
            # 4. GMM кластеризация и репликация 
            replicated_personas, clustering_stats = replicate_personas_with_gmm(
                russian_df=russian_data,
                russian_norm=russian_norm,
                american_norm=american_norm,
                nemo_full=nemo.copy(),
                group_cols=MATCH_COLS,
                top_n_categories=TOP_N_CATEGORIES,
                gmm_params=GMM_CONFIG,
                use_gower=False,
                sampling_method='gmm_sample',
                ta_name=ta_name
            )

            final_personas = translate_ocean_to_readable(replicated_personas)

        else:
            final_personas = russian_data
           
        # 5. Добавление метаданных 
        final_personas = _add_metadata_to_personas(
            personas=final_personas,
            ta_index=ta_index,
            ta_name=ta_name,
            data_source=data_source
        )
        
        # 6. Формирование статистики
        ta_stats = _create_ta_stats(
            ta_index=ta_index,
            ta_name=ta_name,
            data_source=data_source,
            original_size=original_size,
            replicated_size=len(final_personas),
            unique_clusters=None if not ocean_flag or 'cluster_id' not in final_personas.columns else final_personas['cluster_id'].nunique(),
            unique_groups=None if not ocean_flag or 'group_key' not in final_personas.columns else final_personas['group_key'].nunique()
        )

        logger.info(f"[TA:{ta_name}] Обработка завершена: {len(final_personas)} персон")
        return final_personas, ta_stats
    
    return await _process_target_audiences_parallel(
        evidence_list, process_target_audience, output_dir, ta_concurrency
    )

def _add_metadata_to_personas(
    personas: pd.DataFrame,
    ta_index: int,
    ta_name: str,
    data_source: str
) -> pd.DataFrame:
    """Добавляет метаданные к персонам"""
    personas = personas.copy()
    personas['target_audience_id'] = ta_index
    personas['target_audience_name'] = ta_name
    personas['data_source'] = data_source
    return personas


def _create_ta_stats(
    ta_index: int,
    ta_name: str,
    data_source: str,
    original_size: int,
    replicated_size: int,
    unique_clusters: int,
    unique_groups: int
) -> Dict:
    """Создает статистику для целевой аудитории"""
    return {
        'target_audience_id': ta_index,
        'target_audience_name': ta_name,
        'data_source': data_source,
        'original_size': original_size,
        'replicated_size': replicated_size,
        'unique_clusters': unique_clusters,
        'unique_groups': unique_groups
    }


async def _process_target_audiences_parallel(
    evidence_list: List[Dict],
    process_function: Callable,
    output_dir: Path,
    ta_concurrency: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Общая функция для параллельной обработки целевых аудиторий"""
    
    logger.info(f"Параллельная обработка {len(evidence_list)} целевых аудиторий (параллелизм: {ta_concurrency})")
    
    semaphore = asyncio.Semaphore(ta_concurrency)
    
    async def process_with_semaphore(ta_index, evidence):
        async with semaphore:
            return await process_function(ta_index, evidence)
    
    tasks = [
        process_with_semaphore(i, evidence) 
        for i, evidence in enumerate(evidence_list)
    ]
    
    results = await asyncio.gather(*tasks)
    all_personas_list, all_stats_list = zip(*results)
    
    all_personas = pd.concat(all_personas_list, ignore_index=True)
    ta_summary_stats = pd.DataFrame(all_stats_list)
    
    ta_summary_stats.to_csv(output_dir / 'target_audiences_summary.csv', index=False)
    total_personas = len(all_personas)
    total_tas = len(evidence_list)
    logger.info(f"Обработка завершена: {total_personas} персон через {total_tas} целевых аудиторий")
    
    for _, row in ta_summary_stats.iterrows():
        logger.info(f"  • {row['target_audience_name']}: {row['replicated_size']} персон "
                   f"({row['unique_clusters']} кластеров) - {row['data_source']}")
    
    return all_personas, ta_summary_stats

def get_sim_reasonings(state: Dict) -> Dict[str, List[str]]:
    return {
        "emotional_reasonings": [entry['reasoning'] for entry in state['emotional_history']],
        "rational_reasonings": [entry['reasoning'] for entry in state['rational_history']],
        "social_reasonings": [entry['reasoning'] for entry in state['social_history']],
        "ideological_reasonings": [entry['reasoning'] for entry in state['ideological_history']],
    }

def get_sim_last_reactions(state: Dict) -> Dict[str, str]:
    return {
        "emotional_reaction": state['emotional_history'][-1]['reaction'] if state.get('emotional_history') else "",
        "rational_reaction": state['rational_history'][-1]['reaction'] if state.get('rational_history') else "",
        "social_reaction": state['social_history'][-1]['reaction'] if state.get('social_history') else "",
        "ideological_reaction": state['ideological_history'][-1]['reaction'] if state.get('ideological_history') else "",
    }

def get_sim_embeddings(state: Dict) -> Dict[str, List[float]]:
    reasonings = get_sim_reasonings(state)
    joined_reasonings = {key: "\n".join(value) for key, value in reasonings.items()}
    joined_reasonings['decision_reasoning'] = get_decision(state['final_decision'])['reasoning']
    return {
        "emotional_vector": get_embedding(joined_reasonings['emotional_reasonings'], query=False),
        "rational_vector": get_embedding(joined_reasonings['rational_reasonings'], query=False),
        "social_vector": get_embedding(joined_reasonings['social_reasonings'], query=False),
        "ideological_vector": get_embedding(joined_reasonings['ideological_reasonings'], query=False),
        "decision_vector": get_embedding(joined_reasonings['decision_reasoning'], query=False),
        "general_vector": get_embedding(
            "\n".join(joined_reasonings.values()),
            query=False
        ),
    }

def get_decision(decision: Dict | str | None) -> Dict:
    if isinstance(decision, dict):
        return {
            "reasoning": decision['reasoning'],
            "decision": decision['decision'],
            "confidence": decision['confidence'],
        }
    elif isinstance(decision, str):
        return {
            "reasoning": decision,
            "decision": None,
            "confidence": 0,
        }
    else:
        return {
            "reasoning": "",
            "decision": None,
            "confidence": 0,
        }
