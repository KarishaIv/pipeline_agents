import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# LLM настройки
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.5
LLM_MAX_RETRIES = 3

# Пути к данным
DATA_PATHS = {
    'sber': './data/Real data/Sber_sample.xlsx',
    'synthetic': './data/Synthetic/rosstat_dataset_extended.csv',
    'nemotron_sample': './data/Nemotron/Nemotron_sample.csv',
    'nemotron': './data/Nemotron/nemotron.parquet',
    'evidence': './data/survey_evidence.json',
    'survey': './data/survey_questions.json' 
}

# './data/Nemotron/nemo_01.parquet' ,


# Настройки PGM
PGM_CONFIG = {
    'age_bins': np.arange(0, 100, 5),
    'income_bins': np.array([0, 17733, 27000, 45000, 123500, 250000, 500000, np.inf]),
    'income_labels': ['Низкий', 'Выше МРОТ', 'Средний', 'Выше_среднего', 'Высокий', 'Очень_высокий', 'Ultima'],
    'rich_regions': [
    'Москва',
    'Санкт-Петербург',
    'Новосибирская область',
    'Свердловская область',
    'Нижегородская область',
    'Самарская область',
    'Ростовская область',
    'Красноярский край',
    'Воронежская область',
    'Пермский край',
    'Волгоградская область',
    'Республика Башкортостан',
    'Республика Татарстан',
    'Челябинская область',
    'Омская область',
    'Краснодарский край'
]
}

# Маппинги для унификации
GENDER_MAPPING = {'Мужской': 0, 'Женский': 1, 'Male': 0, 'Female': 1}

EDUCATION_MAPPING = {
    'Неполное среднее': 0, 'Среднее': 1, 'среднего профессионального образования': 1, 
    'магистратура': 4, 'бакалавриат': 3, 'Незаконченное высшее': 2, 'аспирантура' : 5, 'специалитет' : 3,
    'less_than_9th': 0, 'high_school': 1, 'some_college': 2, 
    'graduate': 4, 'bachelors': 3, '9th_12th_no_diploma': 2, 'associates': 5
}

MARITAL_MAPPING = {
    'Не женат': 0, 'Не замужем': 0, 'Женат': 1, 'Замужем': 1, 
    'Разведен': 2, 'Разведена': 2, 'Вдовец': 3, 'Вдова': 3,
    'never_married': 0, 'married_present': 1, 'divorced': 2, 'widowed': 3, 'separated': 2
}
MATCH_COLS = ['age_group', 'education', 'marital_status', 'gender']

TOP_N_NEIGHBORS = 5
TOP_N_CATEGORIES = 2
DEFAULT_NEMO_SIZE = 10000

def set_openai_api_key(api_key: str):
    """Set OpenAI API key"""
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

GMM_CONFIG = {'n_components': 1,        
    'min_components': 3,           # минимальное количество кластеров для auto
    'max_components': 3,           # максимальное количество кластеров для auto
    'covariance_type': 'full',    
    'random_state': 42,
}

OCEAN_CALCULATION = {
    'batch_size': 100,
    'max_retries': 3,
    'cache_results': True
}

OUTPUT_COLUMNS = [
     *MATCH_COLS, # социо-демо
    "openness", "conscientiousness", "extraversion",  "agreeableness", "neuroticism"                    # OCEAN
    'cluster_id', 'group_key', 'cluster_weight',            # кластер инфо
    'n_americans_in_group', 'n_clusters_total'              # статистика
]