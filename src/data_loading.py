import pandas as pd
import numpy as np
from config import *
import json
from src.utils import get_income_range

def load_evidence_from_json(json_path: str):
    """Load evidence from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)
    return evidence_data

def load_sber_data():
    portrait = pd.read_excel(DATA_PATHS['sber'], sheet_name=2)
    portrait = portrait[(portrait['Возраст'] != 'Все группы') & (portrait['Доход'] != 'Все группы')]
    return portrait

def load_synthetic_data():
    df = pd.read_csv(DATA_PATHS['synthetic'])
    df['marital_status'] = df['marital_status'].replace('Замужем', 'Женат')
    return df

def load_american_data(value: int) -> pd.DataFrame:
    """Загрузка американских данных"""
    df = pd.read_parquet(DATA_PATHS['nemotron'])
    print(df.shape)
    return df.head(value)

def load_nemotron_sample() -> pd.DataFrame:
    """Загрузка сэмпла Nemotron"""
    nemo_sample = pd.read_csv(DATA_PATHS['nemotron_sample'])
    nemo_sample.rename(columns={
        'возраст': 'age', 'пол': 'gender', 'образование': 'education',
        'семейное_положение': 'marital_status', 'род_занятий': 'occupation',
        'город': 'region'
    }, inplace=True)
    return nemo_sample

def load_survey_data():
    try:
        with open(DATA_PATHS['survey'], 'r', encoding='utf-8') as f:
            survey_data = json.load(f)
            survey_questions = survey_data.get('questions', [])
        return survey_questions
    except Exception as e:
        print(f" Failed to load survey questions: {e}")
        return


def preprocess_pgm_data(df):
    """Preprocess data for PGM model"""
    df_prep = df.copy()
    
    df_prep['age_group'] = df['age'].apply(
        lambda x: np.digitize(x, PGM_CONFIG['age_bins'])
    )
    
    income_bins = PGM_CONFIG.get('income_bins', None)
    income_labels = PGM_CONFIG.get('income_labels', None)
    
    if income_bins is not None and income_labels is not None:
        df_prep['income_level'] = pd.to_numeric(df['income'], errors='coerce').parallel_apply(
            lambda x: get_income_range(x, income_bins, income_labels)
        )
    
    df_prep['children_group'] = df['children'].apply(
        lambda x: '0' if x == 0 else '1' if x == 1 else '2' if x == 2 else '3+'
    )
    
    df_prep['region'] = df['region'].apply(lambda x : 'Регион миллионник' if x in PGM_CONFIG['rich_regions'] else 'Бедный регион')

    return df_prep