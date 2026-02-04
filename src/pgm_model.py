import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
from config import *


def create_pgm_model():
    """Create PGM model with proper causal structure"""
    model = DiscreteBayesianNetwork([
        # Region as root node - influences demographics
        ('region', 'age_group'),
        ('region', 'education'),
        ('region', 'income_level'),
        
        # Basic demographics
        ('age_group', 'marital_status'),
        ('gender', 'marital_status'),
        
        # Children depend on demographics
        ('age_group', 'children_group'),
        ('marital_status', 'children_group'),
        ('gender', 'children_group'),
        ('region', 'children_group'),  # рождаемость варьируется по регионам
        
        # Education influences occupation and income
        ('education', 'occupation'),
        ('age_group', 'occupation'),
        
        # Income depends on multiple factors
        ('region', 'income_level'),
        ('education', 'income_level'),
        ('occupation', 'income_level'),
        ('age_group', 'income_level'),
        ('gender', 'income_level'),
    ])
    return model


def train_pgm_model(model: DiscreteBayesianNetwork, df_prep: pd.DataFrame) -> DiscreteBayesianNetwork:
    """
    Строит дискретную байесовскую сеть (DiscreteBayesianNetwork) на основе подготовленных данных.
    Модель обучается максимальным правдоподобием по признакам:
    age_group, gender, marital_status, children_group, education, occupation, region_type, income_level.
    Возвращает обученную модель.
    """
    
    model_features = [
        'age_group', 'gender', 'marital_status', 'children_group',
        'education', 'occupation', 'region', 'income_level'
    ]
    
    model_data = df_prep[model_features].dropna()
    model.fit(model_data, estimator=MaximumLikelihoodEstimator)
    return model


def generate_synthetic_data(model, evidence=None, size=10):
    """
    Генерирует синтетические данные на основе обученной байесовской модели PGM.
    - Если evidence задано, применяет взвешенную по правдоподобию выборку пол запрос evidence.
    - Если evidence не задано, выполняет прямую (forward) выборку по общему распределению.
    Возвращает DataFrame с синтетическими наблюдениями.
    """
    
    sampler = BayesianModelSampling(model)
    
    if evidence:
        print(evidence)
        synthetic_data = sampler.likelihood_weighted_sample(
            evidence=evidence, size=size
        ).drop('_weight', axis=1)
    else:
        synthetic_data = sampler.forward_sample(size=size)
    
    return synthetic_data