import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import *
import warnings
import logging
from src.data_loading import load_american_data

warnings.filterwarnings('ignore')
from src.agents.ocean_agent import calculate_ocean_profiles

# import gower
GOWER_AVAILABLE = False
ocean_cols=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

logger = logging.getLogger(__name__)


def group_russian_personas(russian_norm, group_cols):
    """
    Группирует российских персон по общим социо-демографическим признакам
    """
    grouped = {}
    
    for name, group in russian_norm.groupby(group_cols):
        group_key = tuple(name) if isinstance(name, (list, tuple, np.ndarray)) else (name,)
        grouped[group_key] = {
            'indices': group.index.tolist(),
            'values': dict(zip(group_cols, group_key)),
            'size': len(group)
        }
    
    logger.debug(f"Сгруппировано {len(grouped)} уникальных групп российских персон")
    return grouped


def find_americans_for_group(american_norm, group_values, group_cols, top_n=4, use_gower=False):
    """
    Находит TOP_N наиболее похожих категорий американцев для группы российских персон
    """
    request_dict = dict(zip(group_cols, group_values))
    
    # Находим уникальные категории американцев
    unique_cats = american_norm[group_cols].drop_duplicates().reset_index(drop=True)
    
    # Вычисляем обычное евклидово расстояние
    dists = cdist(
        [list(request_dict.values())], 
        unique_cats.values, metric='euclidean'
    ).flatten()
    
    # Топ-N категорий
    top_idxs = np.argsort(dists)[:top_n]
    top_cat_combos = unique_cats.iloc[top_idxs].to_dict(orient='records')
    top_dists = dists[top_idxs]
    
    # Находим всех американцев из этих категорий
    mask = np.zeros(len(american_norm), dtype=bool)
    for combo in top_cat_combos:
        msk = np.ones(len(american_norm), dtype=bool)
        for col, val in combo.items():
            msk &= (american_norm[col] == val)
        mask |= msk
    
    matched_indices = american_norm[mask].index.tolist()
    
    logger.debug(f"Найдено {len(matched_indices)} американцев из {len(top_cat_combos)} категорий, расстояние: [{top_dists.min():.3f}, {top_dists.max():.3f}]")
    
    return matched_indices, top_cat_combos, top_dists


def cluster_ocean_gmm(ocean_df, n_components='auto', min_components=2, 
                     max_components=8, covariance_type='full', random_state=42):
    """
    Кластеризует американцев по OCEAN значениям используя Gaussian Mixture Model
    """
    ocean_values = ocean_df[ocean_cols].dropna().values
    
    if len(ocean_values) < min_components:
        logger.warning(f"Недостаточно данных ({len(ocean_values)}), используем 1 компоненту")
        min_components = 1
        max_components = 1
        n_components = 1
    
    # Стандартизация
    scaler = StandardScaler()
    ocean_scaled = scaler.fit_transform(ocean_values)
    
    # Автоматический выбор количества компонент через BIC или AIC
    if n_components in ['auto', 'bic', 'aic']:
        criterion = 'bic' if n_components in ['auto', 'bic'] else 'aic'
        
        scores, models = [], []
        n_range = range(min_components, min(max_components + 1, len(ocean_values)))
        
        for n in n_range:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=covariance_type,
                    random_state=random_state,
                    n_init=10,
                    max_iter=200
                )
                gmm.fit(ocean_scaled)
                
                score = gmm.bic(ocean_scaled) if criterion == 'bic' else gmm.aic(ocean_scaled)
                scores.append(score)
                models.append(gmm)
            except Exception as e:
                logger.warning(f"Ошибка GMM для n={n}: {e}")
                scores.append(np.inf)
                models.append(None)
        
        # Выбираем модель с минимальным BIC/AIC
        best_idx = np.argmin(scores)
        gmm_model = models[best_idx]
        n_components = min_components + best_idx
        bic_score = scores[best_idx]
        
        logger.info(f"Оптимальное количество компонент: {n_components} ({criterion.upper()}={bic_score:.2f})")
    
    else:
        # Фиксированное количество компонент
        gmm_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=10,
            max_iter=200
        )
        gmm_model.fit(ocean_scaled)
        bic_score = gmm_model.bic(ocean_scaled)
    
    # Получаем метки и вероятности
    cluster_labels = gmm_model.predict(ocean_scaled)
    
    # Информация о кластерах
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_info = ", ".join([f"Кластер {label}: {count} ({count/len(cluster_labels)*100:.1f}%)" 
                            for label, count in zip(unique, counts)])
    logger.info(f"Распределение по кластерам: {cluster_info}")
    
    return cluster_labels, gmm_model, n_components, bic_score, scaler


def sample_ocean_from_gmm(gmm_model, cluster_labels, ocean_df, scaler, 
                          method='gmm_sample', n_samples_per_cluster=1):
    """
    Сэмплирует OCEAN значения из каждого кластера GMM
    """
    n_clusters = len(np.unique(cluster_labels))
    sampled_ocean = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_ocean = ocean_df.loc[cluster_mask, ocean_cols]
        
        if method == 'gmm_sample':
            # Генерируем из многомерного гауссиана данного кластера
            mean = gmm_model.means_[cluster_id]
            covariance = gmm_model.covariances_[cluster_id]
            
            # Генерируем сэмпл в стандартизованном пространстве
            sample_scaled = np.random.multivariate_normal(mean, covariance, size=1)[0]
            
            # Обратное преобразование в оригинальное пространство
            sample = scaler.inverse_transform(sample_scaled.reshape(1, -1))[0]
            
            # Ограничиваем значения диапазоном [0, 1]
            ocean_sample = {}
            for i, col in enumerate(ocean_cols):
                min_val = cluster_ocean[col].min()
                max_val = cluster_ocean[col].max()
                ocean_sample[col] = np.clip(sample[i], min_val, max_val)
        
        elif method == 'cluster_mean':
            # Берем среднее значение кластера
            ocean_sample = cluster_ocean.mean().to_dict()
        
        elif method == 'cluster_random':
            # Берем случайную персону из кластера
            random_idx = np.random.choice(cluster_ocean.index)
            ocean_sample = cluster_ocean.loc[random_idx].to_dict()
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        sampled_ocean.append(ocean_sample)
    
    logger.debug(f"Сэмплировано OCEAN значений для {n_clusters} кластеров методом {method}")
    return sampled_ocean


def calculate_or_get_ocean(nemo_full, indices, ta_name="unknown"):
    """
    Получает OCEAN значения из DataFrame или рассчитывает их если отсутствуют
    """

    if not all(col in nemo_full.columns for col in ocean_cols):
        logger.info(f"[TA:{ta_name}] OCEAN колонки отсутствуют в nemo_full, создаем их")
        for col in ocean_cols:
            if col not in nemo_full.columns:
                nemo_full[col] = np.nan

    matched_americans = nemo_full.iloc[indices]


    mask_null = matched_americans[ocean_cols].isnull().any(axis=1)
    indices_to_calc = matched_americans[mask_null].index.tolist()
    
    if len(indices_to_calc) > 0:
        americans_to_calc = nemo_full.iloc[indices_to_calc]
        _, ocean_df_calc = calculate_ocean_profiles(americans_to_calc)
        

        for col in ocean_cols:
            if col in ocean_df_calc.columns:
                nemo_full.loc[indices_to_calc, col] = ocean_df_calc[col].values
        
        logger.info(f"[TA:{ta_name}] OCEAN рассчитаны для {len(indices_to_calc)} персон")
    

    ocean_df = nemo_full.loc[indices, ocean_cols].copy()
    
    return ocean_df


def replicate_personas_with_gmm(russian_df, russian_norm, american_norm, 
                                nemo_full, group_cols, gmm_params, 
                                top_n_categories=10, use_gower=False, 
                                sampling_method='gmm_sample', ta_name="unknown"):
    """
    Главная функция: группирует RU персон, находит американцев, кластеризует GMM и создает копии
    """
    logger.info(f"[TA:{ta_name}] Начало репликации персон для целевой аудитории")
    nemo_full = load_american_data(nemo_full.shape[0])
    
    # 1. Делим RU ЦА на подгруппы
    grouped = group_russian_personas(russian_norm, group_cols)
    logger.info(f"[TA:{ta_name}] Обнаружено {len(grouped)} социо-демографических групп")

    replicated_personas, clustering_stats = [], []
    
    for group_key, group_info in tqdm(grouped.items(), desc=f"Processing groups for {ta_name}"):
        ru_indices = group_info['indices']
        
        logger.info(f"[TA:{ta_name}] Обработка группы: {group_info['values']}, {group_info['size']} персон")

        # 2. Для каждой подгруппы ищем наиболее похожих американцев 
        american_indices, top_categories, top_dists = find_americans_for_group(
            american_norm, group_key, group_cols, 
            top_n=top_n_categories, use_gower=use_gower
        )
        
        if len(american_indices) == 0:
            logger.warning(f"[TA:{ta_name}] Не найдено американцев для группы {group_key}, пропускаем")
            continue
        
        logger.info(f"[TA:{ta_name}] Найдено {len(american_indices)} американцев из {len(top_categories)} категорий")

        # 3. Получаем/рассчитываем OCEAN для найденных американцев
        ocean_df = calculate_or_get_ocean(nemo_full, american_indices, ta_name)
        
        # 4. GMM кластеризация
        
        try:
            cluster_labels, gmm_model, n_components, bic, scaler = cluster_ocean_gmm(ocean_df, **gmm_params)
        except Exception as e:
            logger.error(f"[TA:{ta_name}] Ошибка GMM кластеризации: {e}, используем 1 кластер")
            n_components = 1
            cluster_labels = np.zeros(len(ocean_df))
            gmm_model = None
            scaler = None
            bic = None
        
        # 5. Сэмплируем OCEAN
        if gmm_model is not None:
            sampled_ocean = sample_ocean_from_gmm(
                gmm_model, cluster_labels, ocean_df, scaler, 
                method=sampling_method
            )
        else:
            sampled_ocean = [ocean_df.mean().to_dict()]
        
        # Статистика
        clustering_stats.append({
            'target_audience': ta_name,
            'group_key': str(group_key),
            'group_values': group_info['values'],
            'n_ru_personas': group_info['size'],
            'n_americans': len(american_indices),
            'n_categories': len(top_categories),
            'n_clusters': n_components,
            'bic_score': bic,
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_components)],
            'avg_distance': float(top_dists.mean())
        })
        
        # 6. Создаем копии RU персон
        for ru_idx in ru_indices:
            ru_person = russian_df.iloc[ru_idx].copy()
            
            for cluster_id, ocean_vals in enumerate(sampled_ocean):
                persona_copy = ru_person.to_dict()
                persona_copy['cluster_id'] = cluster_id
                persona_copy['group_key'] = str(group_key)
                persona_copy['n_americans_in_group'] = len(american_indices)
                persona_copy['n_clusters_total'] = n_components
                
                for ocean_trait, val in ocean_vals.items():
                    persona_copy[ocean_trait] = val
                
                cluster_size = np.sum(cluster_labels == cluster_id)
                persona_copy['cluster_weight'] = float(cluster_size / len(cluster_labels))
                
                replicated_personas.append(persona_copy)
        
        logger.info(f"[TA:{ta_name}] Создано {group_info['size'] * n_components} реплицированных персон для группы")
    
    result_df = pd.DataFrame(replicated_personas)
    stats_df = pd.DataFrame(clustering_stats)
    
    # 7. Сохраняем обновленный nemo если были изменения
    logger.info(f"[TA:{ta_name}] Сохранение обновленного nemo с OCEAN значениями...")
    try:
        nemo_save_path = DATA_PATHS.get('nemotron', './data/Nemotron/nemotron_with_ocean.parquet')
        nemo_full.to_parquet(nemo_save_path, index=False)
        logger.info(f"[TA:{ta_name}] nemo_full сохранен в: {nemo_save_path}")
    except Exception as e:
        logger.error(f"[TA:{ta_name}] Ошибка сохранения nemo: {e}")
        try:
            csv_path = nemo_save_path.replace('.parquet', '.csv')
            nemo_full.to_csv(csv_path, index=False)
            logger.info(f"[TA:{ta_name}] nemo_full сохранен в CSV: {csv_path}")
        except Exception as e2:
            logger.error(f"[TA:{ta_name}] Ошибка сохранения в CSV: {e2}")
    
    logger.info(f"[TA:{ta_name}] Репликация завершена: создано {len(result_df)} персон, {len(clustering_stats)} групп")
    
    # Сводная статистика
    if len(clustering_stats) > 0:
        avg_personas = stats_df['n_ru_personas'].mean()
        avg_americans = stats_df['n_americans'].mean()
        avg_clusters = stats_df['n_clusters'].mean()
        logger.info(f"[TA:{ta_name}] Средняя статистика: {avg_personas:.1f} персон/группу, {avg_americans:.1f} американцев/группу, {avg_clusters:.1f} кластеров/группу")
    
    return result_df, stats_df


def analyze_cluster_characteristics(result_df, ocean_cols=['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']):
    """
    Анализирует характеристики найденных кластеров
    """
    logger.info("Анализ характеристик кластеров по OCEAN")
    
    for group_key in result_df['group_key'].unique():
        group_data = result_df[result_df['group_key'] == group_key]
        
        logger.info(f"Группа: {group_key}")
        
        for cluster_id in group_data['cluster_id'].unique():
            cluster_data = group_data[group_data['cluster_id'] == cluster_id]
            
            cluster_info = []
            for col in ocean_cols:
                if col in cluster_data.columns:
                    mean_val = cluster_data[col].mean()
                    cluster_info.append(f"{col}: {mean_val:.3f}")
            
            logger.info(f"  Кластер {cluster_id}: {', '.join(cluster_info)}")