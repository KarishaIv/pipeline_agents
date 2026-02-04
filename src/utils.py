import pandas as pd
import numpy as np
from config import *
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Any
import logging 
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from yandex_chain import YandexLLM
from pydantic import BaseModel
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def _get_text_from_response(response) -> str:
    """Универсальная функция для получения текста из ответа LLM"""
    # YandexLLM может возвращать строку напрямую или объект с .content
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        return response.content
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)

def _extract_values_from_nested_dict(data: Any) -> Any:
    """Извлекает фактические значения из вложенных структур JSON.
    
    Если значение - это словарь с ключами 'description' или 'value', 
    извлекает значение из него. Иначе возвращает как есть.
    """
    if isinstance(data, dict):
        # Если это словарь с описанием схемы (содержит 'type' и 'description'), 
        # пытаемся извлечь значение
        if 'type' in data and 'description' in data:
            # Это похоже на JSON schema описание, извлекаем description как значение
            desc = data.get('description', '')
            if isinstance(desc, str):
                return desc
            # Если description - тоже словарь, рекурсивно обрабатываем
            return _extract_values_from_nested_dict(desc)
        # Если есть ключ 'value', используем его
        elif 'value' in data:
            return _extract_values_from_nested_dict(data['value'])
        # Рекурсивно обрабатываем все значения словаря
        else:
            return {k: _extract_values_from_nested_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_extract_values_from_nested_dict(item) for item in data]
    else:
        return data

def _parse_json_from_response(text: str) -> dict:
    """Извлекает JSON из текстового ответа LLM"""
    # Пытаемся найти JSON в ответе
    # Ищем блоки между { и }
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            # Извлекаем значения из вложенных структур
            return _extract_values_from_nested_dict(parsed)
        except json.JSONDecodeError:
            pass
    
    # Если не нашли, пытаемся распарсить весь текст
    try:
        parsed = json.loads(text)
        # Извлекаем значения из вложенных структур
        return _extract_values_from_nested_dict(parsed)
    except json.JSONDecodeError:
        logger.warning(f"Не удалось распарсить JSON из ответа: {text[:100]}")
        return {}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def robust_llm_call(prompt: str, model = None, temperature: float = 0.5, structured_output: Optional[BaseModel]=None):
    import os
    from yandex_chain import YandexGPTModel
    # Используем модель из конфига по умолчанию
    if model is None:
        model = LLM_MODEL
    
    # Настройка Yandex GPT
    api_key = os.getenv("OPENAI_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    
    # Создаем YandexLLM с folder_id только если он указан
    llm_params = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature
    }
    if folder_id:
        llm_params["folder_id"] = folder_id
    
    llm = YandexLLM(**llm_params)
    
    # Добавляем инструкцию для структурированного вывода
    if structured_output:
        # Получаем JSON схему (поддерживаем Pydantic v1 и v2)
        try:
            if hasattr(structured_output, 'model_json_schema'):
                schema = structured_output.model_json_schema()  # Pydantic v2
            elif hasattr(structured_output, 'schema'):
                schema = structured_output.schema()  # Pydantic v1
            else:
                schema = {}
            schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        except Exception:
            schema_json = "{}"
        
        enhanced_prompt = f"""{prompt}

ВАЖНО: Верни ответ ТОЛЬКО в формате JSON с ПРЯМЫМИ ЗНАЧЕНИЯМИ (не схемой!).
Пример правильного формата:
{{"goal_description": "Текст цели", "motivation": "Текст мотивации"}}

НЕПРАВИЛЬНО (не делай так):
{{"goal_description": {{"description": "Текст", "type": "string"}}, "motivation": {{"description": "Текст", "type": "string"}}}}

Схема для справки (но возвращай только значения):
{schema_json}

Ответ должен быть валидным JSON объектом с прямыми значениями полей, без дополнительного текста."""
        response = await asyncio.to_thread(lambda: llm.invoke(enhanced_prompt))
        response_text = _get_text_from_response(response)
        json_data = _parse_json_from_response(response_text)
        # Валидируем через Pydantic
        try:
            if hasattr(structured_output, 'model_validate'):
                return structured_output.model_validate(json_data)  # Pydantic v2
            else:
                return structured_output(**json_data)  # Pydantic v1
        except Exception as e:
            logger.warning(f"Ошибка валидации Pydantic модели: {e}, возвращаем сырые данные")
            return json_data
    else:
        response = await asyncio.to_thread(lambda: llm.invoke(prompt))
        return _get_text_from_response(response)


def filter_real_russian_data(evidence: Dict, df_russian_preprocessed: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """
    Фильтрация реальных российских данных по evidence
    
    Args:
        evidence: Словарь с условиями фильтрации
        df_russian_preprocessed: Предобработанные российские данные
        sample_size: Максимальное количество возвращаемых записей
    
    Returns:
        Отфильтрованный DataFrame
    """
    df_filtered = df_russian_preprocessed.copy()
    
    normalized_evidence = normalize_evidence(evidence)
    evidence_dict = dict(normalized_evidence)
    
    for key, value in evidence_dict.items():
        if key in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[key] == value]
    
    if len(df_filtered) > sample_size:
        df_filtered = df_filtered.sample(n=sample_size, random_state=42)
    
    return df_filtered
    

def normalize_features(russian_df, american_df):
    """
    Предобрабатывает исходный DataFrame для построения вероятностной графовой модели.
    - Биннинг возраста (age_group)
    - Категоризация дохода (income_level)
    - Группировка по количеству детей (children_group)
    - Классификация регионов (region_type)
    Возвращает подготовленный DataFrame.

    TODO: важно сделать нормальную кодировку под реальные данные 
    """
    russian_normalized = pd.DataFrame({
        'age_group': russian_df['age_group'].values,
        'gender': russian_df['gender'].map(GENDER_MAPPING).values,
        'education': russian_df['education'].map(EDUCATION_MAPPING).values,
        'marital_status': russian_df['marital_status'].map(MARITAL_MAPPING).values,
    })


    american_normalized = pd.DataFrame({
        'age_group': american_df['age'].apply(
            lambda x: np.digitize(x, PGM_CONFIG['age_bins'])
        ).values,
        'gender': american_df['sex'].map(GENDER_MAPPING).values,
        'education': american_df['education_level'].map(EDUCATION_MAPPING).values,
        'marital_status': american_df['marital_status'].map(MARITAL_MAPPING).values,
    })

    return russian_normalized, american_normalized


def normalize_evidence(evidence: dict):
    """Normalize evidence to match training data format"""
    normalized_evidence = []
    
    for key, value in evidence.items():
        if key == 'age_group':
            age = int(value)
            age_group = np.digitize(age, PGM_CONFIG['age_bins'])
            normalized_evidence.append((key, age_group))

        elif key in MATCH_COLS:
            normalized_evidence.append((key, value))
    
    return normalized_evidence


def prepare_request_from_evidence(evidence: dict):
    """Prepare request dict for matching from evidence"""
    normalized = normalize_evidence(evidence)
    request_dict = {}
    
    for key, value in normalized:
        if key in MATCH_COLS:
            if key == 'education':
                mapped_education = EDUCATION_MAPPING.get(value, value)
                request_dict[key] = mapped_education
            elif key == 'marital_status':
                mapped_marital = MARITAL_MAPPING.get(value, value)
                request_dict[key] = mapped_marital
            elif key == 'gender':
                mapped_gender = GENDER_MAPPING.get(value, value)
                request_dict[key] = mapped_gender
            else:
                request_dict[key] = value
    
    return request_dict



def translate_ocean_score(score: Optional[float], dimension: str) -> str:
    """
    Переводит числовой OCEAN скор (0-10) в текстовое описание.
    
    Каждое измерение имеет свой диапазон интерпретаций (5 уровней).
    Если скор отсутствует → "не указано"
    
    Args:
        score: Числовой скор 0-10 (или None)
        dimension: Тип измерения (openness, conscientiousness, extraversion, 
                  agreeableness, neuroticism)
    
    Returns:
        Текстовое описание личностной черты
    """
    
    if score is None:
        return "не указано"
    
    try:
        score = float(score)
    except (ValueError, TypeError):
        return "не указано"
    
    if dimension == "openness":
        if score < 3:
            return "закрыт от нового, предпочитает традиции"
        elif score < 5:
            return "осторожен к новому, предпочитает проверенное"
        elif score < 7:
            return "умеренно открыт новому"
        elif score < 9:
            return "открыт новому опыту и идеям"
        else:
            return "очень открыт новому, любит экспериментировать"
    
    elif dimension == "conscientiousness":
        if score < 3:
            return "спонтанен, импульсивен, без планов"
        elif score < 5:
            return "гибкий подход, не всегда организован"
        elif score < 7:
            return "обычно ответственен и организован"
        elif score < 9:
            return "дисциплинирован, пунктуален, ответственен"
        else:
            return "очень организован, планирует всё до деталей"
    
    elif dimension == "extraversion":
        if score < 3:
            return "интроверт, предпочитает одиночество"
        elif score < 5:
            return "склонен к одиночеству, избегает социума"
        elif score < 7:
            return "умеренно общителен, зависит от настроения"
        elif score < 9:
            return "общительный, энергичный, легко находит друзей"
        else:
            return "очень общителен, постоянно ищет социального взаимодействия"
    
    elif dimension == "agreeableness":
        if score < 3:
            return "независимый, конкурентный, не обеспокоен чужим мнением"
        elif score < 5:
            return "предпочитает независимость, может быть критичен"
        elif score < 7:
            return "склонен к сотрудничеству, хороший товарищ"
        elif score < 9:
            return "альтруист, эмпатичен, готов помочь"
        else:
            return "очень эмпатичен, ставит нужды других выше своих"
    
    elif dimension == "neuroticism":
        if score < 3:
            return "эмоционально стабилен, спокоен при стрессе"
        elif score < 5:
            return "обычно спокоен, но может реагировать на сложности"
        elif score < 7:
            return "умеренно чувствителен к стрессу"
        elif score < 9:
            return "чувствителен к стрессу, может быть тревожен"
        else:
            return "очень чувствителен, часто переживает и беспокоится"
    
    else:
        return "неизвестное измерение"


from pandarallel import pandarallel
import pandas as pd
pandarallel.initialize(progress_bar=False, nb_workers=4) 

def translate_ocean_to_readable(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    
    dimensions = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    
    for dim in dimensions:
        if dim in df.columns:
            result_df[f"{dim}"] = df[dim].parallel_apply(
                lambda x: translate_ocean_score(x, dim)
            )
            
    return result_df


def get_income_range(income: float, bins, labels) -> str:
    """Преобразует числовой доход в диапазон с меткой."""
    import numpy as np
    
    for i in range(len(bins) - 1):
        if bins[i] <= income < bins[i + 1]:
            label = labels[i]
            lower = int(bins[i])
            upper = int(bins[i + 1]) if bins[i + 1] != np.inf else "+"
            return f"{label}: {f'{lower:,}'.replace(',', ' ')}-{upper}"
    
    label = labels[-1]
    lower = int(bins[-2])
    return f"{label}: {f'{lower:,}'.replace(',', ' ')}+"