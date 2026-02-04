from typing import TypedDict
import pandas as pd
from yandex_chain import YandexLLM
from src.schemas.ocean_schema import ocean_schema
from config import LLM_MODEL, LLM_TEMPERATURE
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
import json
import re

load_dotenv()


ocean_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an expert psychologist specializing in personality assessment using the OCEAN (Big Five) model.\n"
        "Given a person's profile, analyze and evaluate their likely personality traits according to the OCEAN framework.\n"
        "Return ONLY a valid JSON object with scores for openness, conscientiousness, extraversion, agreeableness, and neuroticism (each 0-10).\n"
        "Example format: {{\"openness\": 7, \"conscientiousness\": 8, \"extraversion\": 6, \"agreeableness\": 7, \"neuroticism\": 4}}"
         ),
        ("human", "Profile: {profile}")
    ])


class OceanState(TypedDict):
    neighbor_profiles: pd.DataFrame
    enriched_neighbors: pd.DataFrame
    ocean_df: pd.DataFrame

def _parse_json_from_response(text: str) -> dict:
    """Извлекает JSON из текстового ответа LLM"""
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def calculate_ocean_profiles(neighbor_profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate OCEAN profiles for a set of neighbor profiles."""
    api_key = os.getenv('OPENAI_API_KEY')
    folder_id = os.getenv('YANDEX_FOLDER_ID')
    
    llm_params = {
        "api_key": api_key,
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }
    if folder_id:
        llm_params["folder_id"] = folder_id
    
    llm = YandexLLM(**llm_params)
    chain = ocean_prompt | llm

    results = []
    for _, row in neighbor_profiles.iterrows():
        response = chain.invoke({'profile': row.to_dict()})
        # Парсим JSON из ответа (универсальная обработка разных типов ответов)
        if isinstance(response, str):
            response_text = response
        elif hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        json_data = _parse_json_from_response(response_text)
        # Валидируем по схеме
        validated = {
            'openness': json_data.get('openness', 5),
            'conscientiousness': json_data.get('conscientiousness', 5),
            'extraversion': json_data.get('extraversion', 5),
            'agreeableness': json_data.get('agreeableness', 5),
            'neuroticism': json_data.get('neuroticism', 5)
        }
        results.append(validated)

    ocean_df = pd.DataFrame(results, index=neighbor_profiles.index)
    enriched_neighbors = pd.concat([neighbor_profiles, ocean_df], axis=1)
    return enriched_neighbors, ocean_df