from typing import TypedDict
from langgraph.graph import StateGraph, END
from yandex_chain import YandexLLM
from src.schemas.credit_schema import persona_response_schema
from config import LLM_MODEL, LLM_TEMPERATURE
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
import json
import re

load_dotenv()


credit_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Ты — эксперт по поведенческой экономике и кредитному скорингу.\n"
         "Твоя задача — предсказать, согласится ли российская персона взять кредит при call-to-action в банковском приложении.\n"
         "Верни ответ ТОЛЬКО в формате JSON, который соответствует схеме persona_response_schema."
         ),
        ("human", 
         "РОССИЙСКАЯ ПЕРСОНА:\n{russian_person}\n\n"
         "ЗАДАЧА: Предскажи, возьмет ли эта российская персона кредит при показе предложения в банковском приложении."
         )
    ])

class CreditDecisionState(TypedDict):
    """State for credit decision making"""
    russian_person: dict
    credit_decision: dict

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

def predict_credit_decision(russian_person: dict) -> dict:
    """Credit decision agent"""
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
        
    prompt = credit_prompt.partial(russian_person=russian_person)
    chain = prompt | llm
    response = chain.invoke({})
    
    # Парсим JSON из ответа
    if isinstance(response, str):
        response_text = response
    elif hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)
    json_data = _parse_json_from_response(response_text)
    
    # persona_response_schema - это словарь JSON схемы, не Pydantic модель
    # Просто возвращаем распарсенные данные
    validated_response = json_data
    
    return {
        'credit_decision': validated_response
    }