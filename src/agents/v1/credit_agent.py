from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from src.schemas.credit_schema import persona_response_schema
from config import LLM_MODEL, LLM_TEMPERATURE
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


credit_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Ты — эксперт по поведенческой экономике и кредитному скорингу.\n"
         "Твоя задача — предсказать, согласится ли российская персона взять кредит при call-to-action в банковском приложении.\n\n"
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

def predict_credit_decision(russian_person: dict) -> dict:
    """Credit decision agent"""
    llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE'),
        )
        
    prompt = credit_prompt.partial(russian_person=russian_person)
    chain = prompt | llm.with_structured_output(persona_response_schema)
    response = chain.invoke({})
    return {
        'credit_decision': response
    }