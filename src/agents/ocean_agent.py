from typing import TypedDict
import pandas as pd
from langchain_openai import ChatOpenAI
from src.schemas.ocean_schema import ocean_schema
from config import LLM_MODEL, LLM_TEMPERATURE
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


ocean_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an expert psychologist specializing in personality assessment using the OCEAN (Big Five) model.\n"
        "Given a person's profile, analyze and evaluate their likely personality traits according to the OCEAN framework:\n"
         ),
        ("human", "Profile: {profile}")
    ])


class OceanState(TypedDict):
    neighbor_profiles: pd.DataFrame
    enriched_neighbors: pd.DataFrame
    ocean_df: pd.DataFrame

def calculate_ocean_profiles(neighbor_profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate OCEAN profiles for a set of neighbor profiles."""
    llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE'),
        )
    chain = ocean_prompt | llm.with_structured_output(ocean_schema)

    results = []
    for _, row in neighbor_profiles.iterrows():
        response = chain.invoke({'profile': row.to_dict()})
        results.append(response)

    ocean_df = pd.DataFrame(results, index=neighbor_profiles.index)
    enriched_neighbors = pd.concat([neighbor_profiles, ocean_df], axis=1)
    return enriched_neighbors, ocean_df