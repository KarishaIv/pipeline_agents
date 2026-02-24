import os

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from src.meta_agent.tools import qdrant_tools

META_AGENT_SYSTEM_PROMPT = (
    "Ты мета-агент, который отвечает на вопросы, "
    "используя инструменты для поиска вопросов в базе данных."
)

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt/latest"

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://llm.api.cloud.yandex.net/v1",
    model=MODEL_URI,
)

graph = create_react_agent(
    model=llm,
    tools=qdrant_tools,
    prompt=META_AGENT_SYSTEM_PROMPT,
)
