"""FastAPI-сервер для мета-агента.

Запуск:
    uvicorn src.scripts.serve_meta_agent:app --host 0.0.0.0 --port 8000
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import warnings
warnings.filterwarnings(
    "ignore",
    message="Mixing V1 models and V2 models",
    category=UserWarning,
)

import asyncio

from fastapi import FastAPI, Query
from tenacity import retry, stop_after_attempt, wait_exponential

from src.meta_agent.meta_agent import invoke

app = FastAPI(title="Meta Agent API")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def _invoke(question: str) -> str:
    return invoke(question)


@app.get("/ask")
async def ask(q: str = Query(..., description="Вопрос для мета-агента")):
    answer = await asyncio.to_thread(_invoke, q)
    return {"answer": answer}
