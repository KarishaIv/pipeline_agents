"""FastAPI-сервер для мета-агента.

Запуск:
    uvicorn src.scripts.serve_meta_agent:app --host 0.0.0.0 --port 8000
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import logging
import warnings

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="Mixing V1 models and V2 models",
    category=UserWarning,
)

from fastapi import FastAPI, Query

from src.meta_agent.meta_agent import invoke

app = FastAPI(title="Meta Agent API")


@app.get("/ask")
async def ask(q: str = Query(..., description="Вопрос для мета-агента")):
    answer = await invoke(q)
    return {"answer": answer}
