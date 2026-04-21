"""FastAPI-сервер для мета-агента.

Запуск:
    uvicorn src.scripts.serve_meta_agent:app --host 0.0.0.0 --port 8000

LangSmith tracing (optional):
    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=<your-key>
    export LANGCHAIN_PROJECT=meta-agent
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import logging
import warnings
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sgr_agent_core").setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message="Mixing V1 models and V2 models",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
)

from fastapi import FastAPI, Query

from src.meta_agent import meta_graph_manager


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    yield
    await meta_graph_manager.aclose()


app = FastAPI(title="Meta Agent API", lifespan=app_lifespan)


@app.get("/ask")
async def ask(
    q: str = Query(..., description="Вопрос для мета-агента"),
    thread_id: str | None = Query(
        default=None,
        description="Идентификатор сессии: null — продолжить предыдущую, -1 — начать новую, иначе — использовать переданный",
    ),
):
    out = await meta_graph_manager.invoke_graph_session(q, thread_id)
    return {"answer": out.answer, "thread_id": out.thread_id}
