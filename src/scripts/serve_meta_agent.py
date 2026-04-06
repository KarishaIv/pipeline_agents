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

from src.meta_agent import invoke_graph

app = FastAPI(title="Meta Agent API")


@app.get("/ask")
async def ask(q: str = Query(..., description="Вопрос для мета-агента")):
    answer = await invoke_graph(q)
    return {"answer": answer}
