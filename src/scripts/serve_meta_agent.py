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

from fastapi import FastAPI, HTTPException

from src.meta_agent import (
    AskRequest,
    MetaAgentApiResponse,
    TextOutput,
    meta_graph_manager,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    yield
    await meta_graph_manager.aclose()


app = FastAPI(title="Meta Agent API", lifespan=app_lifespan)


@app.post("/ask")
async def ask_json(request: AskRequest) -> MetaAgentApiResponse:
    """Ask the meta-agent a question via structured JSON request.

    Args:
        request: AskRequest containing question and optional thread_id.

    Returns:
        MetaAgentApiResponse with thread_id and list of outputs (primarily text for MVP).

    Raises:
        HTTPException: On LLM, Qdrant, or graph execution errors.
    """
    try:
        result = await meta_graph_manager.invoke_graph_session(
            request.question, request.thread_id
        )
        # MVP: wrap the answer as a single text output
        outputs = [TextOutput(text=result.answer)]
        return MetaAgentApiResponse(thread_id=result.thread_id, outputs=outputs)
    except Exception as e:
        logger.exception("Error in /ask endpoint")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.scripts.serve_meta_agent:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
