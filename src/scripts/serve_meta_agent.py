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
from pathlib import Path

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
from fastapi.responses import FileResponse

from src.meta_agent import (
    AskRequest,
    MetaAgentApiResponse,
    meta_graph_manager,
)
from src.meta_agent.configs import CHARTS_DIR

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
        MetaAgentApiResponse with thread_id and list of outputs (graph-native outputs).

    Raises:
        HTTPException: On LLM, Qdrant, or graph execution errors.
    """
    try:
        result = await meta_graph_manager.invoke_graph_session(
            request.question, request.thread_id
        )
        # result.outputs contains the ordered AgentOutput list from the graph
        return MetaAgentApiResponse(thread_id=result.thread_id, outputs=result.outputs)
    except Exception as e:
        logger.exception("Error in /ask endpoint")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


@app.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str) -> FileResponse:
    """Serve an artifact (chart, file, etc.) by ID with security validation.

    Args:
        artifact_id: The artifact filename (just the name, not a path).

    Returns:
        FileResponse with the artifact content.

    Raises:
        HTTPException: If artifact not found or path traversal attempt detected.
    """
    # Validate that artifact_id doesn't contain path traversal characters
    if "/" in artifact_id or "\\" in artifact_id or artifact_id.startswith("."):
        logger.warning("Path traversal attempt: %s", artifact_id)
        raise HTTPException(status_code=400, detail="Invalid artifact ID")

    artifact_path = (CHARTS_DIR / artifact_id).resolve()

    # Security check: ensure the resolved path is within CHARTS_DIR
    try:
        artifact_path.relative_to(CHARTS_DIR.resolve())
    except ValueError:
        logger.warning("Path traversal detected for artifact: %s", artifact_id)
        raise HTTPException(status_code=403, detail="Access denied")

    if not artifact_path.exists():
        logger.warning("Artifact not found: %s", artifact_path)
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Determine MIME type based on file extension
    mime_type = "application/octet-stream"
    if artifact_path.suffix.lower() == ".png":
        mime_type = "image/png"
    elif artifact_path.suffix.lower() in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    elif artifact_path.suffix.lower() == ".pdf":
        mime_type = "application/pdf"
    elif artifact_path.suffix.lower() == ".csv":
        mime_type = "text/csv"
    elif artifact_path.suffix.lower() == ".json":
        mime_type = "application/json"

    return FileResponse(
        artifact_path,
        media_type=mime_type,
        filename=artifact_path.name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.scripts.serve_meta_agent:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
