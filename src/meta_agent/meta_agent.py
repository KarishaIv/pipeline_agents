import json
import os
from typing import List, Optional, TYPE_CHECKING

from openai import AsyncOpenAI
from pydantic import Field

from sgr_agent_core import AgentConfig
from sgr_agent_core.agents.iron_agent import IronAgent
from sgr_agent_core.base_tool import BaseTool
import sgr_agent_core.tools as sgr_tools

from src.meta_agent.tools import (
    search as qdrant_search,
    filter_points as qdrant_filter,
    scroll_points as qdrant_scroll,
    retrieve_by_id as qdrant_retrieve,
    AVAILABLE_COLLECTIONS,
    COLLECTION_ENUM_DESC,
)

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig as AgentConfigType
    from sgr_agent_core.models import AgentContext

META_AGENT_SYSTEM_PROMPT = (
    "Ты мета-агент, который отвечает на вопросы, "
    "используя инструменты для поиска данных в векторной базе данных Qdrant. "
    f"Доступные коллекции: {', '.join(AVAILABLE_COLLECTIONS)}. "
    "Используй инструменты поиска для нахождения релевантных данных, "
    "затем вызови FinalAnswerTool с готовым ответом."
)


# ---------------------------------------------------------------------------
# Custom BaseTool wrappers around the Qdrant helper functions
# ---------------------------------------------------------------------------

class QdrantSearchTool(BaseTool):
    """Semantic search over a Qdrant collection via cosine similarity."""

    tool_name = "search"

    reasoning: str = Field(description="Why this search is needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    query: str = Field(description="Natural-language search string")
    vector_name: str = Field(
        default="embedding",
        description=(
            "Named vector to search against. "
            "questions/personas/target_audiences use 'embedding'. "
            "simulations uses: emotional_vector, rational_vector, "
            "social_vector, ideological_vector, decision_vector, general_vector"
        ),
    )
    limit: int = Field(default=5, description="Maximum number of results")

    async def __call__(self, context, config, **_) -> str:
        result = qdrant_search(
            collection=self.collection,
            query=self.query,
            vector_name=self.vector_name,
            limit=self.limit,
        )
        return json.dumps(result, ensure_ascii=False, default=str)


class QdrantFilterTool(BaseTool):
    """Filter a Qdrant collection by an exact payload field match."""

    tool_name = "filter_points"

    reasoning: str = Field(description="Why this filter is needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    field: str = Field(description='Payload field name to filter on (e.g. "question", "name")')
    value: str = Field(description="Expected exact value of the field")
    limit: int = Field(default=10, description="Maximum number of results")

    async def __call__(self, context, config, **_) -> str:
        result = qdrant_filter(
            collection=self.collection,
            field=self.field,
            value=self.value,
            limit=self.limit,
        )
        return json.dumps(result, ensure_ascii=False, default=str)


class QdrantScrollTool(BaseTool):
    """Paginated scroll through all points in a Qdrant collection."""

    tool_name = "scroll_points"

    reasoning: str = Field(description="Why paginated scroll is needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    limit: int = Field(default=10, description="Page size")
    offset: Optional[int] = Field(
        default=None,
        description="Point id to start from (from previous next_offset)",
    )

    async def __call__(self, context, config, **_) -> str:
        result = qdrant_scroll(
            collection=self.collection,
            limit=self.limit,
            offset=self.offset,
        )
        return json.dumps(result, ensure_ascii=False, default=str)


class QdrantRetrieveTool(BaseTool):
    """Retrieve specific points from a Qdrant collection by their integer IDs."""

    tool_name = "retrieve_by_id"

    reasoning: str = Field(description="Why these specific points are needed")
    collection: str = Field(description=COLLECTION_ENUM_DESC)
    ids: List[int] = Field(description="List of integer point IDs to retrieve")

    async def __call__(self, context, config, **_) -> str:
        result = qdrant_retrieve(collection=self.collection, ids=self.ids)
        return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(question: str) -> IronAgent:
    folder_id = os.getenv("YANDEX_FOLDER_ID", "")
    model_uri = f"gpt://{folder_id}/yandexgpt/latest"
    api_key = os.getenv("YANDEX_API_KEY", "")
    base_url = "https://llm.api.cloud.yandex.net/v1"

    agent_config = AgentConfig()
    agent_config.llm.api_key = api_key
    agent_config.llm.base_url = base_url
    agent_config.llm.model = model_uri
    agent_config.prompts.system_prompt_str = META_AGENT_SYSTEM_PROMPT

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    return IronAgent(
        task_messages=[{"role": "user", "content": question}],
        openai_client=openai_client,
        agent_config=agent_config,
        toolkit=[
            QdrantSearchTool,
            QdrantFilterTool,
            QdrantScrollTool,
            QdrantRetrieveTool,
            sgr_tools.FinalAnswerTool,
        ],
    )


async def invoke(question: str) -> str:
    """Run the meta-agent tool-calling loop and return the final answer."""
    agent = _build_agent(question)
    result = await agent.execute()
    if isinstance(result, str):
        return result
    if hasattr(result, "answer"):
        return result.answer
    return str(result)
