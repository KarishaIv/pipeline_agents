"""Agent lifecycle: creation, execution, result extraction."""

import asyncio
import json
import logging
import os

from langsmith.wrappers import wrap_openai
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

from sgr_agent_core import AgentConfig
from sgr_agent_core.agents.tool_calling_agent import ToolCallingAgent
from sgr_agent_core.base_tool import BaseTool

import src.meta_agent.lib_patches  # noqa: F401  apply all third-party patches

from config import get_model_uri, YANDEX_BASE_URL
from src.meta_agent.prompts import MAX_HISTORY_CHARS

logger = logging.getLogger("meta_agent")

TRANSIENT_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
MAX_RETRIES = 3
RETRY_BACKOFF = (5, 10, 30)


class TransientError:
    """Sentinel returned by run_agent() when all retries are exhausted on a
    transient network/API error. Graph nodes can check for this to retry the
    node instead of treating the error as a valid result."""

    def __init__(self, message: str, attempts: int):
        self.message = message
        self.attempts = attempts

    def __str__(self) -> str:
        return f"TransientError after {self.attempts} attempts: {self.message}"


def _make_openai_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client, wrapped for LangSmith tracing when enabled."""
    api_key = os.getenv("YANDEX_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key, base_url=YANDEX_BASE_URL)
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        client = wrap_openai(client)
    return client


def make_agent(task: str, system_prompt: str, toolkit: list, *, name: str = "agent") -> ToolCallingAgent:
    """Instantiate a ToolCallingAgent backed by the Yandex LLM."""
    api_key = os.getenv("YANDEX_API_KEY", "")

    cfg = AgentConfig()
    cfg.llm.api_key = api_key
    cfg.llm.base_url = YANDEX_BASE_URL
    cfg.llm.model = get_model_uri()
    cfg.prompts.system_prompt_str = system_prompt

    return ToolCallingAgent(
        task_messages=[{"role": "user", "content": task}],
        openai_client=_make_openai_client(),
        agent_config=cfg,
        toolkit=toolkit,
        def_name=name,
    )


def unwrap(result) -> str | None:
    """Extract a string answer from an agent execution result."""
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if hasattr(result, "answer"):
        return result.answer
    return str(result)


async def run_agent(task: str, system_prompt: str, toolkit: list, *, name: str) -> str | TransientError:
    """Run a ToolCallingAgent with retries on transient failures.

    Returns the agent's answer string on success, or a TransientError sentinel
    when all retry attempts are exhausted on a transient API error.
    Non-transient exceptions still produce a JSON error string.
    """
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            agent = make_agent(task, system_prompt, toolkit, name=name)
            raw = unwrap(await agent.execute())
            if raw is None:
                return json.dumps({"error": "Agent returned no result"}, ensure_ascii=False)
            return raw

        except TRANSIENT_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF[attempt - 1]
                logger.warning(
                    "Agent '%s' hit transient error (attempt %d/%d): %s. Retrying in %ds…",
                    name, attempt, MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Agent '%s' exhausted %d retries on transient error: %s",
                    name, MAX_RETRIES, exc,
                )

        except Exception as exc:
            logger.exception("Agent '%s' failed with non-transient error", name)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    return TransientError(message=str(last_exc), attempts=MAX_RETRIES)


def truncate_history(history: list) -> str:
    """Render history to text, keeping only the tail within MAX_HISTORY_CHARS."""
    parts = [f"[{m['role'].upper()}]: {m['content']}" for m in history]
    text = "\n\n".join(parts)
    if len(text) > MAX_HISTORY_CHARS:
        text = "…(truncated)…\n\n" + text[-MAX_HISTORY_CHARS:]
    return text
