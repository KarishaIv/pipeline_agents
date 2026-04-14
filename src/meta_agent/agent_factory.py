"""Жизненный цикл агента: создание, выполнение и извлечение результата."""

import asyncio
import json
import logging
import os
from typing import Any, NamedTuple

from langsmith.wrappers import wrap_openai
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

from sgr_agent_core import AgentConfig
from sgr_agent_core.agents.tool_calling_agent import ToolCallingAgent
from sgr_agent_core.base_tool import BaseTool

from config import get_model_uri, YANDEX_BASE_URL
logger = logging.getLogger("meta_agent")

TRANSIENT_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
MAX_RETRIES = 3
RETRY_BACKOFF = (5, 10, 30)


class AgentRunResult(NamedTuple):
    raw: str | "TransientError"
    custom_context: dict[str, Any] | None


class TransientError:
    """Сигнал, который попадает в AgentRunResult.raw, когда исчерпаны все ретраи
    при временной сетевой/API-ошибке. Узлы графа могут проверить его и
    переиспользовать в логике обработки вместо обычного результата."""

    def __init__(self, message: str, attempts: int):
        self.message = message
        self.attempts = attempts

    def __str__(self) -> str:
        return f"TransientError after {self.attempts} attempts: {self.message}"


def _make_openai_client() -> AsyncOpenAI:
    """Создать AsyncOpenAI-клиент и, при включённом tracing, обернуть его для LangSmith."""
    api_key = os.getenv("YANDEX_API_KEY", "")
    client = AsyncOpenAI(api_key=api_key, base_url=YANDEX_BASE_URL)
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        client = wrap_openai(client)
    return client


def make_agent(
    task: str,
    system_prompt: str,
    toolkit: list,
    *,
    name: str = "agent",
    initial_custom_context: dict[str, Any] | None = None,
) -> ToolCallingAgent:
    """Создать экземпляр ToolCallingAgent на базе Yandex LLM."""
    api_key = os.getenv("YANDEX_API_KEY", "")

    cfg = AgentConfig()
    cfg.llm.api_key = api_key
    cfg.llm.base_url = YANDEX_BASE_URL
    cfg.llm.model = get_model_uri()
    cfg.prompts.system_prompt_str = system_prompt

    agent = ToolCallingAgent(
        task_messages=[{"role": "user", "content": task}],
        openai_client=_make_openai_client(),
        agent_config=cfg,
        toolkit=toolkit,
        def_name=name,
    )
    if initial_custom_context is not None:
        agent._context.custom_context = initial_custom_context
    return agent


def unwrap(result) -> str | None:
    """Извлечь строковый ответ из результата выполнения агента."""
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if hasattr(result, "answer"):
        return result.answer
    return str(result)


async def run_agent(
    task: str,
    system_prompt: str,
    toolkit: list,
    *,
    name: str,
    initial_custom_context: dict[str, Any] | None = None,
) -> AgentRunResult:
    """Запустить ToolCallingAgent с ретраями при временных ошибках.

    Возвращает AgentRunResult:
    - raw: строка ответа, JSON-ошибка или TransientError;
    - custom_context: финальный custom_context после выполнения агента.
    """
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            agent = make_agent(
                task,
                system_prompt,
                toolkit,
                name=name,
                initial_custom_context=initial_custom_context,
            )
            raw = unwrap(await agent.execute())
            if raw is None:
                return AgentRunResult(
                    raw=json.dumps({"error": "Агент не вернул результат"}, ensure_ascii=False),
                    custom_context=agent._context.custom_context,
                )
            return AgentRunResult(raw=raw, custom_context=agent._context.custom_context)

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
            logger.exception("Агент '%s' завершился невременной ошибкой", name)
            return AgentRunResult(
                raw=json.dumps({"error": str(exc)}, ensure_ascii=False),
                custom_context=initial_custom_context,
            )

    return AgentRunResult(
        raw=TransientError(message=str(last_exc), attempts=MAX_RETRIES),
        custom_context=initial_custom_context,
    )


