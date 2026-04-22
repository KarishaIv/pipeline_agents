"""Жизненный цикл агента: создание, выполнение и извлечение результата."""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI

from sgr_agent_core import AgentConfig
from sgr_agent_core.agents.tool_calling_agent import ToolCallingAgent

from config import get_model_uri, YANDEX_BASE_URL
from src.meta_agent.config import MAX_AGENT_ITERATIONS
logger = logging.getLogger("meta_agent")


@dataclass
class AgentRunResult:
    output: str
    context: dict[str, Any] | None = None


def _safe_get_custom_context(agent: Any | None) -> dict[str, Any] | None:
    """Safely extract custom_context. Returns copy. No fallback to initial (to force
    proper context passing and avoid stale data bugs).
    """
    if not agent or not hasattr(agent, "_context"):
        return None
    try:
        ctx = getattr(agent._context, "custom_context", None)
        if isinstance(ctx, dict):
            return dict(ctx)  # copy
        return ctx
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to extract custom_context from agent: %s", e)
    return None


def make_openai_client(api_key: str | None = None) -> AsyncOpenAI:
    """Создать AsyncOpenAI-клиент и, при включённом tracing, обернуть его для LangSmith."""
    api_key = api_key if api_key is not None else os.getenv("YANDEX_API_KEY", "")
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
    model: str | None = None,
    initial_custom_context: dict[str, Any] | None = None,
) -> ToolCallingAgent:
    """Создать экземпляр ToolCallingAgent на базе Yandex LLM. 
    initial_custom_context is set via private API
    """
    api_key = os.getenv("YANDEX_API_KEY", "")

    cfg = AgentConfig()
    cfg.llm.api_key = api_key
    cfg.llm.base_url = YANDEX_BASE_URL
    cfg.llm.model = get_model_uri(model)
    cfg.prompts.system_prompt_str = system_prompt
    cfg.execution.max_iterations = MAX_AGENT_ITERATIONS

    agent = ToolCallingAgent(
        task_messages=[{"role": "user", "content": task}],
        openai_client=make_openai_client(api_key=api_key),
        agent_config=cfg,
        toolkit=toolkit,
        def_name=name,
    )
    if initial_custom_context is not None:
        agent._context.custom_context = initial_custom_context
    return agent


def _unwrap(result: Any) -> str:
    """Извлекает ответ из результата агента.
    """
    if result is None:
        error_msg = (
            "Agent returned None (likely no tool_call selected or "
            "model failed to produce valid tool call response). "
        )
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    if isinstance(result, str):
        return result
    if hasattr(result, "execution_result"):
        return result.execution_result or str(result)
    if hasattr(result, "answer"):
        return result.answer
    return str(result)


async def run_agent(
    task: str,
    system_prompt: str,
    toolkit: list,
    *,
    name: str,
    model: str | None = None,
    initial_custom_context: dict[str, Any] | None = None,
) -> AgentRunResult:
    """Simple run_agent
    """
    agent = make_agent(
        task,
        system_prompt,
        toolkit,
        name=name,
        model=model,
        initial_custom_context=initial_custom_context,
    )
    try:
        result = await agent.execute()
        output = _unwrap(result)
        context = _safe_get_custom_context(agent)
        return AgentRunResult(output=output, context=context)
    except Exception as exc:
        logger.exception("Agent '%s' failed", name)
        # Return error as output for nodes to handle via fallback (simplification)
        error_output = json.dumps({"error": str(exc)}, ensure_ascii=False)
        context = _safe_get_custom_context(agent) or initial_custom_context
        return AgentRunResult(output=error_output, context=context)
