"""Жизненный цикл агента: создание, выполнение и извлечение результата."""

import logging
import os

from sgr_agent_core import AgentConfig
from sgr_agent_core.agents.tool_calling_agent import ToolCallingAgent
from sgr_agent_core.base_tool import BaseTool

import src.meta_agent.lib_patches  # noqa: F401  apply all third-party patches

from config import get_model_uri, YANDEX_BASE_URL
from src.meta_agent.configs import LLM_TEMPERATURE, MAX_AGENT_ITERATIONS
from src.meta_agent.utils.json_responses import json_error
from src.utils import make_openai_client

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


def make_agent(
    task: str,
    system_prompt: str,
    toolkit: list,
    *,
    name: str = "agent",
    model: str | None = None,
    temperature: float | None = None,
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
    cfg.llm.temperature = temperature if temperature is not None else LLM_TEMPERATURE
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
        return json_error(error_msg, error_type="agent_error")

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
    temperature: float | None = None,
    initial_custom_context: dict[str, Any] | None = None,
) -> AgentRunResult:
    """Run an LLM agent with tools and return structured result.

    Creates agent via make_agent(), executes it, unwraps output,
    and captures custom context. On error, returns JSON error string
    as output (for node fallback handling) while preserving context.
    """
    agent = make_agent(
        task,
        system_prompt,
        toolkit,
        name=name,
        model=model,
        temperature=temperature,
        initial_custom_context=initial_custom_context,
    )
    try:
        result = await agent.execute()
        output = _unwrap(result)
        context = _safe_get_custom_context(agent)
        return AgentRunResult(output=output, context=context)
    except Exception as exc:
        logger.exception("Agent '%s' failed", name)

        # Return JSON error string as output (nodes handle via fallback logic)
        error_output = json_error(str(exc), error_type="agent_exception")
        context = _safe_get_custom_context(agent) or initial_custom_context

        return AgentRunResult(output=error_output, context=context)
