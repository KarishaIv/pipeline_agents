"""Worker execution helpers for meta-agent nodes.

Encapsulates worker execution with DTO extraction and structured parsing.
Worker configuration (definitions) is now in src.meta_agent.configs.workers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.meta_agent.agent_factory import run_agent
from src.meta_agent.configs.workers import WORKER_DEFINITIONS, WorkerDefinition
from src.meta_agent.tools.dto_tools import DTO_STORE_KEY
from src.meta_agent.utils.json_responses import json_node_failure
from src.meta_agent.utils.state import state_to_dict

logger = logging.getLogger("meta_agent")


def _get_worker_definition(worker_name: str) -> WorkerDefinition:
    """Return unified definition for a worker."""
    if worker_name not in WORKER_DEFINITIONS:
        raise ValueError(f"Unknown worker: {worker_name}. Must be one of {list(WORKER_DEFINITIONS.keys())}")
    return WORKER_DEFINITIONS[worker_name]


def _extract_dto_store(custom_context: dict | None) -> dict:
    """Извлекает и возвращает копию dto_store из custom_context агента run_agent.

    Гарантирует, что все DTO, зарегистрированные во время выполнения (включая ошибки),
    правильно попадают обратно в состояние графа. Всегда возвращает свежий словарь.
    """
    dto_store: dict = {}
    if isinstance(custom_context, dict):
        maybe_store = custom_context.get(DTO_STORE_KEY, {})
        if isinstance(maybe_store, dict):
            dto_store = dict(maybe_store)  # копия для безопасности
    return dto_store


async def _run_worker(
    state: dict | Any,
    definition: WorkerDefinition,
    task: str,
) -> Any:
    """Core helper: runs the inner ToolCallingAgent with timing, logging, and DTO context.
    """
    state_dict = state_to_dict(state)
    dto_store = state_dict.get("dto_store", {})

    # Use model_override if specified, else no explicit model (uses default from run_agent)
    effective_model = definition.model_override

    t0 = time.perf_counter()
    run_result = await run_agent(
        task=task,
        system_prompt=definition.system_prompt,
        toolkit=definition.tools,
        name=definition.worker_name,
        model=effective_model,
        initial_custom_context={DTO_STORE_KEY: dict(dto_store)},
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "%s завершён за %.1fс", definition.worker_name.replace("_", " ").title(), elapsed
    )
    return run_result


async def run_structured_worker(
    state: dict | Any,
    definition: WorkerDefinition,
    task: str,
) -> tuple[Any | None, dict]:
    """Execute worker with structured parsing and fallback handling.

    Encapsulates the common pattern across all nodes:
    1. Run the worker agent
    2. Parse output to definition.report_tool
    3. Handle parse failures with fallback
    4. Extract DTOs from context
    5. Format history entry

    Args:
        state: Current graph state
        definition: WorkerDefinition configuration for this worker
        task: Task/prompt to pass to the worker

    Returns:
        Tuple of:
        - parsed_report: The parsed Pydantic model instance, or None on parse failure
        - result_dict: State update dict containing:
          - history: Formatted history entry (list of dicts with role/content)
          - dto_store: Extracted DTOs from worker context
    """
    run_result = await _run_worker(
        state,
        definition,
        task,
    )
    output = run_result.output

    parsed = None
    fallback_content = None

    try:
        parsed = definition.report_tool.model_validate_json(output)
    except Exception as exc:
        logger.warning(
            "Не удалось распарсить %s для %s: %s",
            definition.report_tool.__name__,
            definition.worker_name,
            exc,
        )
        if definition.fallback_on_parse_error:
            fallback_content = json_node_failure(
                worker=definition.worker_name,
                raw_output=output,
                expected_tool=getattr(definition.report_tool, "tool_name", definition.report_tool.__name__),
                parse_error=exc,
            )

    if parsed is None:
        content = fallback_content or output
    elif definition.format_content:
        try:
            content = definition.format_content(parsed)
        except Exception as e:
            logger.warning(
                "Failed to format content for %s: %s",
                definition.worker_name,
                e,
            )
            content = output
    else:
        content = output

    dto_store = _extract_dto_store(run_result.context)

    return parsed, {
        "history": [{"role": definition.worker_name, "content": content}],
        "dto_store": dto_store,
    }

