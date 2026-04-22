"""Узлы графа мета-агента: супервайзер, извлечение данных и аналитика."""

import json
import logging
import time
from typing import Any
from langchain_core.runnables import RunnableConfig

from langsmith import traceable

from src.meta_agent.agent_factory import run_agent
from src.meta_agent.config import BIG_MODEL, MAX_DELEGATED_ATTEMPTS, MAX_HISTORY_CHARS, MAX_SUPERVISOR_ITERATIONS
from src.meta_agent.utils.history import build_role_history_text_async, summarize_history_text
from src.meta_agent.utils.state import state_to_dict
from src.meta_agent.workers import _get_worker_config
from src.meta_agent.tools.dto_tools import DTO_STORE_KEY
from src.meta_agent.tools import (
    AnalyzerDecisionTool,
    CodeExecutionReportTool,
    DataExtractionReportTool,
    SupervisorDecisionTool,
)

logger = logging.getLogger("meta_agent")


def _fallback_worker_payload(
    *,
    worker: str,
    raw_output: str,
    expected_tool: str,
    parse_error: Exception | None = None,
) -> str:
    """Формирует единообразный JSON-payload ошибки для истории графа при неудачном парсинге отчёта."""
    payload = {
        "status": "failed",
        "worker": worker,
        "error_type": "report_parse_error" if parse_error else "unexpected_output",
        "expected_report_tool": expected_tool,
        "message": (
            f"Не удалось распарсить отчёт инструмента {expected_tool}"
            if parse_error
            else "Воркер вернул неожиданный формат ответа"
        ),
        "details": str(parse_error) if parse_error else "",
        "raw_output": raw_output,
    }
    return json.dumps(payload, ensure_ascii=False)


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


def _process_analyzer_decision(
    decision: Any,
    delegated_attempts: int,
    run_result_context: Any | None,
) -> dict:
    """Обрабатывает решение AnalyzerDecisionTool (report vs delegate с проверкой лимита).
    Выделяет сложную логику ветвления из analyzer_node.
    """
    dto_store = _extract_dto_store(run_result_context)

    if decision.decision == "report":
        findings_text = "\n".join(f"- {item}" for item in decision.key_findings)
        content = f"Ключевые выводы:\n{findings_text}\n\nЗаключения: {decision.conclusions}"
        return {
            "next_worker": "supervisor",
            "history": [{"role": "analyzer", "content": content}],
            "dto_store": dto_store,
            "delegated_attempts": delegated_attempts,
        }

    # delegate to code_writer
    if delegated_attempts >= MAX_DELEGATED_ATTEMPTS:
        limit_content = (
            f"[ЛИМИТ ДЕЛЕГИРОВАНИЯ] Достигнут лимит переходов analyzer -> code_writer: "
            f"{MAX_DELEGATED_ATTEMPTS}. Продолжаю без новых делегаций."
        )
        return {
            "next_worker": "supervisor",
            "history": [{"role": "analyzer", "content": limit_content}],
            "dto_store": dto_store,
            "delegated_attempts": delegated_attempts,
        }

    decision_content = (
        "Решение аналитика: code_writer.\n"
        f"Причина: {decision.delegate_reason or decision.reasoning}\n"
        f"Задача: {decision.task}"
    )
    return {
        "next_worker": "code_writer",
        "current_task": decision.task,
        "history": [{"role": "analyzer", "content": decision_content}],
        "dto_store": dto_store,
        "delegated_attempts": delegated_attempts + 1,
    }



async def _get_prior_worker_history(state: dict | Any) -> str:
    """Shared history builder for analyzer/code_writer. Uses role filter + summarization."""
    state_dict = state_to_dict(state)
    return await build_role_history_text_async(
        state_dict.get("history", []),
        roles=("data_extractor", "analyzer", "code_writer"),
        max_chars=MAX_HISTORY_CHARS,
    )


async def _run_worker(
    state: dict | Any,
    worker_name: str,
    task: str,
    model: str | None = None,
) -> Any:
    """Core helper: runs the inner ToolCallingAgent with timing, logging, and DTO context.
    """
    state_dict = state_to_dict(state)
    config = _get_worker_config(worker_name)
    dto_store = state_dict.get("dto_store", {})

    # Use config default_model unless explicitly overridden (for code_writer)
    effective_model = model or config.default_model

    t0 = time.perf_counter()
    run_result = await run_agent(
        task=task,
        system_prompt=config.system_prompt,
        toolkit=config.tools,
        name=worker_name,
        model=effective_model,
        initial_custom_context={DTO_STORE_KEY: dict(dto_store)},
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "%s завершён за %.1fс", worker_name.replace("_", " ").title(), elapsed
    )
    return run_result


def _safe_parse_output(
    output: str,
    tool_model: type,
    worker: str,
) -> tuple[Any | None, str | None]:
    """Parse output to Pydantic model or generate fallback JSON payload for history.
    """
    try:
        parsed = tool_model.model_validate_json(output)
        return parsed, None
    except Exception as exc:
        logger.warning(
            "Не удалось распарсить %s для %s: %s", tool_model.__name__, worker, exc
        )
        content = _fallback_worker_payload(
            worker=worker,
            raw_output=output,
            expected_tool=getattr(tool_model, "tool_name", tool_model.__name__),
            parse_error=exc,
        )
        return None, content


# === Nodes ===


@traceable(name="node.supervisor", run_type="chain")
async def supervisor_node(state: dict | Any, config: RunnableConfig | None = None) -> dict:
    """Узел супервайзера: анализирует историю, ставит следующую задачу,
    маршрутизирует к воркеру или завершает с итоговым ответом.
    """
    state = state_to_dict(state)
    iterations = state.get("iterations", 0)
    history: list = state.get("history", [])

    if iterations >= MAX_SUPERVISOR_ITERATIONS:
        logger.warning("Супервайзер достиг лимита итераций (%d)", MAX_SUPERVISOR_ITERATIONS)
        last = history[-1]["content"] if history else "Недостаточно данных для ответа."
        return {"next_worker": "end", "answer": last, "iterations": iterations + 1}

    history_text = await summarize_history_text(history)
    task = f"Вопрос пользователя: {state.get('question', '')}"
    if history_text:
        task += f"\n\nИстория работы:\n{history_text}"

    run_result = await _run_worker(state, "supervisor", task)
    output = run_result.output

    try:
        decision = SupervisorDecisionTool.model_validate_json(output)
    except Exception:
        logger.warning("Не удалось распарсить ответ супервайзера как SupervisorDecisionTool; считаем его финальным ответом")
        return {
            "next_worker": "end",
            "answer": output,
            "history": [{"role": "supervisor", "content": output}],
            "iterations": iterations + 1,
        }

    logger.info("Решение супервайзера: next=%s task=%s", decision.next, (decision.task or "")[:120])
    return {
        "next_worker": decision.next,
        "current_task": decision.task,
        "answer": decision.final_answer if decision.next == "end" else "",
        "history": [{"role": "supervisor", "content": output}],
        "iterations": iterations + 1,
    }


@traceable(name="node.data_extractor", run_type="chain")
async def data_extractor_node(state: dict | Any, config: RunnableConfig | None = None) -> dict:
    """Узел извлечения данных: самостоятельно выбирает Qdrant-инструменты
    и запросы, затем отчитывается через DataExtractionReportTool.
    """
    state = state_to_dict(state)
    task = (
        f"Задача от супервайзера: {state.get('current_task', '')}\n\n"
        f"Контекст — исходный вопрос пользователя: {state.get('question', '')}"
    )

    run_result = await _run_worker(state, "data_extractor", task)
    output = run_result.output

    parsed, fallback_content = _safe_parse_output(
        output, DataExtractionReportTool, "data_extractor"
    )
    if parsed is None:
        content = fallback_content or output
    else:
        content = f"Кратко: {parsed.summary}\n\nДанные: {parsed.dto_references}"

    return {
        "history": [{"role": "data_extractor", "content": content}],
        "dto_store": _extract_dto_store(run_result.context),
    }


@traceable(name="node.analyzer", run_type="chain")
async def analyzer_node(state: dict | Any, config: RunnableConfig | None = None) -> dict:
    """Узел аналитики: использует unified AnalyzerDecisionTool для выбора между
    report (выводы) и delegate (code_writer). Сложная логика ветвления
    вынесена в _process_analyzer_decision.
    """
    state = state_to_dict(state)
    prior_data = await _get_prior_worker_history(state)
    task = (
        f"Задача от супервайзера: {state.get('current_task', '')}\n\n"
        f"Исходный вопрос: {state.get('question', '')}\n\n"
        f"Собранные данные:\n{prior_data}"
    )
    delegated_attempts = int(state.get("delegated_attempts", 0))

    run_result = await _run_worker(state, "analyzer", task)
    output = run_result.output

    parsed, fallback_content = _safe_parse_output(
        output, AnalyzerDecisionTool, "analyzer"
    )
    if parsed is None:
        content = fallback_content or output
        return {
            "next_worker": "supervisor",
            "history": [{"role": "analyzer", "content": content}],
            "dto_store": _extract_dto_store(run_result.context),
            "delegated_attempts": delegated_attempts,
        }

    return _process_analyzer_decision(
        parsed, delegated_attempts, run_result.context
    )


@traceable(name="node.code_writer", run_type="chain")
async def code_writer_node(state: dict | Any, config: RunnableConfig | None = None) -> dict:
    """Узел code_writer: пишет, валидирует и запускает код с BIG_MODEL."""
    state = state_to_dict(state)
    code_task = state.get("current_task", "").strip()
    if not code_task:
        content = (
            "[ОШИБКА ROUTING] analyzer направил в code_writer без задачи. "
            "Возврат к analyzer."
        )
        return {
            "next_worker": "analyzer",
            "history": [{"role": "code_writer", "content": content}],
            "dto_store": state.get("dto_store", {}),
        }

    prior_data = await _get_prior_worker_history(state)
    task = (
        f"Задача от analyzer: {code_task}\n\n"
        f"Исходный вопрос: {state.get('question', '')}\n\n"
        f"Контекст предыдущих шагов:\n{prior_data}"
    )

    run_result = await _run_worker(state, "code_writer", task, model=BIG_MODEL)
    output = run_result.output

    parsed, fallback_content = _safe_parse_output(
        output, CodeExecutionReportTool, "code_writer"
    )
    if parsed is None:
        content = fallback_content or output
    else:
        findings_text = "\n".join(f"- {item}" for item in parsed.findings)
        content = (
            f"Задача: {parsed.task}\n"
            f"Найдено:\n{findings_text}\n\n"
            f"Валидация: {parsed.validation}\n"
            f"Выполнение: {parsed.execution}"
        )

    return {
        "next_worker": "analyzer",
        "history": [{"role": "code_writer", "content": content}],
        "dto_store": _extract_dto_store(run_result.context),
        "current_task": code_task,
        "delegated_attempts": int(state.get("delegated_attempts", 0)),
    }
