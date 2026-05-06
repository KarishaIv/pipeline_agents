"""Узлы графа мета-агента: супервайзер, извлечение данных и аналитика."""

import logging
from typing import Any
from langchain_core.runnables import RunnableConfig

from langsmith import traceable

from src.meta_agent.configs import MAX_DELEGATED_ATTEMPTS, MAX_HISTORY_CHARS, MAX_SUPERVISOR_ITERATIONS
from src.meta_agent.dto import DtoPayload
from src.meta_agent.tools.output_tools import AnalyzerDecisionTool
from src.meta_agent.utils.history import build_role_history_text_async, summarize_history_text
from src.meta_agent.utils.state import state_to_dict
from src.meta_agent.workers import _get_worker_definition, run_structured_worker

logger = logging.getLogger("meta_agent")


def _process_analyzer_decision(
    decision: AnalyzerDecisionTool,
    delegated_attempts: int,
    dto_store: dict[str, DtoPayload],
) -> dict:
    """Обрабатывает решение AnalyzerDecisionTool (report vs delegate с проверкой лимита).
    Выделяет сложную логику ветвления из analyzer_node.
    """
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
        return {
            "next_worker": "end",
            "outputs": [{"type": "text", "text": last}],  # Create text output for final answer
            "iterations": iterations + 1
        }

    history_text = await summarize_history_text(history)
    task = f"Вопрос пользователя: {state.get('question', '')}"
    if history_text:
        task += f"\n\nИстория работы:\n{history_text}"

    definition = _get_worker_definition("supervisor")
    parsed, result = await run_structured_worker(state, definition, task)

    if parsed is None:
        return {
            "next_worker": "end",
            "outputs": [{"type": "text", "text": result["history"][0]["content"]}],
            "history": result["history"],
            "dto_store": result["dto_store"],
            "artifacts": result.get("artifacts", []),
            "iterations": iterations + 1,
        }

    logger.info("Решение супервайзера: next=%s task=%s", parsed.next, (parsed.task or "")[:120])
    return {
        "next_worker": parsed.next,
        "current_task": parsed.task,
        "outputs": [{"type": "text", "text": parsed.final_answer}] if parsed.next == "end" else [],
        "history": result["history"],
        "dto_store": result["dto_store"],
        "artifacts": result.get("artifacts", []),
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

    definition = _get_worker_definition("data_extractor")
    parsed, result = await run_structured_worker(state, definition, task)

    return {
        "history": result["history"],
        "dto_store": result["dto_store"],
        "artifacts": result.get("artifacts", []),
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

    definition = _get_worker_definition("analyzer")
    parsed, result = await run_structured_worker(state, definition, task)

    if parsed is None:
        return {
            "next_worker": "supervisor",
            "history": result["history"],
            "dto_store": result["dto_store"],
            "artifacts": result.get("artifacts", []),
            "delegated_attempts": delegated_attempts,
        }

    decision_result = _process_analyzer_decision(
        parsed, delegated_attempts, result["dto_store"]
    )
    # Merge artifacts from the worker into the decision result
    decision_result["artifacts"] = result.get("artifacts", [])
    return decision_result

    if decision.decision == "report":
        findings_text = "\n".join(f"- {item}" for item in decision.key_findings)
        content = f"Ключевые выводы:\n{findings_text}\n\nЗаключения: {decision.conclusions}"
        return {
            "next_worker": "supervisor",
            "history": state.get("history", []) + [{"role": "analyzer", "content": content}],
            "dto_store": _extract_dto_store(run_result.context),
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
            "history": state.get("history", []) + [{"role": "analyzer", "content": limit_content}],
            "dto_store": _extract_dto_store(run_result.context),
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
        "history": state.get("history", []) + [{"role": "analyzer", "content": decision_content}],
        "dto_store": _extract_dto_store(run_result.context),
        "delegated_attempts": delegated_attempts + 1,
    }


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
            "artifacts": [],
        }

    prior_data = await _get_prior_worker_history(state)
    task = (
        f"Задача от analyzer: {code_task}\n\n"
        f"Исходный вопрос: {state.get('question', '')}\n\n"
        f"Контекст предыдущих шагов:\n{prior_data}"
    )

    definition = _get_worker_definition("code_writer")
    parsed, result = await run_structured_worker(state, definition, task)

    return {
        "next_worker": "analyzer",
        "history": result["history"],
        "dto_store": result["dto_store"],
        "artifacts": result.get("artifacts", []),
        "current_task": code_task,
        "delegated_attempts": int(state.get("delegated_attempts", 0)),
    }
