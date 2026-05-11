"""Узлы графа мета-агента: супервайзер, извлечение данных и аналитика."""

import logging
from typing import Any
from langchain_core.runnables import RunnableConfig

from langsmith import traceable

from src.meta_agent.configs import MAX_DELEGATED_ATTEMPTS, MAX_HISTORY_CHARS, MAX_SUPERVISOR_ITERATIONS
from src.meta_agent.prompts import OOD_CHECKER_SYSTEM
from src.meta_agent.dto import DtoPayload
from src.meta_agent.tools.output_tools import AnalyzerDecisionTool, OODCheckResult
from src.meta_agent.utils.history import build_role_history_text_async, summarize_history_text
from src.meta_agent.utils.state import state_to_dict
from src.utils import robust_llm_call
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
    prior_data = await _get_prior_worker_history(state)
    task = (
        f"Задача от супервайзера: {state.get('current_task', '')}\n\n"
        f"Исходный вопрос: {state.get('question', '')}"
    )
    if prior_data:
        task += f"\n\nИстория предыдущих шагов:\n{prior_data}"

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


FORCE_GUIDANCE = (
    "\n\nВы можете добавить префикс /force к вашему вопросу, "
    "чтобы пропустить эту проверку и продолжить работу с мета-агентом."
)


@traceable(name="node.ood_checker", run_type="chain")
async def ood_checker_node(state: dict | Any, config: RunnableConfig | None = None) -> dict:
    """Первый узел графа: проверяет релевантность вопроса к симуляционному пайплайну.

    Делает один structured LLM вызов через robust_llm_call.
    Если релевантно — передаёт супервайзеру.
    Если нет — завершает с redirect-сообщением (к которому дописывается guidance про /force).
    """
    state = state_to_dict(state)
    question = state.get("question", "").strip()
    if not question:
        return {"next_worker": "end", "outputs": [{"type": "text", "text": "Пожалуйста, задайте вопрос."}]}

    history = state.get("history", [])
    previous_steps = "Нет предыдущих шагов."
    if history:
        try:
            previous_steps = await summarize_history_text(history, max_chars=MAX_HISTORY_CHARS)
        except Exception:
            # fallback to last few messages
            previous_steps = "\n\n".join(
                f"[{m.get('role','?').upper()}]: {str(m.get('content',''))[:200]}"
                for m in history[-3:]
            )

    prompt = OOD_CHECKER_SYSTEM.format(question=question, previous_steps=previous_steps)
    result = await robust_llm_call(prompt, structured_output=OODCheckResult)

    if isinstance(result, dict):
        # fallback if not validated
        is_relevant = result.get("is_relevant", False)
        msg = result.get("redirect_message")
    else:
        is_relevant = getattr(result, "is_relevant", False)
        msg = getattr(result, "redirect_message", None)

    if is_relevant:
        return {
            "next_worker": "supervisor",
            "history": [{"role": "ood_checker", "content": "Вопрос релевантен симуляциям. Продолжаю."}],
        }

    # not relevant: append fixed /force guidance (post-LLM, never in prompt)
    final_msg = (msg or "Ваш вопрос не относится к симуляционному пайплайну.") + FORCE_GUIDANCE
    return {
        "next_worker": "end",
        "outputs": [{"type": "text", "text": final_msg}],
        "history": [{"role": "ood_checker", "content": final_msg}],
    }
