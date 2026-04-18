"""Узлы графа мета-агента: супервайзер, извлечение данных и аналитика."""

import json
import logging
import time
from typing import Any

from langsmith import traceable

from config import BIG_MODEL
from src.meta_agent.agent_factory import run_agent
from src.meta_agent.config import MAX_DELEGATED_ATTEMPTS, MAX_SUPERVISOR_ITERATIONS
from src.meta_agent.prompts import (
    ANALYZER_SYSTEM,
    CODE_WRITER_SYSTEM,
    EXTRACTOR_SYSTEM,
    SUPERVISOR_SYSTEM,
)
from src.meta_agent.tools.dto_tools import DTO_STORE_KEY
from src.meta_agent.tools import (
    AnalyzerDecisionTool,
    CodeExecutionReportTool,
    ComputeStatsTool,
    CreateChartTool,
    DataExtractionReportTool,
    ExecuteCodeTool,
    ListDtoNamesTool,
    QdrantCollectionSchema,
    QdrantFilterTool,
    QdrantRetrieveTool,
    QdrantScrollTool,
    QdrantSearchTool,
    RemainingStepsTool,
    SampleDtoTool,
    SummarizeTextsTool,
    SupervisorDecisionTool,
    ValidateCodeTool,
)
from src.meta_agent.utils import state_to_dict, truncate_history

logger = logging.getLogger("meta_agent")


def _fallback_worker_payload(
    *,
    worker: str,
    raw_output: str,
    expected_tool: str,
    parse_error: Exception | None = None,
) -> str:
    """Сформировать единообразный fallback-пейлоад для истории графа."""
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
    """Extract and return a copy of dto_store from run_agent custom_context.

    Ensures DTOs registered during agent execution (including on error paths)
    are properly merged back into graph state. Always returns fresh dict to
    prevent cross-node mutation.
    """
    dto_store: dict = {}
    if isinstance(custom_context, dict):
        maybe_store = custom_context.get(DTO_STORE_KEY, {})
        if isinstance(maybe_store, dict):
            dto_store = dict(maybe_store)  # copy for safety
    return dto_store


@traceable(name="node.supervisor", run_type="chain")
async def supervisor_node(state: dict | Any) -> dict:
    """Узел супервайзера: анализирует историю, ставит следующую задачу,
    маршрутизирует к воркеру или завершает с итоговым ответом."""
    state = state_to_dict(state)  # support Pydantic state from LangGraph reducers
    iterations = state.get("iterations", 0)
    history: list = state.get("history", [])

    if iterations >= MAX_SUPERVISOR_ITERATIONS:
        logger.warning("Супервайзер достиг лимита итераций (%d)", MAX_SUPERVISOR_ITERATIONS)
        last = history[-1]["content"] if history else "Недостаточно данных для ответа."
        return {"next_worker": "end", "answer": last, "iterations": iterations + 1}

    history_text = truncate_history(history)
    task = f"Вопрос пользователя: {state['question']}"
    if history_text:
        task += f"\n\nИстория работы:\n{history_text}"

    t0 = time.perf_counter()
    run_result = await run_agent(
        task=task,
        system_prompt=SUPERVISOR_SYSTEM,
        toolkit=[RemainingStepsTool, SupervisorDecisionTool],
        name="supervisor",
    )
    output = run_result.output
    elapsed = time.perf_counter() - t0
    logger.info("Итерация супервайзера %d завершена за %.1fс", iterations + 1, elapsed)

    try:
        decision = SupervisorDecisionTool.model_validate_json(output)
    except Exception:
        logger.warning("Не удалось распарсить ответ супервайзера как SupervisorDecisionTool; считаем его финальным ответом")
        return {
            "next_worker": "end",
            "answer": output,
            "history": history + [{"role": "supervisor", "content": output}],
            "iterations": iterations + 1,
        }

    logger.info("Решение супервайзера: next=%s task=%s", decision.next, decision.task[:120])
    return {
        "next_worker": decision.next,
        "current_task": decision.task,
        "answer": decision.final_answer if decision.next == "end" else "",
        "history": history + [{"role": "supervisor", "content": output}],
        "iterations": iterations + 1,
    }


@traceable(name="node.data_extractor", run_type="chain")
async def data_extractor_node(state: dict | Any) -> dict:
    """Узел извлечения данных: самостоятельно выбирает Qdrant-инструменты
    и запросы, затем отчитывается через DataExtractionReportTool."""
    state = state_to_dict(state)  # support Pydantic state from LangGraph reducers
    task = (
        f"Задача от супервайзера: {state['current_task']}\n\n"
        f"Контекст — исходный вопрос пользователя: {state['question']}"
    )

    t0 = time.perf_counter()
    run_result = await run_agent(
        task=task,
        system_prompt=EXTRACTOR_SYSTEM,
        toolkit=[
            QdrantCollectionSchema,
            QdrantSearchTool,
            QdrantFilterTool,
            QdrantScrollTool,
            QdrantRetrieveTool,
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            DataExtractionReportTool,
        ],
        name="data_extractor",
        initial_custom_context={DTO_STORE_KEY: state.get("dto_store", {})},
    )
    output = run_result.output
    logger.info("Извлечение данных завершено за %.1fс", time.perf_counter() - t0)

    try:
        report = DataExtractionReportTool.model_validate_json(output)
        content = f"Кратко: {report.summary}\n\nДанные: {report.dto_references}"
    except Exception as exc:
        logger.warning("Fallback data_extractor: не удалось распарсить DataExtractionReportTool: %s", exc)
        content = _fallback_worker_payload(
            worker="data_extractor",
            raw_output=output,
            expected_tool=DataExtractionReportTool.tool_name,
            parse_error=exc,
        )

    return {
        "history": state.get("history", []) + [{"role": "data_extractor", "content": content}],
        "dto_store": _extract_dto_store(run_result.context),
    }


@traceable(name="node.analyzer", run_type="chain")
async def analyzer_node(state: dict | Any) -> dict:
    """Узел аналитики: использует unified AnalyzerDecisionTool для выбора между
    report (выводы) и delegate (code_writer)
    """
    state = state_to_dict(state)  # support Pydantic state from LangGraph reducers
    prior_data = "\n\n".join(
        f"[{message['role'].upper()}]: {message['content']}"
        for message in state.get("history", [])
        if message["role"] in ("data_extractor", "analyzer", "code_writer")
    )
    task = (
        f"Задача от супервайзера: {state['current_task']}\n\n"
        f"Исходный вопрос: {state['question']}\n\n"
        f"Собранные данные:\n{prior_data}"
    )
    delegated_attempts = int(state.get("delegated_attempts", 0))

    t0 = time.perf_counter()
    run_result = await run_agent(
        task=task,
        system_prompt=ANALYZER_SYSTEM,
        toolkit=[
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            SummarizeTextsTool,
            ComputeStatsTool,
            CreateChartTool,
            AnalyzerDecisionTool,
        ],
        name="analyzer",
        initial_custom_context={DTO_STORE_KEY: state.get("dto_store", {})},
    )
    output = run_result.output
    logger.info("Аналитика завершена за %.1fс", time.perf_counter() - t0)

    try:
        decision = AnalyzerDecisionTool.model_validate_json(output)
    except Exception as exc:
        logger.warning("Не удалось распарсить AnalyzerDecisionTool: %s", exc)
        content = _fallback_worker_payload(
            worker="analyzer",
            raw_output=output,
            expected_tool=AnalyzerDecisionTool.tool_name,
            parse_error=exc,
        )
        return {
            "next_worker": "supervisor",
            "history": state.get("history", []) + [{"role": "analyzer", "content": content}],
            "dto_store": _extract_dto_store(run_result.context),
            "delegated_attempts": delegated_attempts,
        }

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
async def code_writer_node(state: dict | Any) -> dict:
    """Узел code_writer: пишет, валидирует и запускает код с BIG_MODEL."""
    state = state_to_dict(state)  # support Pydantic state from LangGraph reducers
    code_task = state.get("current_task", "").strip()
    if not code_task:
        content = (
            "[ОШИБКА ROUTING] analyzer направил в code_writer без задачи. "
            "Возврат к analyzer."
        )
        return {
            "next_worker": "analyzer",
            "history": state.get("history", []) + [{"role": "code_writer", "content": content}],
            "dto_store": state.get("dto_store", {}),
        }

    prior_data = "\n\n".join(
        f"[{message['role'].upper()}]: {message['content']}"
        for message in state.get("history", [])
        if message["role"] in ("data_extractor", "analyzer", "code_writer")
    )
    task = (
        f"Задача от analyzer: {code_task}\n\n"
        f"Исходный вопрос: {state['question']}\n\n"
        f"Контекст предыдущих шагов:\n{prior_data}"
    )

    t0 = time.perf_counter()
    run_result = await run_agent(
        task=task,
        system_prompt=CODE_WRITER_SYSTEM,
        toolkit=[
            ListDtoNamesTool,
            RemainingStepsTool,
            SampleDtoTool,
            ValidateCodeTool,
            ExecuteCodeTool,
            CodeExecutionReportTool,
        ],
        name="code_writer",
        model=BIG_MODEL,
        initial_custom_context={DTO_STORE_KEY: state.get("dto_store", {})},
    )
    output = run_result.output
    logger.info("Code_writer завершён за %.1fс", time.perf_counter() - t0)

    try:
        report = CodeExecutionReportTool.model_validate_json(output)
        findings_text = "\n".join(f"- {item}" for item in report.findings)
        content = (
            f"Задача: {report.task}\n"
            f"Найдено:\n{findings_text}\n\n"
            f"Валидация: {report.validation}\n"
            f"Выполнение: {report.execution}"
        )
    except Exception as exc:
        logger.warning("Fallback code_writer: не удалось распарсить CodeExecutionReportTool: %s", exc)
        content = _fallback_worker_payload(
            worker="code_writer",
            raw_output=output,
            expected_tool=CodeExecutionReportTool.tool_name,
            parse_error=exc,
        )

    return {
        "next_worker": "analyzer",
        "history": state.get("history", []) + [{"role": "code_writer", "content": content}],
        "dto_store": _extract_dto_store(run_result.context),
        "current_task": code_task,
        "delegated_attempts": int(state.get("delegated_attempts", 0)),
    }
