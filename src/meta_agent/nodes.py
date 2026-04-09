"""Узлы графа мета-агента: супервайзер, извлечение данных и аналитика."""

import logging
import time

from langsmith import traceable

from src.meta_agent.agent_factory import TransientError, run_agent
from src.meta_agent.prompts import (
    ANALYZER_SYSTEM,
    EXTRACTOR_SYSTEM,
    MAX_SUPERVISOR_ITERATIONS,
    SUPERVISOR_SYSTEM,
)
from src.meta_agent.tools import (
    AnalysisReportTool,
    ComputeStatsTool,
    CreateChartTool,
    DataExtractionReportTool,
    ExecuteCodeTool,
    QdrantCollectionSchema,
    QdrantFilterTool,
    QdrantRetrieveTool,
    QdrantScrollTool,
    QdrantSearchTool,
    SummarizeTextsTool,
    SupervisorDecisionTool,
)
from src.meta_agent.utils import truncate_history

logger = logging.getLogger("meta_agent")


@traceable(name="node.supervisor", run_type="chain")
async def supervisor_node(state: dict) -> dict:
    """Узел супервайзера: анализирует историю, ставит следующую задачу,
    маршрутизирует к воркеру или завершает с итоговым ответом."""
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
    raw = await run_agent(
        task=task,
        system_prompt=SUPERVISOR_SYSTEM,
        toolkit=[SupervisorDecisionTool],
        name="supervisor",
    )
    elapsed = time.perf_counter() - t0
    logger.info("Итерация супервайзера %d завершена за %.1fс", iterations + 1, elapsed)

    if isinstance(raw, TransientError):
        logger.error("У супервайзера временная ошибка: %s", raw)
        return {
            "next_worker": "end",
            "answer": "Произошла временная ошибка при обращении к LLM. Попробуйте повторить запрос позже.",
            "history": history + [{"role": "supervisor", "content": f"[ВРЕМЕННАЯ ОШИБКА] {raw}"}],
            "iterations": iterations + 1,
        }

    try:
        decision = SupervisorDecisionTool.model_validate_json(raw)
    except Exception:
        logger.warning("Не удалось распарсить ответ супервайзера как SupervisorDecisionTool; считаем его финальным ответом")
        return {
            "next_worker": "end",
            "answer": raw,
            "history": history + [{"role": "supervisor", "content": raw}],
            "iterations": iterations + 1,
        }

    logger.info("Решение супервайзера: next=%s task=%s", decision.next, decision.task[:120])
    return {
        "next_worker": decision.next,
        "current_task": decision.task,
        "answer": decision.final_answer if decision.next == "end" else "",
        "history": history + [{"role": "supervisor", "content": raw}],
        "iterations": iterations + 1,
    }


@traceable(name="node.data_extractor", run_type="chain")
async def data_extractor_node(state: dict) -> dict:
    """Узел извлечения данных: самостоятельно выбирает Qdrant-инструменты
    и запросы, затем отчитывается через DataExtractionReportTool."""
    task = (
        f"Задача от супервайзера: {state['current_task']}\n\n"
        f"Контекст — исходный вопрос пользователя: {state['question']}"
    )

    t0 = time.perf_counter()
    raw = await run_agent(
        task=task,
        system_prompt=EXTRACTOR_SYSTEM,
        toolkit=[
            QdrantCollectionSchema,
            QdrantSearchTool,
            QdrantFilterTool,
            QdrantScrollTool,
            QdrantRetrieveTool,
            DataExtractionReportTool,
        ],
        name="data_extractor",
    )
    logger.info("Извлечение данных завершено за %.1fс", time.perf_counter() - t0)

    if isinstance(raw, TransientError):
        logger.error("У агента извлечения данных временная ошибка: %s", raw)
        content = f"[ВРЕМЕННАЯ ОШИБКА] Не удалось получить данные: {raw.message}"
    else:
        try:
            report = DataExtractionReportTool.model_validate_json(raw)
            content = f"Кратко: {report.summary}\n\nДанные: {report.raw_data}"
        except Exception:
            content = raw

    return {
        "history": state.get("history", []) + [{"role": "data_extractor", "content": content}],
    }


@traceable(name="node.analyzer", run_type="chain")
async def analyzer_node(state: dict) -> dict:
    """Узел аналитики: считает статистику, пишет код, строит графики
    и передаёт выводы через AnalysisReportTool."""
    prior_data = "\n\n".join(
        f"[{message['role'].upper()}]: {message['content']}"
        for message in state.get("history", [])
        if message["role"] in ("data_extractor", "analyzer")
    )
    task = (
        f"Задача от супервайзера: {state['current_task']}\n\n"
        f"Исходный вопрос: {state['question']}\n\n"
        f"Собранные данные:\n{prior_data}"
    )

    t0 = time.perf_counter()
    raw = await run_agent(
        task=task,
        system_prompt=ANALYZER_SYSTEM,
        toolkit=[
            SummarizeTextsTool,
            ComputeStatsTool,
            ExecuteCodeTool,
            CreateChartTool,
            AnalysisReportTool,
        ],
        name="analyzer",
    )
    logger.info("Аналитика завершена за %.1fс", time.perf_counter() - t0)

    if isinstance(raw, TransientError):
        logger.error("У агента аналитики временная ошибка: %s", raw)
        content = f"[ВРЕМЕННАЯ ОШИБКА] Анализ не выполнен: {raw.message}"
    else:
        try:
            report = AnalysisReportTool.model_validate_json(raw)
            findings_text = "\n".join(f"- {item}" for item in report.key_findings)
            content = f"Ключевые выводы:\n{findings_text}\n\nЗаключения: {report.conclusions}"
        except Exception:
            content = raw

    return {
        "history": state.get("history", []) + [{"role": "analyzer", "content": content}],
    }
