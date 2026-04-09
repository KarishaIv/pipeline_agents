"""LangGraph supervisor pipeline: graph state, node functions, and entry points."""

import logging
import time

from langsmith import traceable
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.meta_agent.agent_factory import TransientError, run_agent, truncate_history
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

logger = logging.getLogger("meta_agent")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class MetaAgentState(TypedDict):
    question: str
    history: list          # [{"role": str, "content": str}]
    next_worker: str       # routing decision
    current_task: str      # high-level task for the next worker
    answer: str            # set when next_worker == "end"
    iterations: int


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

@traceable(name="node.supervisor", run_type="chain")
async def supervisor_node(state: MetaAgentState) -> dict:
    """Supervisor agent: reviews history, assigns the next task, routes to
    a worker or terminates with a final answer."""
    iterations = state.get("iterations", 0)
    history: list = state.get("history", [])

    if iterations >= MAX_SUPERVISOR_ITERATIONS:
        logger.warning("Supervisor hit iteration limit (%d)", MAX_SUPERVISOR_ITERATIONS)
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
    logger.info("Supervisor iteration %d completed in %.1fs", iterations + 1, elapsed)

    if isinstance(raw, TransientError):
        logger.error("Supervisor hit transient error: %s", raw)
        return {
            "next_worker": "end",
            "answer": "Произошла временная ошибка при обращении к LLM. Попробуйте повторить запрос позже.",
            "history": history + [{"role": "supervisor", "content": f"[ВРЕМЕННАЯ ОШИБКА] {raw}"}],
            "iterations": iterations + 1,
        }

    try:
        decision = SupervisorDecisionTool.model_validate_json(raw)
    except Exception:
        logger.warning("Could not parse supervisor output as SupervisorDecisionTool, treating as final answer")
        return {
            "next_worker": "end",
            "answer": raw,
            "history": history + [{"role": "supervisor", "content": raw}],
            "iterations": iterations + 1,
        }

    logger.info("Supervisor decision: next=%s task=%s", decision.next, decision.task[:120])

    return {
        "next_worker": decision.next,
        "current_task": decision.task,
        "answer": decision.final_answer if decision.next == "end" else "",
        "history": history + [{"role": "supervisor", "content": raw}],
        "iterations": iterations + 1,
    }


@traceable(name="node.data_extractor", run_type="chain")
async def data_extractor_node(state: MetaAgentState) -> dict:
    """Data-extractor agent: autonomously decides which Qdrant tools and
    queries to use; reports back via DataExtractionReportTool."""
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
    logger.info("Data extractor completed in %.1fs", time.perf_counter() - t0)

    if isinstance(raw, TransientError):
        logger.error("Data extractor hit transient error: %s", raw)
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
async def analyzer_node(state: MetaAgentState) -> dict:
    """Analyzer agent: computes statistics, writes code, creates charts,
    and reports conclusions via AnalysisReportTool."""
    prior_data = "\n\n".join(
        f"[{m['role'].upper()}]: {m['content']}"
        for m in state.get("history", [])
        if m["role"] in ("data_extractor", "analyzer")
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
    logger.info("Analyzer completed in %.1fs", time.perf_counter() - t0)

    if isinstance(raw, TransientError):
        logger.error("Analyzer hit transient error: %s", raw)
        content = f"[ВРЕМЕННАЯ ОШИБКА] Анализ не выполнен: {raw.message}"
    else:
        try:
            report = AnalysisReportTool.model_validate_json(raw)
            findings_text = "\n".join(f"- {f}" for f in report.key_findings)
            content = f"Ключевые выводы:\n{findings_text}\n\nЗаключения: {report.conclusions}"
        except Exception:
            content = raw

    return {
        "history": state.get("history", []) + [{"role": "analyzer", "content": content}],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _route_supervisor(state: MetaAgentState) -> str:
    return state.get("next_worker", "end")


def build_graph():
    """
    Compile the supervisor-loop LangGraph pipeline.

    Topology::

        START → supervisor ─┬─→ data_extractor ─┐
                            ├─→ analyzer ───────┤
                            └─→ END             │
                   ┌────────────────────────────┘
                   └→ supervisor (loop)
    """
    graph = StateGraph(MetaAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_extractor", data_extractor_node)
    graph.add_node("analyzer", analyzer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {"data_extractor": "data_extractor", "analyzer": "analyzer", "end": END},
    )
    graph.add_edge("data_extractor", "supervisor")
    graph.add_edge("analyzer", "supervisor")

    return graph.compile()


_graph = None


@traceable(name="meta_agent.invoke_graph", run_type="chain")
async def invoke_graph(question: str) -> str:
    """Run the supervisor-loop LangGraph pipeline and return the final answer."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    logger.info("Starting graph for question: %s", question[:200])
    t0 = time.perf_counter()
    result = await _graph.ainvoke({"question": question, "history": [], "iterations": 0})
    logger.info("Graph completed in %.1fs", time.perf_counter() - t0)
    return result["answer"]
