"""Пайплайн супервайзера на LangGraph: объектная оркестрация и точки входа."""

import logging
import time
from typing import NamedTuple

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from src.meta_agent.nodes import analyzer_node, data_extractor_node, supervisor_node
from src.meta_agent.utils import build_persisted_history, build_turn_state_update, resolve_thread_id, route_supervisor

logger = logging.getLogger("meta_agent")

class MetaAgentResult(NamedTuple):
    answer: str
    thread_id: str


class MetaAgentState(TypedDict):
    question: str
    history: list          # [{"role": str, "content": str}]
    dto_store: dict        # {dto_name: dto_payload}
    next_worker: str       # решение маршрутизации
    current_task: str      # задача верхнего уровня для следующего воркера
    answer: str            # заполняется при next_worker == "end"
    iterations: int


class MetaAgentGraphManager:
    """Объект-оркестратор выполнения графа и управления сессиями."""

    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        self._graph = None
        self._last_thread_id: str | None = None

    def _build_graph(self):
        """Собрать и скомпилировать граф с checkpointer-ом в памяти."""
        graph = StateGraph(MetaAgentState)

        graph.add_node("supervisor", supervisor_node)
        graph.add_node("data_extractor", data_extractor_node)
        graph.add_node("analyzer", analyzer_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {"data_extractor": "data_extractor", "analyzer": "analyzer", "end": END},
        )
        graph.add_edge("data_extractor", "supervisor")
        graph.add_edge("analyzer", "supervisor")

        return graph.compile(checkpointer=self._checkpointer)

    def get_graph(self):
        """Вернуть скомпилированный граф (ленивая инициализация)."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _resolve_session_thread_id(self, thread_id: str | None) -> str:
        """Разрешить thread_id для текущего запроса и запомнить его."""
        resolved = resolve_thread_id(thread_id, self._last_thread_id)
        self._last_thread_id = resolved
        return resolved

    async def _prepare_invoke(self, question: str, thread_id: str) -> tuple[dict, dict]:
        """Сформировать config и state update до вызова графа."""
        graph = self.get_graph()
        runnable_config: dict = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await graph.aget_state(runnable_config)
        state_update = build_turn_state_update(question, state_snapshot.values)
        return runnable_config, state_update

    async def _finalize_invoke(self, runnable_config: dict, result: dict, question: str) -> str:
        """Сохранить обновлённую историю после вызова графа и вернуть ответ."""
        graph = self.get_graph()
        answer = result.get("answer", "")
        truncated_history = build_persisted_history(result, question)
        await graph.aupdate_state(runnable_config, {"history": truncated_history})
        return answer

    @traceable(name="meta_agent.invoke_graph_session", run_type="chain")
    async def invoke_graph_session(self, question: str, thread_id: str | None = None) -> MetaAgentResult:
        """Запустить граф в персистентной сессии, заданной thread_id."""
        resolved_thread_id = self._resolve_session_thread_id(thread_id)
        runnable_config, state_update = await self._prepare_invoke(question, resolved_thread_id)

        logger.info("Сессия %s — вопрос: %s", resolved_thread_id, question[:200])
        t0 = time.perf_counter()

        result = await self.get_graph().ainvoke(state_update, runnable_config)
        answer = await self._finalize_invoke(runnable_config, result, question)

        elapsed = time.perf_counter() - t0
        logger.info("Граф завершён за %.1fс", elapsed)
        return MetaAgentResult(answer=answer, thread_id=resolved_thread_id)

    @traceable(name="meta_agent.invoke_graph", run_type="chain")
    async def invoke_graph(self, question: str) -> str:
        """Запуск без сохранения состояния: для каждого вызова создаётся новая сессия и возвращается только ответ."""
        out = await self.invoke_graph_session(question, "-1")
        return out.answer


meta_graph_manager = MetaAgentGraphManager()
