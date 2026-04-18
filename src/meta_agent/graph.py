"""Пайплайн супервайзера на LangGraph: объектная оркестрация и точки входа."""

import logging
import time
from typing import Annotated, Any, NamedTuple

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field
from src.meta_agent.nodes import analyzer_node, code_writer_node, data_extractor_node, supervisor_node
from src.meta_agent.utils import (
    append_history,
    build_persisted_history,
    build_turn_state_update,
    merge_dto_store,
    resolve_thread_id,
    route_analyzer,
    route_supervisor,
)

logger = logging.getLogger("meta_agent")

class MetaAgentResult(NamedTuple):
    answer: str
    thread_id: str


class MetaAgentState(BaseModel):
    """Pydantic model for graph state using LangGraph reducers for safe updates.

    Uses Annotated reducers so that partial dict updates from nodes are merged
    correctly (history appends instead of overwriting; dto_store merges).
    This is the idiomatic LangGraph built-in approach for Pydantic state.
    """
    question: str = Field(default="")
    # Reducers ensure history always appends and dto_store merges safely
    history: Annotated[list[dict[str, Any]], append_history] = Field(
        default_factory=list
    )  # [{"role": str, "content": str}]
    dto_store: Annotated[dict[str, dict[str, Any]], merge_dto_store] = Field(
        default_factory=dict
    )  # {dto_name: dto_payload}
    next_worker: str = Field(default="")
    current_task: str = Field(default="")
    delegated_attempts: int = Field(default=0)
    answer: str = Field(default="")
    iterations: int = Field(default=0)

    model_config = {"arbitrary_types_allowed": True}


class MetaAgentGraphManager:
    """Объект-оркестратор выполнения графа и управления сессиями."""

    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        self._graph = None

    def _build_graph(self):
        """Собрать и скомпилировать граф с checkpointer-ом в памяти.
        Uses Pydantic MetaAgentState with LangGraph reducers for history and dto_store.
        """
        graph = StateGraph(MetaAgentState)

        graph.add_node("supervisor", supervisor_node)
        graph.add_node("data_extractor", data_extractor_node)
        graph.add_node("analyzer", analyzer_node)
        graph.add_node("code_writer", code_writer_node)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {"data_extractor": "data_extractor", "analyzer": "analyzer", "end": END},
        )
        graph.add_conditional_edges(
            "analyzer",
            route_analyzer,
            {"code_writer": "code_writer", "supervisor": "supervisor"},
        )
        graph.add_edge("data_extractor", "supervisor")
        graph.add_edge("code_writer", "analyzer")

        return graph.compile(checkpointer=self._checkpointer)

    def get_graph(self):
        """Вернуть скомпилированный граф (ленивая инициализация)."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _resolve_session_thread_id(self, thread_id: str | None) -> str:
        """Разрешить thread_id для текущего запроса."""
        resolved = resolve_thread_id(thread_id)
        return resolved

    async def _prepare_invoke(self, question: str, thread_id: str) -> tuple[dict, dict]:
        """Сформировать config и state update до вызова графа.

        Uses LangGraph's Pydantic state support + reducers. We always validate
        the snapshot to a MetaAgentState instance for consistent .model_dump().
        Reducers (append_history, merge_dto_store) handle partial updates from nodes.
        """
        graph = self.get_graph()
        runnable_config: dict = {"configurable": {"thread_id": thread_id}}
        state_snapshot = await graph.aget_state(runnable_config)

        # LangGraph checkpoint returns dict; validate to model for type safety
        if isinstance(state_snapshot.values, dict):
            state_model = MetaAgentState.model_validate(state_snapshot.values)
        else:
            state_model = state_snapshot.values

        snapshot_values = state_model.model_dump()
        state_update = build_turn_state_update(question, snapshot_values)
        return runnable_config, state_update

    async def _finalize_invoke(self, runnable_config: dict, result: Any, question: str) -> str:
        """Сохранить обновлённую историю после вызова графа и вернуть ответ.

        Uses LangGraph aupdate_state. The history reducer will handle appending.
        """
        graph = self.get_graph()
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        answer = result_dict.get("answer", "")
        truncated_history = build_persisted_history(result_dict, question)
        # Only update history; reducers + other fields from previous state are preserved
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
