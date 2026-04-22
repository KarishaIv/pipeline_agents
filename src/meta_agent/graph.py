"""Оркестратор мета-агента на LangGraph.

Содержит MetaAgentGraphManager, MetaAgentState, сессионное управление,
подготовку invoke и finalize
"""

import logging
import time
import uuid
import asyncio
from pathlib import Path
from typing import Any, NamedTuple

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from src.meta_agent.config import CHECKPOINT_DB_PATH
from src.meta_agent.nodes import analyzer_node, code_writer_node, data_extractor_node, supervisor_node
from src.meta_agent.utils.state import MetaAgentState, build_turn_state_update
from src.meta_agent.utils.history import (
    build_persisted_history,
)
from src.meta_agent.utils.routing import route_analyzer, route_supervisor

logger = logging.getLogger("meta_agent")


class MetaAgentResult(NamedTuple):
    """Результат выполнения мета-агента."""
    answer: str
    thread_id: str


class MetaAgentGraphManager:
    """Объект-оркестратор выполнения графа и управления сессиями."""

    def __init__(self, checkpoint_db_path: Path | None = None) -> None:
        self._checkpointer_cm = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._checkpoint_db_path = checkpoint_db_path or CHECKPOINT_DB_PATH
        self._graph = None
        self._graph_lock = asyncio.Lock()

    async def _initialize_graph(self) -> None:
        """Инициализирует checkpointer и граф один раз."""
        self._checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(self._checkpoint_db_path))
        self._checkpointer = await self._checkpointer_cm.__aenter__()
        await self._checkpointer.setup()

        self._graph = self._build_graph(self._checkpointer)

    def _build_graph(self, checkpointer: AsyncSqliteSaver):
        """Собрать и скомпилировать граф с SQLite checkpointer-ом.
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

        return graph.compile(checkpointer=checkpointer)

    async def get_graph(self):
        """Вернуть скомпилированный граф (ленивая async-инициализация)."""
        if self._graph is None:
            async with self._graph_lock:
                if self._graph is None:
                    await self._initialize_graph()
        return self._graph

    async def aclose(self) -> None:
        """Явно освобождает ресурсы checkpointer-а и графа."""
        async with self._graph_lock:
            if self._checkpointer_cm is None:
                return
            await self._checkpointer_cm.__aexit__(None, None, None)
            self._checkpointer_cm = None
            self._checkpointer = None
            self._graph = None

    def _resolve_session_thread_id(self, thread_id: str | None) -> str:
        """Разрешить thread_id для текущего запроса.

        - "-1" или None — генерирует новый uuid7.
        - иначе — использует переданный.
        """
        if thread_id == "-1" or thread_id is None:
            return str(uuid.uuid7())
        return thread_id

    async def _prepare_invoke(self, question: str, thread_id: str) -> tuple[RunnableConfig, dict]:
        """Формирует конфиг и обновление состояния перед вызовом графа.

        Использует поддержку Pydantic-состояния в LangGraph + редьюсеры.
        Всегда валидирует snapshot в MetaAgentState для .model_dump().
        Редьюсеры (append_history, merge_dto_store) обрабатывают частичные обновления.
        """
        graph = await self.get_graph()
        runnable_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        state_snapshot = await graph.aget_state(runnable_config)

        # Checkpoint возвращает dict; валидируем в модель для type safety
        if isinstance(state_snapshot.values, dict):
            state_model = MetaAgentState.model_validate(state_snapshot.values)
        else:
            state_model = state_snapshot.values

        snapshot_values = state_model.model_dump()
        state_update = build_turn_state_update(question, snapshot_values)
        return runnable_config, state_update

    async def _finalize_invoke(self, runnable_config: RunnableConfig, result: Any, question: str) -> str:
        """Сохраняет обновлённую историю после выполнения графа и возвращает ответ.

        Использует aupdate_state. Редьюсер history обрабатывает добавление.
        """
        graph = await self.get_graph()
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        answer = result_dict.get("answer", "")
        summarized_history = await build_persisted_history(result_dict)

        # history имеет append-reducer, поэтому для финального усечения используем явную замену
        await graph.aupdate_state(runnable_config, {"history": {"__replace__": summarized_history}})
        return answer

    @traceable(name="meta_agent.invoke_graph_session", run_type="chain")
    async def invoke_graph_session(self, question: str, thread_id: str | None = None) -> MetaAgentResult:
        """Запустить граф в персистентной сессии, заданной thread_id."""
        resolved_thread_id = self._resolve_session_thread_id(thread_id)
        runnable_config, state_update = await self._prepare_invoke(question, resolved_thread_id)

        logger.info("Сессия %s — вопрос: %s", resolved_thread_id, question[:200])
        t0 = time.perf_counter()

        graph = await self.get_graph()
        result = await graph.ainvoke(state_update, runnable_config)
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
