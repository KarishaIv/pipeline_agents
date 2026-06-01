"""Оркестратор мета-агента на LangGraph.

Содержит MetaAgentGraphManager, MetaAgentState, сессионное управление,
подготовку invoke и finalize
"""

import logging
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, NamedTuple

import aiosqlite

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from langsmith import traceable
from src.meta_agent.configs import CHECKPOINT_DB_PATH
from src.meta_agent.nodes import (
    analyzer_node,
    code_writer_node,
    data_extractor_node,
    ood_checker_node,
    supervisor_node,
)
from src.meta_agent.utils.state import MetaAgentState, build_turn_state_update, state_to_dict
from src.meta_agent.utils.history import (
    build_persisted_history,
)
from src.meta_agent.utils.routing import route_analyzer, route_ood_checker, route_supervisor
from src.meta_agent.utils.thread_ids import generate_thread_id

logger = logging.getLogger("meta_agent")
DTO_PAYLOAD_MSGPACK_MODULE = ("src.meta_agent.dto", "DtoPayload")
AGENT_ARTIFACT_MSGPACK_MODULE = ("src.meta_agent.output_models", "AgentArtifact")


class MetaAgentResult(NamedTuple):
    """Результат выполнения мета-агента."""
    thread_id: str
    outputs: list  # list[AgentOutput]


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
        self._checkpointer_cm = self._open_checkpointer()
        self._checkpointer = await self._checkpointer_cm.__aenter__()
        await self._checkpointer.setup()

        self._graph = self._build_graph(self._checkpointer)

    @asynccontextmanager
    async def _open_checkpointer(self) -> AsyncIterator[AsyncSqliteSaver]:
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                DTO_PAYLOAD_MSGPACK_MODULE,
                AGENT_ARTIFACT_MSGPACK_MODULE,
            ]
        )
        async with aiosqlite.connect(str(self._checkpoint_db_path)) as conn:
            yield AsyncSqliteSaver(conn, serde=serde)

    def _build_graph(self, checkpointer: AsyncSqliteSaver):
        """Собрать и скомпилировать граф с SQLite checkpointer-ом.
        Uses Pydantic MetaAgentState with LangGraph reducers for history and dto_store.
        """
        graph = StateGraph(MetaAgentState)

        graph.add_node("supervisor", supervisor_node)
        graph.add_node("data_extractor", data_extractor_node)
        graph.add_node("analyzer", analyzer_node)
        graph.add_node("code_writer", code_writer_node)
        graph.add_node("ood_checker", ood_checker_node)

        def _route_from_start(s: dict | Any) -> str:
            s = state_to_dict(s)
            return "supervisor" if s.get("force_bypass_ood") else "ood_checker"

        graph.add_conditional_edges(
            START,
            _route_from_start,
            {"ood_checker": "ood_checker", "supervisor": "supervisor"},
        )
        graph.add_conditional_edges(
            "ood_checker",
            route_ood_checker,
            {"supervisor": "supervisor", "end": END},
        )
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
            return generate_thread_id()
        return thread_id

    async def _prepare_invoke(self, question: str, thread_id: str) -> tuple[RunnableConfig, dict]:
        """Формирует конфиг и обновление состояния перед вызовом графа.

        Использует поддержку Pydantic-состояния в LangGraph + редьюсеры.
        Всегда валидирует snapshot в MetaAgentState для .model_dump().
        Редьюсеры (append_list, merge_dto_store) обрабатывают частичные обновления.
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

    async def _finalize_invoke(self, runnable_config: RunnableConfig, result: Any) -> list:
        """Сохраняет обновлённую историю после выполнения графа и возвращает outputs.

        Использует aupdate_state. Редьюсер history обрабатывает добавление.
        Преобразует artifacts (charts) в ImageOutput объекты.
        Возвращает список outputs.
        """
        from src.meta_agent.api_models import ImageOutput, JsonOutput, FileOutput

        graph = await self.get_graph()
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result

        # Extract outputs from final state
        outputs = list(result_dict.get("outputs", []))

        # Convert artifacts to outputs (charts → ImageOutput, JSON → JsonOutput, CSV/files → FileOutput)
        artifacts = result_dict.get("artifacts", [])
        for artifact in artifacts:
            # Get artifact attributes (handle both dict and Pydantic model)
            artifact_dict = artifact.model_dump() if hasattr(artifact, "model_dump") else artifact
            artifact_kind = artifact_dict.get("kind", "")
            filename = artifact_dict.get("filename", "")
            mime_type = artifact_dict.get("mime_type", "application/octet-stream")
            caption = artifact_dict.get("caption")
            metadata = artifact_dict.get("metadata", {})
            file_path = artifact_dict.get("path", "")

            if artifact_kind == "chart" and filename:
                url = f"/artifacts/{filename}"
                image_output = ImageOutput(
                    url=url,
                    caption=caption,
                    alt_text=f"Chart: {metadata.get('chart_type', 'visualization')}",
                    mime_type=mime_type,
                )
                outputs.append(image_output)
            elif artifact_kind == "data" and filename:
                # JSON raw data artifact: load content into JsonOutput when possible
                try:
                    import json
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                    json_output = JsonOutput(
                        data=data,
                        caption=caption or filename,
                    )
                    outputs.append(json_output)
                except Exception as e:
                    logger.warning("Failed to load JSON artifact %s: %s", filename, e)
                    # Fallback: expose as downloadable file
                    download_url = f"/artifacts/{filename}"
                    file_output = FileOutput(
                        filename=filename,
                        mime_type=mime_type or "application/json",
                        download_url=download_url,
                        caption=caption,
                    )
                    outputs.append(file_output)
            elif artifact_kind in ("csv", "file", "pdf") and filename:
                download_url = f"/artifacts/{filename}"
                file_output = FileOutput(
                    filename=filename,
                    mime_type=mime_type,
                    download_url=download_url,
                    caption=caption,
                )
                outputs.append(file_output)

        summarized_history = await build_persisted_history(result_dict)

        # history имеет append-reducer, поэтому для финального усечения используем явную замену
        await graph.aupdate_state(runnable_config, {"history": {"__replace__": summarized_history}})
        return outputs

    @traceable(name="meta_agent.invoke_graph_session", run_type="chain")
    async def invoke_graph_session(self, question: str, thread_id: str | None = None) -> MetaAgentResult:
        """Запустить граф в персистентной сессии, заданной thread_id."""
        resolved_thread_id = self._resolve_session_thread_id(thread_id)
        runnable_config, state_update = await self._prepare_invoke(question, resolved_thread_id)

        logger.info("Сессия %s — вопрос: %s", resolved_thread_id, question[:200])
        t0 = time.perf_counter()

        graph = await self.get_graph()
        result = await graph.ainvoke(state_update, runnable_config)
        outputs = await self._finalize_invoke(runnable_config, result)

        elapsed = time.perf_counter() - t0
        logger.info("Граф завершён за %.1fс", elapsed)
        return MetaAgentResult(thread_id=resolved_thread_id, outputs=outputs)

    @traceable(name="meta_agent.invoke_graph", run_type="chain")
    async def invoke_graph(self, question: str) -> list:
        """Запуск без сохранения состояния: для каждого вызова создаётся новая сессия и возвращаются только outputs."""
        out = await self.invoke_graph_session(question, "-1")
        return out.outputs


meta_graph_manager = MetaAgentGraphManager()
