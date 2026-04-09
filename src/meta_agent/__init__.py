"""Мета-агент: конвейер с циклом супервайзера поверх ToolCallingAgent-воркеров."""

from src.meta_agent.graph import (
    MetaAgentGraphManager,
    MetaAgentResult,
    meta_graph_manager,
)

__all__ = [
    "MetaAgentGraphManager",
    "MetaAgentResult",
    "meta_graph_manager",
]
