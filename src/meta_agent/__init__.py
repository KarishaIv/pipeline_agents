"""Мета-агент: улучшенная структура с разделением на utils (state, history, routing), services, nodes и prompts.

Основной вход — meta_graph_manager.invoke_graph_session(question, thread_id).
"""

from .catalog import AVAILABLE_COLLECTIONS, COLLECTION_DESCRIPTIONS, get_collection_catalog
from .graph import MetaAgentGraphManager, MetaAgentResult, meta_graph_manager
from .utils.state import MetaAgentState, build_turn_state_update, state_to_dict

__all__ = [
    "AVAILABLE_COLLECTIONS",
    "MetaAgentGraphManager",
    "MetaAgentResult",
    "MetaAgentState",
    "build_turn_state_update",
    "get_collection_catalog",
    "meta_graph_manager",
    "state_to_dict",
]
