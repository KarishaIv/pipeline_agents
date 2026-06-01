"""Мета-агент: улучшенная структура с разделением на utils (state, history, routing), services, nodes и prompts.

Основной вход — meta_graph_manager.invoke_graph_session(question, thread_id).
"""

from src.meta_agent.api_models import (
    AskRequest,
    ErrorResponse,
    MetaAgentApiResponse,
    TextOutput,
    JsonOutput,
    FileOutput,
    ImageOutput,
    AgentOutput,
    AgentArtifact,
)
from src.meta_agent.configs import AVAILABLE_COLLECTIONS, get_collection_catalog
from src.meta_agent.graph import MetaAgentGraphManager, MetaAgentResult, meta_graph_manager
from src.meta_agent.utils.state import MetaAgentState, build_turn_state_update, state_to_dict

__all__ = [
    "AVAILABLE_COLLECTIONS",
    "AskRequest",
    "ErrorResponse",
    "FileOutput",
    "ImageOutput",
    "JsonOutput",
    "MetaAgentApiResponse",
    "MetaAgentGraphManager",
    "MetaAgentResult",
    "MetaAgentState",
    "TextOutput",
    "AgentOutput",
    "AgentArtifact",
    "build_turn_state_update",
    "get_collection_catalog",
    "meta_graph_manager",
    "state_to_dict",
]
