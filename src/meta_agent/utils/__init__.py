"""Utilities for the meta-agent: state management, history handling, routing logic, etc.

This package contains helper modules that were previously at the top level of meta_agent.
"""

from .state import (
    MetaAgentState,
    append_history,
    build_turn_state_update,
    merge_dto_store,
    state_to_dict,
)
from .history import (
    build_persisted_history,
    truncate_history,
    truncate_history_list,
    truncate_output_value,
)
from .routing import (
    route_analyzer,
    route_supervisor,
)

__all__ = [
    "MetaAgentState",
    "build_turn_state_update",
    "state_to_dict",
    "build_persisted_history",
    "truncate_history",
    "truncate_history_list",
    "truncate_output_value",
    "route_analyzer",
    "route_supervisor",
    # reducers
    "append_history",
    "merge_dto_store",
]
