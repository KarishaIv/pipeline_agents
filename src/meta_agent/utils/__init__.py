"""Utilities for the meta-agent: state management, history handling, routing logic, etc.

This package contains helper modules that were previously at the top level of meta_agent.
"""

from src.meta_agent.utils.state import (
    MetaAgentState,
    append_list,
    build_turn_state_update,
    merge_dto_store,
    state_to_dict,
)
from src.meta_agent.utils.history import (
    build_persisted_history,
    truncate_history,
    truncate_history_list,
    truncate_output_value,
)
from src.meta_agent.utils.routing import (
    route_analyzer,
    route_supervisor,
    SupervisorRoute,
    AnalyzerRoute,
)
from src.meta_agent.utils.json_responses import (
    json_success,
    json_error,
    serialize_tool_result,
    json_node_failure,
)
from src.meta_agent.utils.thread_ids import generate_thread_id

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
    "SupervisorRoute",
    "AnalyzerRoute",
    "json_success",
    "json_error",
    "serialize_tool_result",
    "json_node_failure",
    "generate_thread_id",
    # reducers
    "append_list",
    "merge_dto_store",
]
