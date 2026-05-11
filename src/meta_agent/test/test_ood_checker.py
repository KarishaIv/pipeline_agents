"""Tests for OOD checker node (first node in graph), /force bypass, routing and redirect message guidance.

BDD-style: these tests are written before implementation and should initially fail (red), then pass after (green).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.meta_agent.tools.output_tools import OODCheckResult
from src.meta_agent.nodes import ood_checker_node
from src.meta_agent.utils.routing import OODRoute, route_ood_checker
from src.meta_agent.utils.state import MetaAgentState, build_turn_state_update


def test_ood_check_result_model():
    """OODCheckResult is a simple Pydantic model for structured LLM output (not a Tool)."""
    r = OODCheckResult(is_relevant=True, redirect_message=None)
    assert r.is_relevant is True
    assert r.redirect_message is None

    r2 = OODCheckResult(is_relevant=False, redirect_message="Not related.")
    assert r2.is_relevant is False
    assert "Not related" in r2.redirect_message


def test_ood_route_enum():
    """OODRoute enum for conditional edges."""
    assert OODRoute.SUPERVISOR.value == "supervisor"
    assert OODRoute.END.value == "end"


def test_route_ood_checker_supervisor():
    """route_ood_checker returns 'supervisor' when next_worker set accordingly."""
    state = {"next_worker": "supervisor"}
    assert route_ood_checker(state) == "supervisor"


def test_route_ood_checker_end():
    """route_ood_checker returns 'end' (with outputs) when off-topic."""
    state = {"next_worker": "end"}
    assert route_ood_checker(state) == "end"


@pytest.mark.asyncio
async def test_ood_checker_node_relevant_routes_to_supervisor(mocker, meta_state):
    """When LLM says relevant, next_worker=supervisor, no outputs."""
    mock_result = OODCheckResult(is_relevant=True, redirect_message=None)
    mocker.patch(
        "src.meta_agent.nodes.robust_llm_call",
        new=AsyncMock(return_value=mock_result),
    )

    state = meta_state.model_copy()
    state.question = "How many personas were created in the simulation?"
    result = await ood_checker_node(state, MagicMock())

    assert result["next_worker"] == "supervisor"
    assert result.get("outputs", []) == []


@pytest.mark.asyncio
async def test_ood_checker_node_off_topic_appends_force_guidance(mocker, meta_state):
    """When LLM says not relevant, appends fixed /force guidance to redirect_message (post-LLM)."""
    mock_result = OODCheckResult(
        is_relevant=False, redirect_message="This question is not about the simulation pipeline."
    )
    mocker.patch(
        "src.meta_agent.nodes.robust_llm_call",
        new=AsyncMock(return_value=mock_result),
    )

    state = meta_state.model_copy()
    state.question = "What is the weather today?"
    result = await ood_checker_node(state, MagicMock())

    assert result["next_worker"] == "end"
    outputs = result.get("outputs", [])
    assert len(outputs) == 1
    text = outputs[0]["text"]
    assert "not about the simulation" in text
    assert "/force" in text  # guidance appended after LLM response (Russian text)


def test_build_turn_state_update_detects_force_prefix():
    """build_turn_state_update strips /force prefix and sets force_bypass_ood=True."""
    snapshot = {"dto_store": {}, "outputs": [], "artifacts": []}
    update = build_turn_state_update("/force What about the simulation results?", snapshot)

    assert update["question"] == "What about the simulation results?"
    assert update.get("force_bypass_ood") is True


def test_meta_state_has_force_bypass_field():
    """MetaAgentState must have force_bypass_ood field default False."""
    s = MetaAgentState(question="test")
    assert hasattr(s, "force_bypass_ood")
    assert s.force_bypass_ood is False


@pytest.mark.asyncio
async def test_graph_starts_with_ood_checker_or_bypasses(mocker):
    """Graph build: START routes to ood_checker unless force_bypass_ood, then to supervisor or end."""
    # This high-level structure test will be verified after graph changes
    from src.meta_agent.graph import MetaAgentGraphManager

    manager = MetaAgentGraphManager()
    graph = await manager.get_graph()
    # After implementation, graph should have 'ood_checker' node and conditional from START
    assert "ood_checker" in graph.nodes  # or inspect graph structure
    # Further topology checks can be added; for now ensure no crash on build
