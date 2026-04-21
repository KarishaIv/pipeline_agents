"""Tests for budget_tools.py - RemainingStepsTool.

Tests verify budget calculation, JSON output, and integration with AgentContext/Config.
"""
import json
from unittest.mock import MagicMock

import pytest

from src.meta_agent.tools.budget_tools import RemainingStepsTool


@pytest.mark.asyncio
async def test_remaining_steps_tool():
    """Test RemainingStepsTool calculates and returns correct budget JSON."""
    tool = RemainingStepsTool()
    mock_context = MagicMock()
    mock_context.iteration = 5
    mock_config = MagicMock()
    mock_config.execution = MagicMock(max_iterations=20)

    result = await tool(mock_context, mock_config)

    assert isinstance(result, str)
    payload = json.loads(result)
    assert payload["current_iteration"] == 5
    assert payload["max_iterations"] == 20
    assert payload["remaining_iterations"] == 15
    assert payload["remaining_tool_calls_estimate"] == 15
    assert "note" in payload
    assert "приблизительная" in payload["note"]


@pytest.mark.parametrize("current,maximum,expected_remaining", [(0, 10, 10), (5, 5, 0), (15, 10, 0)])
@pytest.mark.asyncio
async def test_remaining_steps_boundary_values(current, maximum, expected_remaining):
    """Test edge cases for iteration calculations."""
    tool = RemainingStepsTool()
    mock_context = MagicMock()
    mock_context.iteration = current
    mock_config = MagicMock()
    mock_config.execution = MagicMock(max_iterations=maximum)

    result_str = await tool(mock_context, mock_config)
    result = json.loads(result_str)
    assert result["remaining_iterations"] == expected_remaining
    assert result["current_iteration"] == current
    assert result["max_iterations"] == maximum
