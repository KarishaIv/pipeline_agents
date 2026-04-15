"""Инструменты контроля вычислительного бюджета агентов."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sgr_agent_core.base_tool import BaseTool

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext


class RemainingStepsTool(BaseTool):
    """Показать текущий и оставшийся лимит шагов/вызовов инструментов агента."""

    tool_name = "remaining_steps"
    description = (
        "Вернуть текущий итерационный бюджет агента: текущую итерацию, "
        "максимум итераций и оценку оставшихся вызовов инструментов."
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        max_iterations = int(config.execution.max_iterations)
        current_iteration = int(context.iteration)
        remaining_iterations = max(max_iterations - current_iteration, 0)
        payload = {
            "current_iteration": current_iteration,
            "max_iterations": max_iterations,
            "remaining_iterations": remaining_iterations,
            "remaining_tool_calls_estimate": remaining_iterations,
            "note": (
                "Каждый шаг агента обычно включает один вызов инструмента; "
                "оценка оставшихся вызовов приблизительная."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)
