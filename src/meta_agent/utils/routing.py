"""Функции маршрутизации для условных ребер LangGraph мета-агента."""

from typing import Any

from src.meta_agent.utils.state import state_to_dict


def route_supervisor(state: dict | Any) -> str:
    """Определяет следующий узел по решению супервайзера (next_worker из состояния)."""
    state = state_to_dict(state)
    return state.get("next_worker", "end")


def route_analyzer(state: dict | Any) -> str:
    """Определяет следующий узел после analyzer: code_writer или supervisor."""
    state = state_to_dict(state)
    next_worker = state.get("next_worker", "supervisor")
    if next_worker == "code_writer":
        return "code_writer"
    return "supervisor"
