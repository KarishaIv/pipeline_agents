"""Функции маршрутизации для условных ребер LangGraph мета-агента."""

import logging
from enum import Enum
from typing import Any

from src.meta_agent.utils.state import state_to_dict

logger = logging.getLogger("meta_agent.routing")


class SupervisorRoute(str, Enum):
    """Valid routes from supervisor node."""
    DATA_EXTRACTOR = "data_extractor"
    ANALYZER = "analyzer"
    END = "end"


class AnalyzerRoute(str, Enum):
    """Valid routes from analyzer node."""
    CODE_WRITER = "code_writer"
    SUPERVISOR = "supervisor"


class OODRoute(str, Enum):
    """Valid routes from ood_checker node."""
    SUPERVISOR = "supervisor"
    END = "end"


def route_supervisor(state: dict | Any) -> str:
    """Определяет следующий узел по решению супервайзера (next_worker из состояния).

    Validates the route against SupervisorRoute enum and logs errors for invalid routes.
    Defaults to END if invalid.
    """
    state = state_to_dict(state)
    next_worker = state.get("next_worker", "end")

    try:
        validated = SupervisorRoute(next_worker)
        return validated.value
    except ValueError:
        logger.error("Invalid supervisor route '%s', defaulting to 'end'", next_worker)
        return SupervisorRoute.END.value


def route_analyzer(state: dict | Any) -> str:
    """Определяет следующий узел после analyzer: code_writer или supervisor.

    Validates the route against AnalyzerRoute enum.
    """
    state = state_to_dict(state)
    next_worker = state.get("next_worker", "supervisor")

    try:
        validated = AnalyzerRoute(next_worker)
        return validated.value
    except ValueError:
        logger.error("Invalid analyzer route '%s', defaulting to 'supervisor'", next_worker)
        return AnalyzerRoute.SUPERVISOR.value


def route_ood_checker(state: dict | Any) -> str:
    """Определяет следующий узел после ood_checker: supervisor или end."""
    state = state_to_dict(state)
    next_worker = state.get("next_worker", "end")

    try:
        validated = OODRoute(next_worker)
        return validated.value
    except ValueError:
        logger.error("Invalid ood_checker route '%s', defaulting to 'end'", next_worker)
        return OODRoute.END.value
