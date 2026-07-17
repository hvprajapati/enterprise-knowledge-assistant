"""Conditional routing logic for the LangGraph pipeline.

Each routing function inspects ``AgentState`` and returns the **name**
of the next node to execute.  LangGraph evaluates these functions
after every node to decide where to go next.

Why conditional instead of sequential?
    A fixed ``START -> A -> B -> C -> END`` graph always executes every
    node, even when unnecessary.  Conditional edges let us skip stages
    (e.g. skip rewrite for simple questions) and handle failures
    gracefully (e.g. short-circuit to END when retrieval returns zero
    results).
"""

from __future__ import annotations

import logging
from enum import StrEnum

from app.agent.state import AgentState

logger = logging.getLogger(__name__)


class RouteDecision(StrEnum):
    """Valid next-step destinations recognised by LangGraph."""

    REWRITE = "rewrite_node"
    RETRIEVE = "retrieve_node"
    GENERATE = "generate_node"
    END = "__end__"


# ---------------------------------------------------------------------------
# routing functions
# ---------------------------------------------------------------------------


def route_after_start(state: AgentState) -> str:
    """Decide whether to rewrite the question or go straight to retrieval.

    Returns
    -------
    str
        ``"rewrite_node"`` when the orchestrator flagged the question
        for rewriting, ``"retrieve_node"`` otherwise.
    """
    if state.get("requires_rewrite"):
        _log_decision("START -> rewrite_node", state)
        return RouteDecision.REWRITE

    _log_decision("START -> retrieve_node  (rewrite skipped)", state)
    return RouteDecision.RETRIEVE


def route_after_rewrite(state: AgentState) -> str:
    """After rewriting, always proceed to retrieval."""
    _log_decision("rewrite_node -> retrieve_node", state)
    return RouteDecision.RETRIEVE


def route_after_retrieve(state: AgentState) -> str:
    """After retrieval, decide: generate an answer or end early.

    If *no* documents were found the graph short-circuits to ``END``
    so the caller receives a graceful "no results" message instead of
    a hallucinated answer.
    """
    results = state.get("search_results", [])
    if not results:
        _log_decision("retrieve_node -> END  (no documents found)", state)
        return RouteDecision.END

    _log_decision("retrieve_node -> generate_node", state)
    return RouteDecision.GENERATE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _log_decision(label: str, state: AgentState) -> None:
    executed = state.get("executed_nodes", [])
    logger.info("Routing: %s  (executed: %s)", label, " -> ".join(executed) if executed else "none")
