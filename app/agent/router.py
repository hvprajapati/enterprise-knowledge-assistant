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

    TOOL = "tool_node"
    REWRITE = "rewrite_node"
    RETRIEVE = "retrieve_node"
    GENERATE = "generate_node"
    END = "__end__"


# ---------------------------------------------------------------------------
# routing functions
# ---------------------------------------------------------------------------


def route_after_planner(state: AgentState) -> str:
    """Decide whether to execute tools, rewrite, or go straight to retrieval.

    Priority: tools > rewrite > retrieve.

    Returns
    -------
    str
        ``"tool_node"`` when the execution plan lists expected tools,
        ``"rewrite_node"`` when rewriting is flagged,
        ``"retrieve_node"`` otherwise.
    """
    plan = state.get("execution_plan", {})
    raw_tools = plan.get("expected_tools", [])
    expected_tools: list[str] = (
        [str(t) for t in raw_tools] if isinstance(raw_tools, list) else []
    )

    if expected_tools:
        _log_decision(f"planner_node -> tool_node  (tools: {expected_tools})", state)
        return RouteDecision.TOOL

    if state.get("requires_rewrite"):
        _log_decision("planner_node -> rewrite_node", state)
        return RouteDecision.REWRITE

    _log_decision("planner_node -> retrieve_node  (rewrite skipped)", state)
    return RouteDecision.RETRIEVE


def route_after_tool(state: AgentState) -> str:
    """After tool execution, proceed to rewrite or retrieve.

    Tool results are stored in state — the retrieve/generate nodes
    can consume them later.
    """
    if state.get("requires_rewrite"):
        _log_decision("tool_node -> rewrite_node", state)
        return RouteDecision.REWRITE

    _log_decision("tool_node -> retrieve_node", state)
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


def route_after_retry(state: AgentState) -> str:
    """Read the retry decision and route to the appropriate node.

    Returns
    -------
    str
        One of ``"planner_node"``, ``"retrieve_node"``,
        ``"generate_node"``, or ``"__end__"``.
    """
    decision = state.get("retry_decision", {})
    next_node = str(decision.get("next_node", "__end__"))

    if next_node == "__end__":
        _log_decision("retry_node -> END  (no retry or limit reached)", state)
    else:
        _log_decision(f"retry_node -> {next_node}  (retry loop)", state)

    return next_node


def _log_decision(label: str, state: AgentState) -> None:
    executed = state.get("executed_nodes", [])
    logger.info("Routing: %s  (executed: %s)", label, " -> ".join(executed) if executed else "none")
