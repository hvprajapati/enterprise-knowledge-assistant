"""Typed state carried through the LangGraph execution."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Mutable state passed between graph nodes.

    All fields are optional at construction — nodes populate them
    as the graph executes.
    """

    # -- inputs -----------------------------------------------------------
    question: str

    # -- pipeline outputs --------------------------------------------------
    rewritten_question: str | None
    search_results: list[dict[str, object]]  # serialised SearchResult
    _prompt: str  # carry-forward for generate_node
    answer: str | None

    # -- planner output ----------------------------------------------------
    execution_plan: dict[str, object]  # serialised ExecutionPlan

    # -- tool output --------------------------------------------------------
    tool_decision: dict[str, object]   # serialised ToolDecision
    tool_result: dict[str, object]     # serialised ToolResult

    # -- reflection output --------------------------------------------------
    reflection_result: dict[str, object]  # serialised ReflectionResult

    # -- validation output --------------------------------------------------
    validation_result: dict[str, object]  # serialised ValidationResult

    # -- retry output -------------------------------------------------------
    retry_decision: dict[str, object]  # serialised RetryDecision
    retry_count: int

    # -- metadata ----------------------------------------------------------
    error: str | None
    executed_nodes: list[str]
    requires_rewrite: bool
