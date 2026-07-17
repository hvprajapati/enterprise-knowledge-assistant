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
    tool_execution_plan: dict[str, object]  # serialised ToolExecutionPlan
    tool_results: list[dict[str, object]]   # serialised list[ToolResult]

    # -- reflection output --------------------------------------------------
    reflection_result: dict[str, object]  # serialised ReflectionResult

    # -- validation output --------------------------------------------------
    validation_result: dict[str, object]  # serialised ValidationResult

    # -- retry output -------------------------------------------------------
    retry_decision: dict[str, object]  # serialised RetryDecision
    retry_count: int

    # -- multi-agent tracking ----------------------------------------------
    completed_agents: list[str]
    current_agent: str
    execution_history: list[dict[str, object]]

    # -- metadata ----------------------------------------------------------
    error: str | None
    executed_nodes: list[str]
    requires_rewrite: bool
