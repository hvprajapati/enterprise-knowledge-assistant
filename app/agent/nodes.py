"""LangGraph node functions.

Each node delegates to a specialised agent via ``AgentRegistry``.
This keeps the graph topology stable while allowing agents to be
swapped, tested, and monitored independently.

Nodes access global service singletons initialised at module-import
time by ``EnterpriseKnowledgeAgent.__init__``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.reflection import ReflectionResult
from app.agent.reflection.reflection import ReflectionEngine
from app.agent.retry import RetryDecision
from app.agent.retry.retry_engine import RetryEngine
from app.agent.state import AgentState
from app.agent.validation import ValidationResult
from app.agent.validation.validator import AnswerValidator
from app.tools import ToolResult
from app.tools.planner import (
    MultiToolPlanner,
    SequentialToolExecutor,
    ToolExecutionPlan,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# service placeholders — set by ``EnterpriseKnowledgeAgent.__init__``
# ---------------------------------------------------------------------------

_services: dict[str, Any] = {}


def _get_service(name: str) -> Any:
    """Return the named service, raising if it hasn't been configured."""
    svc = _services.get(name)
    if svc is None:
        raise RuntimeError(
            f"Agent service '{name}' not configured. Did you call EnterpriseKnowledgeAgent(...)?"
        )
    return svc


# ---------------------------------------------------------------------------
# node implementations
# ---------------------------------------------------------------------------


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """Initialise multi-agent execution tracking.

    The supervisor agent coordinates the pipeline.  This node sets up
    state fields that downstream agent nodes update.
    """
    _record(state, "supervisor")
    return {
        "completed_agents": [],
        "execution_history": [],
        "current_agent": "supervisor",
        "executed_nodes": _track(state, "supervisor"),
    }


def _delegate(state: AgentState, agent_name: str) -> dict[str, Any] | None:
    """Try to delegate to an agent; return ``None`` if not available.

    Callers should fall back to their original inline implementation
    when this returns ``None``.
    """
    from app.agents.registry import AgentRegistry

    try:
        registry: AgentRegistry = _get_service("agent_registry")
        agent = registry.lookup(agent_name)
        if agent is not None:
            return agent.execute_with_logging(state)
    except RuntimeError:
        pass
    return None


def planner_node(state: AgentState) -> dict[str, Any]:
    """Analyse question — delegates to PlannerAgent or runs inline."""
    _record(state, "planner")
    delegated = _delegate(state, "planner")
    if delegated is not None:
        return delegated
    question = state["question"]
    planner = _get_service("planner")
    plan = planner.create_plan(question)
    return {
        "execution_plan": plan.model_dump(),
        "requires_rewrite": plan.requires_rewrite,
        "executed_nodes": _track(state, "planner"),
    }


def tool_planner_node(state: AgentState) -> dict[str, Any]:
    """Analyse the question and produce a ``ToolExecutionPlan``.

    Reads
    -----
    question from state.

    Sets
    ----
    tool_execution_plan : dict  (serialised ToolExecutionPlan)
    executed_nodes      : appended ``"tool_planner"``

    When no tools are needed the plan is empty and downstream nodes
    skip execution.
    """
    _record(state, "tool_planner")
    question = state["question"]

    planner: MultiToolPlanner = _get_service("tool_planner")

    plan: ToolExecutionPlan = planner.plan_tools(question)

    logger.info(
        "Tool planner node — tools=%d  names=%s",
        plan.tool_count,
        [t.tool_name for t in plan.tools],
    )

    return {
        "tool_execution_plan": plan.model_dump(),
        "executed_nodes": _track(state, "tool_planner"),
    }


def tool_executor_node(state: AgentState) -> dict[str, Any]:
    """Execute the ``ToolExecutionPlan`` produced by ``tool_planner_node``.

    Reads
    -----
    tool_execution_plan from state.

    Sets
    ----
    tool_results   : list[dict]  (serialised list[ToolResult])
    executed_nodes : appended ``"tool_executor"``

    Tools run sequentially.  Optional tool failures are skipped;
    required tool failures abort the plan.  The graph never crashes
    — failures are captured in ToolResult.error.
    """
    _record(state, "tool_executor")
    plan_dict = state.get("tool_execution_plan", {})

    # Reconstruct the plan from serialised dict
    plan = ToolExecutionPlan.model_validate(plan_dict)

    if plan.is_empty():
        logger.info("Tool executor node — empty plan, skipping.")
        return {
            "tool_results": [],
            "executed_nodes": _track(state, "tool_executor"),
        }

    executor: SequentialToolExecutor = _get_service("tool_sequential_executor")

    results: list[ToolResult] = executor.execute(plan)

    serialised = [r.model_dump() for r in results]

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        "Tool executor node — %d/%d tools succeeded",
        succeeded,
        len(results),
    )

    return {
        "tool_results": serialised,
        "executed_nodes": _track(state, "tool_executor"),
    }


def rewrite_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the question (delegates to QueryRewriter directly)."""
    _record(state, "rewrite")
    question = state["question"]
    rewriter = _get_service("rewriter")
    rewritten = rewriter.rewrite(question)
    logger.info("Rewrite node — %d -> %d chars", len(question), len(rewritten))
    return {
        "rewritten_question": rewritten,
        "executed_nodes": _track(state, "rewrite"),
    }


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Delegate to ``RetrievalAgent`` or run inline."""
    _record(state, "retrieve")
    delegated = _delegate(state, "retrieval")
    if delegated is not None:
        return delegated
    # Fallback: inline retrieval
    question = state.get("rewritten_question") or state["question"]
    retriever = _get_service("retriever")
    reranker = _get_service("reranker")
    embed = _get_service("embedding_service")
    orchestrator = _get_service("orchestrator")
    prompt_builder = _get_service("prompt_builder")
    plan = orchestrator.plan(question)
    embedding = embed.embed_query(question)
    candidates = retriever.retrieve(embedding, top_k=plan.vector_top_k)
    reranked = reranker.rerank(question, candidates, top_k=5)
    tool_results: list[dict[str, object]] = state.get("tool_results", [])
    prompt = prompt_builder.build_prompt(
        question, reranked, tool_results=tool_results if tool_results else None
    )
    serialised = [
        {"chunk_id": str(r.chunk.chunk_id), "text": r.chunk.text,
         "score": r.score, "filename": r.chunk.metadata.filename,
         "page": r.chunk.page_number}
        for r in reranked
    ]
    return {
        "search_results": serialised, "_prompt": prompt,
        "executed_nodes": _track(state, "retrieve"),
    }


def generate_node(state: AgentState) -> dict[str, Any]:
    """Delegate to ``GenerationAgent`` or run inline."""
    _record(state, "generate")
    delegated = _delegate(state, "generation")
    if delegated is not None:
        return delegated
    prompt = state.get("_prompt", "")
    llm = _get_service("llm")
    answer = llm.generate(prompt)
    return {"answer": answer, "executed_nodes": _track(state, "generate")}


def reflection_node(state: AgentState) -> dict[str, Any]:
    """Delegate to ``ReflectionAgent`` or run inline."""
    _record(state, "reflection")
    delegated = _delegate(state, "reflection")
    if delegated is not None:
        return delegated
    question = state.get("rewritten_question") or state["question"]
    answer = state.get("answer") or ""
    search_results = state.get("search_results", [])
    engine: ReflectionEngine = _get_service("reflection_engine")
    result: ReflectionResult = engine.reflect(
        question=question, answer=answer, retrieved_chunks=search_results,
    )
    return {
        "reflection_result": result.model_dump(),
        "executed_nodes": _track(state, "reflection"),
    }


def validation_node(state: AgentState) -> dict[str, Any]:
    """Delegate to ``ValidationAgent`` or run inline."""
    _record(state, "validation")
    delegated = _delegate(state, "validation")
    if delegated is not None:
        return delegated
    reflection_dict = state.get("reflection_result", {})
    reflection = ReflectionResult.model_validate(reflection_dict)
    validator: AnswerValidator = _get_service("validator")
    result: ValidationResult = validator.validate(reflection)
    return {
        "validation_result": result.model_dump(),
        "executed_nodes": _track(state, "validation"),
    }


def retry_node(state: AgentState) -> dict[str, Any]:
    """Decide whether to loop back through the pipeline.

    Reads
    -----
    validation_result, reflection_result, retry_count from state.

    Sets
    ----
    retry_decision : dict    (serialised RetryDecision)
    retry_count    : int      (incremented if a retry is attempted)
    executed_nodes : appended ``"retry"``

    On failure stores a ``NO_RETRY`` fallback so the graph delivers
    the current answer rather than looping infinitely.
    """
    _record(state, "retry")
    validation_dict = state.get("validation_result", {})
    reflection_dict = state.get("reflection_result", {})
    current_retry_count: int = state.get("retry_count", 0)

    # Reconstruct Pydantic models from serialised dicts
    validation = ValidationResult.model_validate(validation_dict)
    reflection = ReflectionResult.model_validate(reflection_dict)

    engine: RetryEngine = _get_service("retry_engine")

    decision: RetryDecision = engine.decide_retry(
        validation=validation,
        reflection=reflection,
        retry_count=current_retry_count,
    )

    # Increment retry count if retrying (so the next loop sees it)
    new_retry_count = current_retry_count + 1 if decision.should_retry else current_retry_count

    logger.info(
        "Retry node — should_retry=%s  strategy=%s  next=%s  count=%d→%d",
        decision.should_retry,
        decision.strategy.value,
        decision.next_node,
        current_retry_count,
        new_retry_count,
    )

    return {
        "retry_decision": decision.model_dump(),
        "retry_count": new_retry_count,
        "executed_nodes": _track(state, "retry"),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record(state: AgentState, name: str) -> None:
    logger.info("Executing node: %s", name)


def _track(state: AgentState, name: str) -> list[str]:
    nodes: list[str] = list(state.get("executed_nodes", []))
    nodes.append(name)
    return nodes
