"""LangGraph node functions.

Each node receives the current ``AgentState`` and returns a partial
state dictionary to merge.  Nodes access global service singletons
that are initialised once at module-import time.
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
from app.tools import ToolDecision, ToolExecutor, ToolResult

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


def planner_node(state: AgentState) -> dict[str, Any]:
    """Analyse question and create an ``ExecutionPlan``.

    Sets
    ----
    execution_plan : dict     (serialised ExecutionPlan)
    requires_rewrite : bool   (from plan)
    executed_nodes  : list    (appended ``"planner"``)
    """
    _record(state, "planner")
    question = state["question"]

    planner = _get_service("planner")
    plan = planner.create_plan(question)

    return {
        "execution_plan": plan.model_dump(),
        "requires_rewrite": plan.requires_rewrite,
        "executed_nodes": _track(state, "planner"),
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute tools identified in the execution plan.

    Reads
    -----
    execution_plan, question from state.

    Sets
    ----
    tool_decision : dict    (serialised ToolDecision)
    tool_result   : dict    (serialised ToolResult)
    executed_nodes : appended ``"tool"``

    When no tools are expected this node is a fast no-op — it stores
    a ``skip_tools`` decision and returns immediately.

    On tool failure stores the error in ``ToolResult`` but does NOT
    crash the graph — the RAG pipeline still runs.
    """
    _record(state, "tool")
    plan_dict = state.get("execution_plan", {})
    question = state["question"]

    raw_tools = plan_dict.get("expected_tools", [])
    expected_tools: list[str] = (
        [str(t) for t in raw_tools] if isinstance(raw_tools, list) else []
    )

    # --- no tools needed ---
    if not expected_tools:
        decision = ToolDecision.skip_tools()
        logger.info("Tool node — no tools expected, skipping.")
        return {
            "tool_decision": decision.model_dump(),
            "tool_result": {},
            "executed_nodes": _track(state, "tool"),
        }

    # --- select & execute the first expected tool ---
    # In the future this will be replaced with an LLM-based selector.
    executor: ToolExecutor = _get_service("tool_executor")
    tool_name = expected_tools[0]

    # Build arguments: map the question to the tool's required params
    arguments = _build_tool_arguments(tool_name, question)

    decision = ToolDecision(
        use_tool=True,
        tool_name=tool_name,
        arguments=arguments,
        confidence=1.0,
        reasoning=f"Planner requested tool: {tool_name}",
    )

    logger.info(
        "Tool node — invoking tool=%s  args=%s",
        tool_name,
        decision.arguments,
    )

    result: ToolResult = executor.execute(
        tool_name=tool_name,
        arguments=decision.arguments,
    )

    if result.success:
        logger.info(
            "Tool node — tool=%s succeeded  latency=%.0fms",
            tool_name,
            result.execution_time_ms,
        )
    else:
        logger.warning(
            "Tool node — tool=%s failed: %s",
            tool_name,
            result.error,
        )

    return {
        "tool_decision": decision.model_dump(),
        "tool_result": result.model_dump(),
        "executed_nodes": _track(state, "tool"),
    }


def rewrite_node(state: AgentState) -> dict[str, Any]:
    """Expand / clarify the question using ``QueryRewriter``.

    Sets
    ----
    rewritten_question : str
    executed_nodes      : appended ``"rewrite"``
    """
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
    """Search FAISS and return ranked results.

    Uses the (possibly rewritten) question and the orchestrator's plan
    to decide which retrieval stages to execute.

    Sets
    ----
    search_results : list[dict]
    executed_nodes  : appended ``"retrieve"``
    """
    _record(state, "retrieve")
    question = state.get("rewritten_question") or state["question"]

    retriever = _get_service("retriever")
    reranker = _get_service("reranker")
    embed = _get_service("embedding_service")
    orchestrator = _get_service("orchestrator")
    prompt_builder = _get_service("prompt_builder")

    # ---- run the plan-driven retrieval pipeline ----
    plan = orchestrator.plan(question)
    logger.info(
        "Retrieve node — plan type=%s  stages=%s",
        plan.question_type.value,
        plan.stages_enabled,
    )

    # Embed + retrieve (simplified: single query, respects plan)
    embedding = embed.embed_query(question)
    candidates = retriever.retrieve(embedding, top_k=plan.vector_top_k)

    # Rerank
    reranked = reranker.rerank(question, candidates, top_k=5)

    # Build prompt (so we have context ready for generate_node)
    # Include tool results if available
    tool_result: dict[str, object] = state.get("tool_result", {})
    prompt = prompt_builder.build_prompt(
        question, reranked, tool_result=tool_result if tool_result else None
    )

    # Serialise for state
    serialised = [
        {
            "chunk_id": str(r.chunk.chunk_id),
            "text": r.chunk.text,
            "score": r.score,
            "filename": r.chunk.metadata.filename,
            "page": r.chunk.page_number,
        }
        for r in reranked
    ]

    logger.info(
        "Retrieve node — %d candidates -> %d reranked",
        len(candidates),
        len(reranked),
    )

    return {
        "search_results": serialised,
        "_prompt": prompt,  # carry-forward for generate_node
        "executed_nodes": _track(state, "retrieve"),
    }


def generate_node(state: AgentState) -> dict[str, Any]:
    """Call the LLM with the assembled prompt.

    Expects ``_prompt`` to have been set by ``retrieve_node``.

    Sets
    ----
    answer : str
    executed_nodes : appended ``"generate"``
    """
    _record(state, "generate")
    prompt = state.get("_prompt", "")
    llm = _get_service("llm")

    logger.info("Generate node — prompt=%d chars", len(prompt))
    answer = llm.generate(prompt)

    return {
        "answer": answer,
        "executed_nodes": _track(state, "generate"),
    }


def reflection_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the generated answer for quality and groundedness.

    Reads
    -----
    question, answer, search_results from state.

    Sets
    ----
    reflection_result : dict  (serialised ReflectionResult)
    executed_nodes     : appended ``"reflection"``

    On failure stores a neutral ``ReflectionResult`` so the graph
    never halts because of a broken evaluator.
    """
    _record(state, "reflection")
    question = state.get("rewritten_question") or state["question"]
    answer = state.get("answer") or ""
    search_results = state.get("search_results", [])

    engine: ReflectionEngine = _get_service("reflection_engine")

    result: ReflectionResult = engine.reflect(
        question=question,
        answer=answer,
        retrieved_chunks=search_results,
    )

    logger.info(
        "Reflection node — quality=%s  grounded=%s  confidence=%.2f",
        result.answer_quality.value,
        result.grounded.value,
        result.confidence_score,
    )

    return {
        "reflection_result": result.model_dump(),
        "executed_nodes": _track(state, "reflection"),
    }


def validation_node(state: AgentState) -> dict[str, Any]:
    """Gate the answer using reflection scores and thresholds.

    Reads
    -----
    reflection_result from state.

    Sets
    ----
    validation_result : dict  (serialised ValidationResult)
    executed_nodes     : appended ``"validation"``

    On failure stores a pessimistic ``ValidationResult``
    (passed=False, retry_required=True) so the graph does not
    deliver a bad answer.
    """
    _record(state, "validation")
    reflection_dict = state.get("reflection_result", {})

    # Reconstruct the ReflectionResult from serialised dict
    reflection = ReflectionResult.model_validate(reflection_dict)

    validator: AnswerValidator = _get_service("validator")

    result: ValidationResult = validator.validate(reflection)

    logger.info(
        "Validation node — passed=%s  retry=%s  severity=%s",
        result.passed,
        result.retry_required,
        result.severity.value,
    )

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


def _build_tool_arguments(tool_name: str, question: str) -> dict[str, Any]:
    """Map a question to the expected arguments for *tool_name*.

    This is a simple deterministic mapping.  When an LLM-based tool
    selector is added, it will replace this function with a prompt
    that extracts arguments from the user's question.
    """
    mapping: dict[str, dict[str, Any]] = {
        "calculator": {"expression": question},
        "document-search": {"query": question, "top_k": 5},
        "current-time": {},
    }
    return mapping.get(tool_name, {"question": question})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record(state: AgentState, name: str) -> None:
    logger.info("Executing node: %s", name)


def _track(state: AgentState, name: str) -> list[str]:
    nodes: list[str] = list(state.get("executed_nodes", []))
    nodes.append(name)
    return nodes
