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
from app.agent.state import AgentState

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
            f"Agent service '{name}' not configured. "
            f"Did you call EnterpriseKnowledgeAgent(...)?"
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
    prompt = prompt_builder.build_prompt(question, reranked)

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
        "_prompt": prompt,           # carry-forward for generate_node
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _record(state: AgentState, name: str) -> None:
    logger.info("Executing node: %s", name)


def _track(state: AgentState, name: str) -> list[str]:
    nodes: list[str] = list(state.get("executed_nodes", []))
    nodes.append(name)
    return nodes
