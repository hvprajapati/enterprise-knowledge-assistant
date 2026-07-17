"""Enterprise Knowledge Agent — high-level LangGraph wrapper.

Usage::

    from app.agent.agent import EnterpriseKnowledgeAgent

    agent = EnterpriseKnowledgeAgent(
        query_service=qs,
        rewriter=rewriter,
        llm=llm,
        ...
    )

    result = agent.run("What is FAISS?")
    print(result["answer"])
    print(result["executed_nodes"])
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.graph import build_graph
from app.agent.nodes import _services
from app.agent.planner.planner import Planner
from app.agent.state import (
    AgentState as _AgentState,  # noqa: F401 — used in initial state construction
)
from app.llm.base import BaseLLM
from app.query.service import QueryService
from app.retrieval.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class EnterpriseKnowledgeAgent:
    """Thin wrapper around a compiled LangGraph ``StateGraph``.

    All pipeline dependencies are injected via the constructor and
    stored in a module-level service registry so graph nodes can
    access them without passing parameters through the state.
    """

    def __init__(
        self,
        *,
        query_service: QueryService,
        rewriter: QueryRewriter | None = None,
        llm: BaseLLM,
    ) -> None:
        # Populate the shared service registry
        _services["query_service"] = query_service
        _services["rewriter"] = rewriter
        _services["llm"] = llm
        _services["planner"] = Planner()
        _services["retriever"] = query_service._retriever
        _services["reranker"] = query_service._reranker
        _services["embedding_service"] = query_service._embed
        _services["prompt_builder"] = query_service._prompt_builder
        _services["orchestrator"] = query_service._orchestrator

        self._graph = build_graph().compile()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict[str, Any]:
        """Execute the graph and return the final ``AgentState``.

        Parameters
        ----------
        question:
            Raw user question.

        Returns
        -------
        dict
            Contains ``answer``, ``executed_nodes``, and optionally
            ``search_results`` and ``error``.
        """
        initial: _AgentState = {
            "question": question,
            "rewritten_question": None,
            "search_results": [],
            "answer": None,
            "error": None,
            "executed_nodes": [],
            "requires_rewrite": False,     # will be set by planner_node
            "execution_plan": {},          # will be set by planner_node
        }

        logger.info(
            "Agent invoked — question=%d chars",
            len(question),
        )

        try:
            result: dict[str, Any] = self._graph.invoke(initial)
        except Exception as exc:
            logger.exception("Agent execution failed")
            return {
                **initial,
                "answer": "I'm sorry, something went wrong while processing your question.",
                "error": str(exc),
                "executed_nodes": initial["executed_nodes"],
            }

        logger.info(
            "Agent finished — executed: %s",
            " -> ".join(result.get("executed_nodes", [])),
        )
        return result
