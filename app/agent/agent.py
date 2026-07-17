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
from app.agent.reflection.reflection import ReflectionEngine
from app.agent.retry.retry_engine import RetryEngine
from app.agent.state import (
    AgentState as _AgentState,  # noqa: F401 — used in initial state construction
)
from app.agent.validation.validator import AnswerValidator, ValidationThresholds
from app.config.settings import settings
from app.llm.base import BaseLLM
from app.query.service import QueryService
from app.retrieval.query_rewriter import QueryRewriter
from app.tools import ToolExecutor, ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.current_time import CurrentTimeTool
from app.tools.document_search import DocumentSearchTool

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
        _services["reflection_engine"] = ReflectionEngine(llm=llm)
        _services["validator"] = AnswerValidator(
            thresholds=ValidationThresholds(
                min_confidence_score=settings.validation_min_confidence_score,
                require_grounded=settings.validation_require_grounded,
                require_completeness=settings.validation_require_completeness,
                require_relevance=settings.validation_require_relevance,
            )
        )
        _services["retry_engine"] = RetryEngine(
            max_retries=settings.max_agent_retries,
        )

        # -- tool calling framework ---------------------------------------
        doc_search = DocumentSearchTool()
        doc_search.configure(query_service)

        tool_registry = ToolRegistry()
        tool_registry.register(CalculatorTool())
        tool_registry.register(CurrentTimeTool())
        tool_registry.register(doc_search)

        _services["tool_registry"] = tool_registry
        _services["tool_executor"] = ToolExecutor(registry=tool_registry)

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
            "requires_rewrite": False,  # will be set by planner_node
            "execution_plan": {},  # will be set by planner_node
            "tool_decision": {},  # will be set by tool_node
            "tool_result": {},  # will be set by tool_node
            "reflection_result": {},  # will be set by reflection_node
            "validation_result": {},  # will be set by validation_node
            "retry_decision": {},  # will be set by retry_node
            "retry_count": 0,  # tracks number of retries
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
