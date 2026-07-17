"""Retrieval Agent — searches FAISS and builds the RAG prompt.

Wraps the existing retrieval pipeline: embed → retrieve → rerank →
prompt build.  Optionally includes tool results in the prompt.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.agents.base import BaseAgent


class RetrievalAgent(BaseAgent):
    """Search the vector store and assemble a grounded RAG prompt.

    Requires the following services to be injected (via the module-level
    ``_services`` dict in ``app.agent.nodes``):

    - ``retriever``
    - ``reranker``
    - ``embedding_service``
    - ``orchestrator``
    - ``prompt_builder``
    """

    @property
    def name(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "Searches the document store and builds a RAG prompt with citations."

    def execute(self, state: AgentState) -> dict[str, Any]:
        # Import here to avoid circular dependency at module level
        from app.agent.nodes import _get_service

        question = state.get("rewritten_question") or state["question"]

        retriever = _get_service("retriever")
        reranker = _get_service("reranker")
        embed = _get_service("embedding_service")
        orchestrator = _get_service("orchestrator")
        prompt_builder = _get_service("prompt_builder")

        # Plan-driven retrieval
        plan = orchestrator.plan(question)

        embedding = embed.embed_query(question)
        candidates = retriever.retrieve(embedding, top_k=plan.vector_top_k)
        reranked = reranker.rerank(question, candidates, top_k=5)

        # Include tool results if available
        tool_results: list[dict[str, object]] = state.get("tool_results", [])
        prompt = prompt_builder.build_prompt(
            question,
            reranked,
            tool_results=tool_results if tool_results else None,
        )

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

        executed = list(state.get("executed_nodes", [])) + ["retrieve"]
        history = list(state.get("execution_history", [])) + [
            {
                "agent": self.name,
                "outcome": "success",
                "candidates": len(candidates),
                "reranked": len(reranked),
            }
        ]

        return {
            "search_results": serialised,
            "_prompt": prompt,
            "executed_nodes": executed,
            "execution_history": history,
            "current_agent": self.name,
        }
