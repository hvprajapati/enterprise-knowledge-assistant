"""Reflection Agent — evaluates the generated answer for quality.

Wraps the existing ``ReflectionEngine``.
"""

from __future__ import annotations

from typing import Any

from app.agent.reflection import ReflectionResult
from app.agent.reflection.reflection import ReflectionEngine
from app.agent.state import AgentState
from app.agents.base import BaseAgent


class ReflectionAgent(BaseAgent):
    """Evaluate the generated answer using the Reflection Engine.

    Requires the ``reflection_engine`` service to be registered.
    """

    @property
    def name(self) -> str:
        return "reflection"

    @property
    def description(self) -> str:
        return "Evaluates answer quality, groundedness, completeness, and relevance."

    def execute(self, state: AgentState) -> dict[str, Any]:
        from app.agent.nodes import _get_service

        question = state.get("rewritten_question") or state["question"]
        answer = state.get("answer") or ""
        search_results = state.get("search_results", [])

        engine: ReflectionEngine = _get_service("reflection_engine")

        result: ReflectionResult = engine.reflect(
            question=question,
            answer=answer,
            retrieved_chunks=search_results,
        )

        executed = list(state.get("executed_nodes", [])) + ["reflection"]
        history = list(state.get("execution_history", [])) + [
            {
                "agent": self.name,
                "outcome": "success",
                "quality": result.answer_quality.value,
                "confidence": result.confidence_score,
            }
        ]

        return {
            "reflection_result": result.model_dump(),
            "executed_nodes": executed,
            "execution_history": history,
            "current_agent": self.name,
        }
