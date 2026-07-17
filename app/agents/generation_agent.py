"""Generation Agent — calls the LLM with the assembled prompt.

Wraps the existing LLM layer.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.agents.base import BaseAgent


class GenerationAgent(BaseAgent):
    """Call the configured LLM with the RAG prompt.

    Requires the ``llm`` service to be registered.
    """

    @property
    def name(self) -> str:
        return "generation"

    @property
    def description(self) -> str:
        return "Generates the final answer using the LLM with retrieved context."

    def execute(self, state: AgentState) -> dict[str, Any]:
        from app.agent.nodes import _get_service

        prompt = state.get("_prompt", "")
        llm = _get_service("llm")

        answer = llm.generate(prompt)

        executed = list(state.get("executed_nodes", [])) + ["generate"]
        history = list(state.get("execution_history", [])) + [
            {"agent": self.name, "outcome": "success", "answer_chars": len(answer)}
        ]

        return {
            "answer": answer,
            "executed_nodes": executed,
            "execution_history": history,
            "current_agent": self.name,
        }
