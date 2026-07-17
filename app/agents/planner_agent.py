"""Planner Agent — analyses the question and creates an execution plan.

Wraps the existing ``Planner`` from ``app.agent.planner``.
"""

from __future__ import annotations

from typing import Any

from app.agent.planner.planner import Planner
from app.agent.state import AgentState
from app.agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    """Analyse the user question and produce an ``ExecutionPlan``.

    Delegates to the existing deterministic ``Planner``.
    """

    def __init__(self, planner: Planner | None = None) -> None:
        self._planner = planner or Planner()

    @property
    def name(self) -> str:
        return "planner"

    @property
    def description(self) -> str:
        return "Analyses the question and creates an execution plan with retrieval strategy."

    def execute(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        plan = self._planner.create_plan(question)

        executed = list(state.get("executed_nodes", [])) + ["planner"]
        history = list(state.get("execution_history", [])) + [
            {"agent": self.name, "outcome": "success", "plan_type": plan.question_type.value}
        ]

        return {
            "execution_plan": plan.model_dump(),
            "requires_rewrite": plan.requires_rewrite,
            "executed_nodes": executed,
            "execution_history": history,
            "current_agent": self.name,
        }
