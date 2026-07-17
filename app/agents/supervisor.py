"""Supervisor Agent — orchestrates the multi-agent pipeline.

The SupervisorAgent is the **central coordinator** of the multi-agent
system.  It is invoked at the start of every graph execution to:

1. Determine the execution order of specialised agents.
2. Track which agents have completed.
3. Handle agent failures (skip or abort).
4. Record execution history for observability.

Design
------
- **Deterministic ordering.**  The execution order is fixed for now:
  planner → retrieval → generation → reflection → validation.
  In the future this can become dynamic (LLM-driven).
- **Delegation, not implementation.**  The supervisor never performs
  retrieval or generation itself — it delegates to agents via the
  ``AgentRegistry``.
- **Failure isolation.**  If one agent fails, the supervisor decides
  whether to continue (non-critical) or abort (critical).
- **Observability.**  Every delegation is logged with agent name,
  latency, and outcome.

The supervisor is itself a ``BaseAgent`` so it can be registered,
monitored, and replaced like any other agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.state import AgentState
from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

# Default execution order — can be made configurable later.
_DEFAULT_PIPELINE = [
    "planner",
    "retrieval",
    "generation",
    "reflection",
    "validation",
]


class SupervisorAgent(BaseAgent):
    """Coordinate the execution of specialised agents.

    Usage::

        registry = AgentRegistry()
        registry.register(PlannerAgent(...))
        # ... register remaining agents

        supervisor = SupervisorAgent(registry=registry, pipeline=[...])
        updated_state = supervisor.execute(state)
    """

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        pipeline: list[str] | None = None,
    ) -> None:
        self._registry = registry
        self._pipeline = pipeline or list(_DEFAULT_PIPELINE)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "supervisor"

    @property
    def description(self) -> str:
        return (
            "Coordinates specialised agents: "
            + " → ".join(self._pipeline)
        )

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Run the full agent pipeline and return merged state.

        Each agent in the pipeline is invoked via ``agent.execute()``.
        The supervisor collects partial state updates, tracks progress,
        and handles failures.

        Returns
        -------
        dict[str, Any]
            Merged partial state from all agents that executed.
        """
        t_start = time.monotonic()
        logger.info(
            "Supervisor — starting pipeline: %s",
            " → ".join(self._pipeline),
        )

        merged: dict[str, Any] = {
            "completed_agents": [],
            "execution_history": [],
            "current_agent": self.name,
        }

        for agent_name in self._pipeline:
            agent = self._registry.lookup(agent_name)
            if agent is None:
                logger.warning(
                    "Supervisor — agent '%s' not registered, skipping.",
                    agent_name,
                )
                merged["execution_history"].append({
                    "agent": agent_name,
                    "outcome": "skipped",
                    "reason": "not registered",
                })
                continue

            # Delegate
            try:
                combined: Any = {**state, **merged}
                result = agent.execute(combined)

                # Accumulate history — agents may return per-agent entries
                agent_history = result.pop("execution_history", [])
                merged.update(result)
                if agent_history:
                    merged["execution_history"].extend(agent_history)
                else:
                    merged["execution_history"].append(
                        {"agent": agent_name, "outcome": "success"}
                    )
                merged["completed_agents"].append(agent_name)

            except Exception as exc:
                logger.exception(
                    "Supervisor — agent '%s' raised unhandled exception.",
                    agent_name,
                )
                merged["execution_history"].append({
                    "agent": agent_name,
                    "outcome": "error",
                    "error": str(exc),
                })
                # For critical agents, break the pipeline
                if agent_name in {"planner", "retrieval", "generation"}:
                    logger.error(
                        "Supervisor — critical agent '%s' failed, aborting pipeline.",
                        agent_name,
                    )
                    merged["error"] = str(exc)
                    break

        elapsed = (time.monotonic() - t_start) * 1000
        logger.info(
            "Supervisor — pipeline complete  agents=%d/%d  latency=%.0fms",
            len(merged["completed_agents"]),
            len(self._pipeline),
            elapsed,
        )

        return merged
