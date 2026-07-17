"""Abstract base class for all agents in the multi-agent system.

Every agent — Planner, Retrieval, Generation, Reflection, Validation,
and the Supervisor — inherits from ``BaseAgent``.  This ensures a
uniform interface that the graph and supervisor can depend on.

Design principles
-----------------
1. **Single responsibility.**  Each agent does exactly one thing.
   Planner plans, Retrieval retrieves, Generation generates, etc.
2. **Uniform contract.**  ``execute(state)`` is the only entry point.
   Every agent reads from ``AgentState`` and returns a partial state
   dict to merge.
3. **Self-describing.**  ``name`` and ``description`` let the
   supervisor and logging layer know what each agent does.
4. **Swappable.**  Agents can be replaced without touching the graph
   — register a new implementation under the same name.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from app.agent.state import AgentState


class BaseAgent(ABC):
    """Abstract agent that every concrete agent must implement.

    Subclass and provide:

    - ``name`` — unique identifier (e.g. ``"planner"``)
    - ``description`` — one-sentence summary for the supervisor
    - ``execute(state)`` — run the agent and return partial state
    """

    # ------------------------------------------------------------------
    # subclasses MUST define these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier (kebab-case, e.g. ``"planner"``)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for supervisor delegation."""
        ...

    # ------------------------------------------------------------------
    # subclasses MUST implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, state: AgentState) -> dict[str, Any]:
        """Run the agent and return a partial state update.

        Parameters
        ----------
        state:
            The current ``AgentState`` — read-only in practice, though
            not enforced by the type system.

        Returns
        -------
        dict[str, Any]
            Partial state dict that LangGraph merges into ``AgentState``.
        """
        ...

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def execute_with_logging(self, state: AgentState) -> dict[str, Any]:
        """Execute the agent with timing and structured logging.

        Called by the graph node wrappers so individual agents don't
        need to duplicate logging boilerplate.
        """
        import logging

        logger = logging.getLogger(__name__)
        t_start = time.monotonic()

        logger.info("Agent [%s] — started", self.name)

        try:
            result = self.execute(state)
            elapsed = (time.monotonic() - t_start) * 1000

            logger.info(
                "Agent [%s] — completed  latency=%.0fms  keys=%s",
                self.name,
                elapsed,
                list(result.keys()),
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - t_start) * 1000
            logger.exception(
                "Agent [%s] — FAILED  latency=%.0fms  error=%s",
                self.name,
                elapsed,
                exc,
            )
            return {"error": str(exc)}
