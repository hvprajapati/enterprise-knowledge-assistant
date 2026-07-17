"""Agent Registry — central catalogue of available agents.

The registry is an in-memory store.  Agents are registered at
bootstrap time and looked up by name during graph execution.

Why a registry?
    - **Discoverability.**  The supervisor can list available agents
      without importing all of them.
    - **Replaceability.**  Swap an agent implementation by registering
      a new one under the same name.
    - **Testability.**  Tests register mock agents without touching
      production code.
"""

from __future__ import annotations

import logging

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """In-memory catalogue of ``BaseAgent`` instances.

    Usage::

        registry = AgentRegistry()
        registry.register(PlannerAgent(...))
        agent = registry.lookup("planner")
        result = agent.execute(state)
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, agent: BaseAgent) -> None:
        """Add an agent to the registry.

        If an agent with the same name already exists it is replaced
        and a warning is logged.
        """
        name = agent.name
        if name in self._agents:
            logger.warning(
                "Agent '%s' is already registered — replacing.", name
            )
        self._agents[name] = agent
        logger.info("Agent registered: %s — %s", name, agent.description)

    def unregister(self, name: str) -> bool:
        """Remove an agent by name.  Returns ``True`` if it existed."""
        existed = self._agents.pop(name, None) is not None
        if existed:
            logger.info("Agent unregistered: %s", name)
        return existed

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> BaseAgent | None:
        """Return the agent with the given name, or ``None``."""
        return self._agents.get(name)

    def list_agents(self) -> list[dict[str, str]]:
        """Return name + description for every registered agent."""
        return [
            {"name": a.name, "description": a.description}
            for a in self._agents.values()
        ]

    def has(self, name: str) -> bool:
        """Check whether an agent is registered."""
        return name in self._agents

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)
