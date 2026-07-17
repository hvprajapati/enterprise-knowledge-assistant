"""Tool Registry — the central catalogue of available tools.

The registry is a simple in-memory store.  It does NOT import tools
eagerly — tools are registered by the agent bootstrap and looked up
by name at execution time.

Why a registry?
    - **Discoverability.**  The planner and LLM-based selectors can
      list available tools without importing them all.
    - **Safety.**  Only registered tools can be invoked — no arbitrary
      code execution.
    - **Testability.**  Tests can register mock tools without touching
      the production tool code.
    - **Extensibility.**  Adding a new tool is two lines:
      ``registry.register(MyNewTool())``.

Thread safety
    The registry uses a plain dict with no locking.  In the current
    architecture all registrations happen at bootstrap time (before
    any requests) and reads are concurrent-safe on CPython thanks to
    the GIL.  If hot-reload is added later, wrap with an RLock.
"""

from __future__ import annotations

import logging
from typing import Any

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """In-memory catalogue of available ``BaseTool`` instances.

    Usage::

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tool = registry.lookup("calculator")
        result = tool.invoke(expression="2 + 2")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Add a tool to the registry.

        If a tool with the same name already exists it is replaced
        (last-write-wins) and a warning is logged.
        """
        name = tool.name
        if name in self._tools:
            logger.warning(
                "Tool '%s' is already registered — replacing with new instance.",
                name,
            )
        self._tools[name] = tool
        logger.info("Tool registered: %s", name)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name.  Returns ``True`` if it existed."""
        existed = self._tools.pop(name, None) is not None
        if existed:
            logger.info("Tool unregistered: %s", name)
        else:
            logger.warning("Attempted to unregister unknown tool: %s", name)
        return existed

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> BaseTool | None:
        """Return the tool with the given name, or ``None``."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return metadata for every registered tool.

        Useful for feeding tool descriptions to the LLM planner.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.schema,
            }
            for t in self._tools.values()
        ]

    def has(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)
