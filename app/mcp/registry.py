"""MCP Registry — manages MCP tool adapters mapped to BaseTool instances.

Thin wrapper around the existing ``ToolRegistry``.  When tools are
registered or unregistered, corresponding ``MCPToolAdapter`` instances
are created or removed automatically.

Why a separate MCP registry?
    - **Isolation.**  The MCP layer works with ``MCPToolAdapter``,
      not raw ``BaseTool`` instances.
    - **Auto-sync.**  Register a tool → an MCP adapter is created
      automatically.  Unregister → adapter removed.
    - **Discovery.**  ``list_tools()`` returns MCP ``Tool`` objects
      directly, ready for the ``list_tools`` handler.
"""

from __future__ import annotations

import logging
from typing import Any

from app.mcp.adapters import MCPToolAdapter
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPRegistry:
    """Manages MCP tool adapters backed by a ``ToolRegistry``.

    Usage::

        tool_registry = ToolRegistry()
        tool_registry.register(CalculatorTool())
        mcp_registry = MCPRegistry(tool_registry=tool_registry)
        mcp_registry.sync()  # auto-create adapters for all tools
    """

    def __init__(self, *, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._adapters: dict[str, MCPToolAdapter] = {}

    # ------------------------------------------------------------------
    # sync
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Create MCP adapters for every tool currently in the ToolRegistry."""
        tools = self._tool_registry.list_tools()
        for tool_info in tools:
            name = str(tool_info["name"])
            if name not in self._adapters:
                tool = self._tool_registry.lookup(name)
                if tool is not None:
                    self._adapters[name] = MCPToolAdapter(tool)
        logger.info(
            "MCPRegistry synced — %d adapters ready", len(self._adapters)
        )

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in both registries and create an MCP adapter."""
        self._tool_registry.register(tool)
        self._adapters[tool.name] = MCPToolAdapter(tool)
        logger.info("MCP tool registered: %s", tool.name)

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from both registries."""
        self._adapters.pop(name, None)
        return self._tool_registry.unregister(name)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> MCPToolAdapter | None:
        """Return the MCP adapter for *name*."""
        return self._adapters.get(name)

    def list_tools(self) -> list[Any]:
        """Return MCP ``Tool`` objects for the ``list_tools`` handler."""
        return [a.to_mcp_tool() for a in self._adapters.values()]

    def has(self, name: str) -> bool:
        return name in self._adapters

    @property
    def tool_count(self) -> int:
        return len(self._adapters)
