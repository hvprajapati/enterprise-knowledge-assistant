"""MCP Server — expose the Enterprise Knowledge Assistant via Model Context Protocol.

Components
----------
- ``MCPServer`` — stdio-based MCP server using the official MCP SDK.
- ``MCPToolAdapter`` — converts ``BaseTool`` → MCP-compatible invocations.
- ``MCPRegistry`` — manages MCP tool adapters synced with ``ToolRegistry``.
- ``MCPServerConfig`` — configuration dataclass.

Usage
-----
::

    from app.mcp import MCPServer, MCPRegistry, MCPServerConfig
    from app.tools import ToolRegistry

    tool_registry = ToolRegistry()
    # ... register tools ...

    mcp_registry = MCPRegistry(tool_registry=tool_registry)
    mcp_registry.sync()

    server = MCPServer(mcp_registry=mcp_registry)
    await server.serve()
"""

from __future__ import annotations

from app.mcp.adapters import MCPToolAdapter
from app.mcp.models import MCPServerConfig
from app.mcp.registry import MCPRegistry
from app.mcp.server import MCPServer

__all__ = [
    "MCPRegistry",
    "MCPServer",
    "MCPServerConfig",
    "MCPToolAdapter",
]
