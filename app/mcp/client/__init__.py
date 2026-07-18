"""MCP Client — connect to external MCP servers and consume their tools.

Components
----------
- ``MCPClient`` — connects to a single external MCP server via stdio.
- ``MCPConnectionManager`` — manages multiple MCP server connections,
  health monitoring, and reconnection.
- ``RemoteToolAdapter`` — wraps remote MCP tools as local ``BaseTool``
  instances, so the app doesn't distinguish local vs remote tools.
- ``MCPServerConnection`` — connection metadata and tool cache.
- ``RemoteToolInfo`` — cached tool metadata from a remote server.
"""

from __future__ import annotations

from app.mcp.client.adapters import RemoteToolAdapter
from app.mcp.client.client import MCPClient
from app.mcp.client.manager import MCPConnectionManager
from app.mcp.client.models import (
    ConnectionStatus,
    MCPServerConnection,
    RemoteToolInfo,
)

__all__ = [
    "ConnectionStatus",
    "MCPClient",
    "MCPConnectionManager",
    "MCPServerConnection",
    "RemoteToolAdapter",
    "RemoteToolInfo",
]
