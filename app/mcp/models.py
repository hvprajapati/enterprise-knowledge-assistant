"""Data models for the MCP Server layer.

Thin wrappers that adapt the application's ``ToolResult`` into
MCP-compliant responses without duplicating business logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MCPToolInfo:
    """Metadata about a tool that the MCP server exposes to clients.

    This is the bridge between ``BaseTool`` (app domain) and the
    MCP ``Tool`` type.  It holds pre-computed metadata so the
    ``list_tools`` handler doesn't need to call every tool at
    discovery time.
    """

    name: str
    description: str
    schema: dict  # JSON Schema dict from BaseTool.schema


@dataclass
class MCPServerConfig:
    """Immutable configuration for the MCP server."""

    name: str = "Enterprise Knowledge Assistant"
    version: str = "0.1.0"
    enable: bool = True
