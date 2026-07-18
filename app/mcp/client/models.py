"""Data models for the MCP Client layer.

Describes connections to external MCP servers, their status,
and cached tool metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConnectionStatus(StrEnum):
    """Lifecycle status of an MCP server connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class RemoteToolInfo:
    """Cached metadata about a tool exposed by a remote MCP server.

    Mirrors the MCP ``Tool`` type fields that the local system
    needs for discovery and invocation.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServerConnection:
    """Represents a single external MCP server connection.

    Tracks the server identity, connection status, and a cache of
    tools discovered during the last successful ``list_tools`` call.
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    tools: list[RemoteToolInfo] = field(default_factory=list)
    last_error: str = ""

    @property
    def is_connected(self) -> bool:
        return self.status == ConnectionStatus.CONNECTED
