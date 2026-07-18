"""Unit tests for the MCP Client framework.

Covers:
- MCPServerConnection model
- RemoteToolInfo model
- RemoteToolAdapter wraps remote tool as BaseTool
- MCPConnectionManager: add/remove server, connect lifecycle
- ConnectionStatus enum
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp.client import (
    ConnectionStatus,
    MCPConnectionManager,
    RemoteToolAdapter,
)
from app.mcp.client.models import MCPServerConnection, RemoteToolInfo
from app.tools.base import BaseTool

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_connection(name: str = "test-server") -> MCPServerConnection:
    return MCPServerConnection(
        name=name,
        command="python",
        args=["-m", "test_server"],
    )


def _make_tool_info(name: str = "greet") -> RemoteToolInfo:
    return RemoteToolInfo(
        name=name,
        description="Greets the user.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        server_name="test-server",
    )


# ---------------------------------------------------------------------------
# ConnectionStatus tests
# ---------------------------------------------------------------------------


class TestConnectionStatus:
    def test_values(self) -> None:
        assert set(ConnectionStatus) == {
            ConnectionStatus.DISCONNECTED,
            ConnectionStatus.CONNECTING,
            ConnectionStatus.CONNECTED,
            ConnectionStatus.ERROR,
        }

    def test_str_comparison(self) -> None:
        assert ConnectionStatus("connected") == ConnectionStatus.CONNECTED


# ---------------------------------------------------------------------------
# MCPServerConnection tests
# ---------------------------------------------------------------------------


class TestMCPServerConnection:
    def test_defaults(self) -> None:
        conn = MCPServerConnection(name="test", command="echo")
        assert conn.name == "test"
        assert conn.status == ConnectionStatus.DISCONNECTED
        assert conn.args == []
        assert conn.tools == []
        assert conn.is_connected is False

    def test_with_args(self) -> None:
        conn = MCPServerConnection(name="srv", command="python", args=["-c", "print(1)"])
        assert conn.args == ["-c", "print(1)"]

    def test_connected_status(self) -> None:
        conn = _make_connection()
        conn.status = ConnectionStatus.CONNECTED
        assert conn.is_connected is True


# ---------------------------------------------------------------------------
# RemoteToolInfo tests
# ---------------------------------------------------------------------------


class TestRemoteToolInfo:
    def test_fields(self) -> None:
        info = _make_tool_info()
        assert info.name == "greet"
        assert info.description == "Greets the user."
        assert "name" in info.input_schema.get("required", [])
        assert info.server_name == "test-server"


# ---------------------------------------------------------------------------
# RemoteToolAdapter tests
# ---------------------------------------------------------------------------


class TestRemoteToolAdapter:
    def test_is_base_tool(self) -> None:
        info = _make_tool_info()
        mock_client = MagicMock()
        mock_client.server_name = "test-server"
        mock_client.call_tool = AsyncMock(return_value={"result": "Hello"})

        adapter = RemoteToolAdapter(tool_info=info, mcp_client=mock_client)
        assert isinstance(adapter, BaseTool)

    def test_name_description_schema(self) -> None:
        info = _make_tool_info()
        mock_client = MagicMock()
        mock_client.server_name = "test"

        adapter = RemoteToolAdapter(tool_info=info, mcp_client=mock_client)
        assert adapter.name == "greet"
        assert adapter.description == "Greets the user."
        assert "name" in adapter.schema.get("required", [])

    def test_invoke_delegates_to_client(self) -> None:
        """invoke() should call the MCP client's call_tool and return the result."""
        info = _make_tool_info()
        mock_client = MagicMock()
        mock_client.server_name = "test-server"
        mock_client.call_tool = MagicMock()

        # Patch asyncio.run to return synchronously
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = {"result": "Hello, World"}
            adapter = RemoteToolAdapter(tool_info=info, mcp_client=mock_client)
            result = adapter.invoke(name="World")

        assert result == {"result": "Hello, World"}

    def test_invoke_propagates_errors(self) -> None:
        info = _make_tool_info()
        mock_client = MagicMock()
        mock_client.server_name = "test-server"

        with patch("asyncio.run", side_effect=RuntimeError("Remote down")):
            adapter = RemoteToolAdapter(tool_info=info, mcp_client=mock_client)
            with pytest.raises(RuntimeError, match="Remote tool 'greet' failed"):
                adapter.invoke(name="test")


# ---------------------------------------------------------------------------
# MCPConnectionManager tests
# ---------------------------------------------------------------------------


class TestMCPConnectionManager:
    def test_add_server(self) -> None:
        manager = MCPConnectionManager()
        manager.add_server(_make_connection("srv1"))
        assert manager.total_count == 1

    def test_add_multiple_servers(self) -> None:
        manager = MCPConnectionManager()
        manager.add_server(_make_connection("srv1"))
        manager.add_server(_make_connection("srv2"))
        assert manager.total_count == 2

    def test_remove_server(self) -> None:
        manager = MCPConnectionManager()
        manager.add_server(_make_connection("srv1"))
        removed = manager.remove_server("srv1")
        assert removed is True
        assert manager.total_count == 0

    def test_remove_unknown_server(self) -> None:
        manager = MCPConnectionManager()
        assert manager.remove_server("nonexistent") is False

    def test_list_servers(self) -> None:
        manager = MCPConnectionManager()
        manager.add_server(_make_connection("srv1"))
        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "srv1"
        assert servers[0]["status"] == "disconnected"

    def test_connected_count(self) -> None:
        manager = MCPConnectionManager()
        manager.add_server(_make_connection("srv1"))
        assert manager.connected_count == 0

    def test_configure(self) -> None:
        manager = MCPConnectionManager()
        manager.configure(auto_reconnect=True, heartbeat_interval=15.0)
        assert manager._auto_reconnect is True
        assert manager._heartbeat_interval == 15.0
