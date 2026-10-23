"""MCP client creation and tool discovery."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import load_mcp_config, resolve_mcp_server_config

logger = logging.getLogger(__name__)


class MCPConnectionError(RuntimeError):
    """Raised when an MCP server cannot be started or its tools cannot be loaded."""


def create_mcp_client(config: dict[str, Any] | None = None) -> MultiServerMCPClient:
    """Create a client for the selected search and optional fetch servers."""
    server_config = config or load_mcp_config()
    configured_servers = server_config["servers"]
    selected_names = [server_config["search_server"]]
    fetch_server = server_config.get("fetch_server")
    if fetch_server and fetch_server not in selected_names:
        selected_names.append(fetch_server)
    servers = {
        name: resolve_mcp_server_config(name, configured_servers[name])
        for name in selected_names
    }
    for name, server in servers.items():
        command = server["command"]
        if shutil.which(command):
            continue

        if command in {"uv", "uvx"}:
            local_launcher = Path(sys.executable).parent / f"{command}.exe"
            if local_launcher.is_file():
                server["command"] = str(local_launcher)
                logger.info(
                    "Using project-environment MCP launcher for %s: %s",
                    name,
                    local_launcher,
                )
                continue

        raise MCPConnectionError(
            f"Could not find MCP command '{command}' for server '{name}'. "
            "Install the command or add it to PATH."
        )

    try:
        return MultiServerMCPClient(servers)
    except Exception as exc:
        logger.exception("Could not initialize the MCP client")
        raise MCPConnectionError(
            "Could not initialize the configured MCP servers. "
            "Check mcp_servers.json and confirm the server command is installed."
        ) from exc


async def load_mcp_tools(
    client: MultiServerMCPClient,
) -> list[BaseTool]:
    """Discover tools exposed by configured MCP servers."""
    try:
        tools = await client.get_tools()
    except Exception as exc:
        logger.exception("Could not connect to MCP server(s) or discover tools")
        raise MCPConnectionError(
            "Could not connect to the configured MCP search server. "
            "Check that the configured command is installed and the server "
            f"configuration is valid. Underlying error: {exc}"
        ) from exc

    if not tools:
        raise MCPConnectionError(
            "The MCP servers started but did not provide any tools."
        )

    logger.info("Discovered MCP tools: %s", ", ".join(tool.name for tool in tools))
    return tools


def find_search_tools(
    tools: list[BaseTool], configured_names: list[str]
) -> list[BaseTool]:
    """Find configured search tools, falling back to names and descriptions."""
    configured = {name.casefold() for name in configured_names}
    exact_matches = [tool for tool in tools if tool.name.casefold() in configured]
    if exact_matches:
        return exact_matches

    inferred = [
        tool
        for tool in tools
        if "search" in f"{tool.name} {tool.description}".casefold()
        and not any(
            excluded in f"{tool.name} {tool.description}".casefold()
            for excluded in ("image search", "news search")
        )
    ]
    return inferred


def find_fetch_tools(
    tools: list[BaseTool], configured_names: list[str]
) -> list[BaseTool]:
    """Find configured optional page-reader tools."""
    configured = {name.casefold() for name in configured_names}
    exact_matches = [tool for tool in tools if tool.name.casefold() in configured]
    if exact_matches:
        return exact_matches

    return [
        tool
        for tool in tools
        if any(
            term in f"{tool.name} {tool.description}".casefold()
            for term in ("fetch", "read page", "webpage", "page reader")
        )
    ]
