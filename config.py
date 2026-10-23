"""Application configuration and environment loading."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# The previous Llama 3.3 model is no longer served by the configured Groq account.
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_SEARCH_ITERATIONS = 3
MAX_ALLOWED_SEARCH_ITERATIONS = 3
MCP_CONFIG_PATH = Path(
    os.getenv("MCP_CONFIG_PATH", str(PROJECT_ROOT / "mcp_servers.json"))
)
ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigurationError(ValueError):
    """Raised when an application configuration file is invalid."""


def load_mcp_config(path: Path = MCP_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the MCP server configuration JSON."""
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"MCP configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in MCP configuration: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigurationError("The MCP configuration must be a JSON object.")

    servers = config.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ConfigurationError(
            "The MCP configuration must contain a non-empty 'servers' object."
        )

    for server_name, server_config in servers.items():
        if not isinstance(server_config, dict):
            raise ConfigurationError(
                f"MCP server '{server_name}' must have an object configuration."
            )
        if server_config.get("transport") != "stdio":
            raise ConfigurationError(
                f"MCP server '{server_name}' must use stdio transport."
            )
        if not server_config.get("command") or not isinstance(
            server_config.get("args", []), list
        ):
            raise ConfigurationError(
                f"MCP server '{server_name}' needs a command and an args list."
            )
        server_env = server_config.get("env", {})
        if not isinstance(server_env, dict):
            raise ConfigurationError(
                f"MCP server '{server_name}' env must be a JSON object."
            )
        for key, value in server_env.items():
            if not isinstance(value, str):
                raise ConfigurationError(
                    f"MCP environment value '{key}' for '{server_name}' "
                    "must be a string."
                )

    search_server = config.get("search_server")
    if search_server not in servers:
        raise ConfigurationError(
            "'search_server' must name an entry in the 'servers' object."
        )
    fetch_server = config.get("fetch_server")
    if fetch_server is not None and fetch_server not in servers:
        raise ConfigurationError(
            "'fetch_server' must name an entry in the 'servers' object."
        )

    search_tool_names = config.get("search_tool_names", [])
    fetch_tool_names = config.get("fetch_tool_names", [])
    if not isinstance(search_tool_names, list) or not all(
        isinstance(name, str) for name in search_tool_names
    ):
        raise ConfigurationError("'search_tool_names' must be a list of strings.")
    if not isinstance(fetch_tool_names, list) or not all(
        isinstance(name, str) for name in fetch_tool_names
    ):
        raise ConfigurationError("'fetch_tool_names' must be a list of strings.")

    logger.info("Loaded MCP configuration for %d server(s)", len(servers))
    return config


def resolve_mcp_server_config(
    server_name: str, server_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve ${ENV_NAME} references for one selected MCP server."""
    resolved_config = dict(server_config)
    resolved_env: dict[str, str] = {}
    for key, value in server_config.get("env", {}).items():
        reference = ENV_REFERENCE.fullmatch(value)
        if reference:
            variable_name = reference.group(1)
            variable_value = os.getenv(variable_name)
            if variable_value is None:
                raise ConfigurationError(
                    f"MCP server '{server_name}' requires environment variable "
                    f"{variable_name}, but it is not set."
                )
            value = variable_value
        resolved_env[key] = value
    resolved_config["env"] = resolved_env
    return resolved_config
