"""PRSM MCP Server — Expose skill tools as MCP-compatible endpoints.

Reads all installed skill packages from the SkillRegistry and exposes
their tools via FastMCP so AI agents (Hermes, OpenClaw, Claude Desktop,
etc.) can discover and invoke PRSM capabilities.

Usage:
    prsm mcp start              # Start the MCP server on port 9100
    prsm mcp config-snippet     # Print config snippet for AI clients
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MCP_PORT = 9100
DEFAULT_MCP_HOST = "localhost"

# ── FastMCP availability check ───────────────────────────────────────

_FASTMCP_AVAILABLE = False
try:
    import fastmcp  # noqa: F401
    _FASTMCP_AVAILABLE = True
except ImportError:
    pass


def is_fastmcp_available() -> bool:
    """Check whether the fastmcp package is installed."""
    return _FASTMCP_AVAILABLE


# ── Config snippet (works without fastmcp) ──────────────────────────

CONFIG_SNIPPET = """\
mcp_servers:
  prsm:
    transport: streamable-http
    url: http://{host}:{port}/mcp
"""


def get_config_snippet(host: str = DEFAULT_MCP_HOST, port: int = DEFAULT_MCP_PORT) -> str:
    """Return a YAML snippet for Hermes/OpenClaw MCP client configuration."""
    return CONFIG_SNIPPET.format(host=host, port=port)


# ── Mock tool handlers ──────────────────────────────────────────────

def _make_mock_handler(tool_def: Dict[str, Any]):
    """Create a mock handler function for a skill tool.

    Since the actual PRSM network isn't running, returns structured
    JSON that demonstrates the schema works.

    FastMCP requires explicit parameter signatures (no **kwargs) and uses
    get_type_hints() for Pydantic schema generation. We use exec() to create
    a real function with the correct signature and annotations.
    """
    from typing import Optional

    tool_name = tool_def["name"]
    skill_name = tool_def["skill"]
    input_schema = tool_def["inputSchema"]
    properties = input_schema.get("properties", {})
    required_params = set(input_schema.get("required", []))

    # Map JSON schema types to Python type names
    _type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }

    # Build parameter strings: required first, then optional
    req_parts = []
    opt_parts = []
    for param_name, param_def in properties.items():
        py_type = _type_map.get(param_def.get("type", "string"), "str")
        safe_name = param_name.replace("-", "_")
        if param_name in required_params:
            req_parts.append(f"{safe_name}: {py_type}")
        else:
            opt_parts.append(f"{safe_name}: {py_type} = None")

    param_str = ", ".join(req_parts + opt_parts)
    func_name = tool_name.replace("-", "_")

    # Capture closure variables
    _tool_name = tool_name
    _skill_name = skill_name
    _properties = properties
    _required_params = required_params

    # Build the function source
    code = f'''
async def {func_name}({param_str}) -> str:
    """Mock handler for {_tool_name}."""
    import json as _json
    _locals = locals()
    received = {{k: v for k, v in _locals.items() if v is not None and not k.startswith("_")}}
    result = {{
        "tool": _tool_name,
        "skill": _skill_name,
        "status": "mock_response",
        "message": f"Tool '{_tool_name}' from skill '{_skill_name}' invoked successfully (mock mode — PRSM network not running)",
        "input_received": received,
        "schema": {{
            "properties": list(_properties.keys()),
            "required": list(_required_params),
        }},
        "mock_data": _generate_mock_data(_tool_name, received),
    }}
    return _json.dumps(result, indent=2, default=str)
'''

    # Execute to create the function with proper signature + annotations
    namespace = {
        "_tool_name": _tool_name,
        "_skill_name": _skill_name,
        "_properties": _properties,
        "_required_params": _required_params,
        "_generate_mock_data": _generate_mock_data,
    }
    exec(code, namespace)
    handler = namespace[func_name]
    handler.__doc__ = tool_def.get("description", f"PRSM tool: {tool_name}")

    return handler


def _generate_mock_data(tool_name: str, kwargs: Dict[str, Any]) -> Any:
    """Generate plausible mock response data based on the tool name."""
    # Provide recognizable mock data for common tool patterns
    name_lower = tool_name.lower()

    if "search" in name_lower or "find" in name_lower or "discover" in name_lower:
        return {
            "results": [
                {"id": "mock-001", "name": "Example Result 1", "score": 0.95},
                {"id": "mock-002", "name": "Example Result 2", "score": 0.87},
            ],
            "total": 2,
        }
    elif "submit" in name_lower or "create" in name_lower or "upload" in name_lower:
        return {
            "id": "mock-new-001",
            "created": True,
            "timestamp": "2026-04-01T12:00:00Z",
        }
    elif "get" in name_lower or "fetch" in name_lower or "retrieve" in name_lower:
        return {
            "id": "mock-001",
            "data": {"key": "value", "status": "available"},
        }
    elif "list" in name_lower:
        return {
            "items": [
                {"id": "item-001", "name": "Item A"},
                {"id": "item-002", "name": "Item B"},
            ],
            "count": 2,
        }
    elif "run" in name_lower or "execute" in name_lower or "invoke" in name_lower:
        return {
            "execution_id": "exec-mock-001",
            "status": "completed",
            "result": "Mock execution completed successfully",
        }
    else:
        return {"acknowledged": True}


# ── Server creation ─────────────────────────────────────────────────

def create_mcp_server(
    host: str = DEFAULT_MCP_HOST,
    port: int = DEFAULT_MCP_PORT,
) -> Any:
    """Create and configure a FastMCP server with all PRSM skill tools.

    Returns the FastMCP server instance (not yet started).

    Raises:
        ImportError: If fastmcp is not installed.
    """
    if not _FASTMCP_AVAILABLE:
        raise ImportError(
            "fastmcp is required to run the PRSM MCP server.\n"
            "Install it with: pip install fastmcp"
        )

    from fastmcp import FastMCP

    # Load skill registry
    from prsm.skills.registry import SkillRegistry

    registry = SkillRegistry(load_builtins=True)
    all_tools = registry.get_all_tools()

    logger.info(
        f"Creating MCP server with {registry.skill_count} skill(s), "
        f"{registry.tool_count} tool(s)"
    )

    # Create the FastMCP server
    mcp = FastMCP(
        name="PRSM",
        instructions=(
            "PRSM MCP Server — Provides access to PRSM's decentralized AI "
            "infrastructure tools. Includes dataset management, model governance, "
            "compute orchestration, and network operations."
        ),
    )

    # Register each skill tool as an MCP tool
    for tool_def in all_tools:
        tool_name = tool_def["name"]
        description = tool_def.get("description", f"PRSM tool: {tool_name}")
        handler = _make_mock_handler(tool_def)

        # Use the FastMCP decorator pattern programmatically
        mcp.tool(name=tool_name, description=description)(handler)
        logger.debug(f"Registered MCP tool: {tool_name}")

    logger.info(f"MCP server configured — {len(all_tools)} tool(s) registered")
    return mcp


def start_mcp_server(
    host: str = DEFAULT_MCP_HOST,
    port: int = DEFAULT_MCP_PORT,
) -> None:
    """Create and start the PRSM MCP server (blocking).

    Raises:
        ImportError: If fastmcp is not installed.
    """
    mcp = create_mcp_server(host=host, port=port)
    mcp.run(transport="streamable-http", host=host, port=port)
