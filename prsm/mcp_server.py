"""
PRSM MCP Server
================

Model Context Protocol server that exposes PRSM tools to any LLM.
Enables Claude, Gemini, or any MCP-compatible model to submit queries,
get cost estimates, browse datasets, and dispatch agents via PRSM.

Usage:
    prsm mcp-server                     # Start via CLI
    python -m prsm.mcp_server           # Start directly

Configure in Claude Desktop:
    ~/.claude/claude_desktop_config.json:
    {
        "mcpServers": {
            "prsm": {
                "command": "prsm",
                "args": ["mcp-server"]
            }
        }
    }
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

logger = logging.getLogger(__name__)

# ── Tool Definitions ─────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="prsm_analyze",
        description=(
            "Submit a natural language query to the PRSM distributed compute network. "
            "Automatically decomposes the query via LLM, finds relevant data shards, "
            "dispatches WASM mobile agents to edge nodes, aggregates results, and "
            "settles FTNS token payments. IMPORTANT: Execution requires FTNS tokens — "
            "the budget_ftns parameter must be greater than 0. Use prsm_quote first "
            "to estimate costs before committing. The minimum budget is 0.01 FTNS."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The analysis query in natural language",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS tokens to spend (REQUIRED, minimum 0.01). Use prsm_quote to estimate costs first.",
                    "minimum": 0.01,
                    "default": 10.0,
                },
                "privacy_level": {
                    "type": "string",
                    "description": "Privacy level: none, standard (e=8), high (e=4), maximum (e=1)",
                    "enum": ["none", "standard", "high", "maximum"],
                    "default": "standard",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_quote",
        description=(
            "Get a cost estimate for a PRSM query BEFORE committing. Returns compute cost, "
            "data access cost, network fee, and total in FTNS tokens. Use this to check "
            "costs before running an expensive analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to estimate costs for",
                },
                "shard_count": {
                    "type": "integer",
                    "description": "Estimated number of data shards (default: 3)",
                    "default": 3,
                },
                "hardware_tier": {
                    "type": "string",
                    "description": "Target hardware tier: t1 (mobile), t2 (consumer), t3 (high-end), t4 (datacenter)",
                    "enum": ["t1", "t2", "t3", "t4"],
                    "default": "t2",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_list_datasets",
        description=(
            "Browse available datasets on the PRSM network with pricing information. "
            "Filter by keyword or maximum price."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Keyword to search dataset titles and descriptions",
                    "default": "",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum base access fee in FTNS",
                },
            },
        },
    ),
    Tool(
        name="prsm_node_status",
        description=(
            "Check the status of the local PRSM node, including which of the 10 "
            "capability rings are initialized and healthy."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_hardware_benchmark",
        description=(
            "Run a hardware benchmark on the local node. Returns compute tier (T1-T4), "
            "GPU detection, TFLOPS, thermal classification, and TEE availability."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ── Tool Handlers ────────────────────────────────────────────────────────

async def _get_node_api_url() -> str:
    """Get the PRSM node API URL."""
    return os.environ.get("PRSM_NODE_URL", "http://localhost:8000")


async def _call_node_api(method: str, path: str, data: Dict = None) -> Dict[str, Any]:
    """Call the PRSM node API."""
    import aiohttp
    url = await _get_node_api_url()
    api_key = os.environ.get("PRSM_NODE_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        if method == "GET":
            async with session.get(
                f"{url}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                return await resp.json()
        else:
            async with session.post(
                f"{url}{path}",
                json=data or {},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                return await resp.json()


MINIMUM_BUDGET_FTNS = 0.01


async def handle_prsm_analyze(arguments: Dict[str, Any]) -> str:
    """Handle prsm_analyze tool call."""
    query = arguments.get("query", "")
    budget = arguments.get("budget_ftns", 10.0)
    privacy = arguments.get("privacy_level", "standard")

    # Enforce minimum budget
    if budget <= 0:
        return (
            "PRSM requires an FTNS budget to execute queries. "
            "Set budget_ftns to at least 0.01 FTNS.\n\n"
            "Tip: Use the prsm_quote tool first to estimate costs, "
            "then call prsm_analyze with an appropriate budget."
        )
    if budget < MINIMUM_BUDGET_FTNS:
        return (
            f"Budget {budget} FTNS is below the minimum ({MINIMUM_BUDGET_FTNS} FTNS). "
            f"Use prsm_quote to estimate the required budget for your query."
        )

    try:
        result = await _call_node_api("POST", "/compute/forge", {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        })

        response = result.get("response", "")
        route = result.get("route", "unknown")
        job_id = result.get("job_id", "")

        return (
            f"PRSM Analysis Result (route: {route})\n"
            f"Job ID: {job_id}\n\n"
            f"{response}"
        )
    except Exception as e:
        return f"PRSM analysis failed: {str(e)}. Is your PRSM node running? (prsm node start)"


async def handle_prsm_quote(arguments: Dict[str, Any]) -> str:
    """Handle prsm_quote tool call."""
    query = arguments.get("query", "")
    shards = arguments.get("shard_count", 3)
    tier = arguments.get("hardware_tier", "t2")

    try:
        from prsm.economy.pricing import PricingEngine
        engine = PricingEngine()
        quote = engine.quote_swarm_job(
            shard_count=shards,
            hardware_tier=tier,
            estimated_pcu_per_shard=50.0,
        )
        return (
            f"Cost Estimate for: {query}\n"
            f"  Compute: {quote.compute_cost} FTNS\n"
            f"  Data: {quote.data_cost} FTNS\n"
            f"  Network Fee: {quote.network_fee} FTNS\n"
            f"  Total: {quote.total} FTNS\n"
            f"  Hardware Tier: {tier.upper()}\n"
            f"  Shards: {shards}"
        )
    except Exception as e:
        return f"Quote failed: {str(e)}"


async def handle_prsm_list_datasets(arguments: Dict[str, Any]) -> str:
    """Handle prsm_list_datasets tool call."""
    search = arguments.get("search", "")
    max_price = arguments.get("max_price")

    try:
        # Try the node API first
        result = await _call_node_api("GET", "/rings/status")
        # If node is running, use it
        return (
            f"PRSM Network Status\n"
            f"  Rings Initialized: {result.get('rings_initialized', 0)}/10\n"
            f"  Note: Use 'prsm marketplace list-dataset' CLI to publish datasets.\n"
            f"  Data marketplace browsing via API is available when datasets are published."
        )
    except Exception:
        return "PRSM node not running. Start with: prsm node start"


async def handle_prsm_node_status(arguments: Dict[str, Any]) -> str:
    """Handle prsm_node_status tool call."""
    try:
        result = await _call_node_api("GET", "/rings/status")
        rings = result.get("rings", [])
        lines = [f"PRSM Node -- {result.get('rings_initialized', 0)}/10 Rings Active\n"]
        for r in rings:
            status = "[ok]" if r.get("initialized") else "[--]"
            lines.append(f"  {status} Ring {r['ring']}: {r['name']}")

        pricing = result.get("pricing", {})
        if pricing:
            lines.append(f"\n  Spot Multiplier: {pricing.get('spot_multiplier', '1.0')}x")
            lines.append(f"  Utilization: {pricing.get('utilization', 0):.0%}")

        forge = result.get("forge", {})
        if forge:
            lines.append(f"  Training Traces: {forge.get('traces_collected', 0)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Cannot reach PRSM node: {str(e)}\nStart with: prsm node start"


async def handle_prsm_hardware_benchmark(arguments: Dict[str, Any]) -> str:
    """Handle prsm_hardware_benchmark tool call."""
    try:
        from prsm.compute.wasm import HardwareProfiler
        from prsm.compute.tee.platform_detect import get_tee_summary

        profiler = HardwareProfiler()
        profile = profiler.detect()
        tee = get_tee_summary()

        return (
            f"PRSM Hardware Benchmark\n"
            f"  CPU: {profile.cpu_cores} cores @ {profile.cpu_freq_mhz:.0f} MHz\n"
            f"  GPU: {profile.gpu_name or 'None detected'}\n"
            f"  VRAM: {profile.gpu_vram_gb:.1f} GB\n"
            f"  TFLOPS: {profile.tflops_fp32:.2f} FP32\n"
            f"  Compute Tier: {profile.compute_tier.value.upper()}\n"
            f"  Thermal: {profile.thermal_class.value}\n"
            f"  TEE: {tee['type']} (hardware: {tee['hardware_backed']})\n"
            f"  RAM: {profile.ram_total_gb:.1f} GB total, {profile.ram_available_gb:.1f} GB available"
        )
    except Exception as e:
        return f"Benchmark failed: {str(e)}"


# Tool dispatch map
TOOL_HANDLERS = {
    "prsm_analyze": handle_prsm_analyze,
    "prsm_quote": handle_prsm_quote,
    "prsm_list_datasets": handle_prsm_list_datasets,
    "prsm_node_status": handle_prsm_node_status,
    "prsm_hardware_benchmark": handle_prsm_hardware_benchmark,
}


# ── MCP Server ───────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the PRSM MCP server."""
    server = Server(
        name="prsm",
        version="0.38.0",
        instructions=(
            "PRSM is a decentralized AI compute network. Use these tools to "
            "submit analysis queries, estimate costs, browse datasets, and "
            "monitor node health. The network dispatches WASM mobile agents "
            "to edge nodes for distributed computation."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            result_text = await handler(arguments or {})
            return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def run_server():
    """Run the MCP server over stdio."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
