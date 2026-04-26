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
from typing import Any, Dict, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
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
    Tool(
        name="prsm_create_agent",
        description=(
            "Create a PRSM mobile agent with a custom instruction manifest. "
            "The agent will execute the specified operations on target data shards.\n\n"
            "AVAILABLE OPERATIONS:\n"
            "- filter: Filter records by field value (params: field, value, operator)\n"
            "- aggregate: Compute sum/count/avg/min/max over records\n"
            "- group_by: Group records by a field before aggregating\n"
            "- sort: Sort records by field (params: field, ascending)\n"
            "- limit: Take first N records (params: n)\n"
            "- count: Count total records matching criteria\n"
            "- sum: Sum a numeric field (params: field)\n"
            "- average: Average a numeric field (params: field)\n"
            "- select: Select specific fields from records (params: fields[])\n"
            "- compare: Compare values across groups or time periods\n"
            "- time_series: Time-based trend analysis (params: date_field, metric_field)\n\n"
            "The instructions are composed as a pipeline: each operation feeds into the next. "
            "PRSM wraps these into a WASM mobile agent that executes securely on remote nodes.\n\n"
            "IMPORTANT: Requires FTNS budget > 0. Use prsm_quote to estimate costs first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Human-readable description of what this agent does",
                },
                "instructions": {
                    "type": "array",
                    "description": "Ordered list of operations to execute on the data",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["filter", "aggregate", "group_by", "sort", "limit",
                                        "count", "sum", "average", "select", "compare", "time_series"],
                                "description": "The operation to perform",
                            },
                            "field": {
                                "type": "string",
                                "description": "The data field this operation targets",
                            },
                            "value": {
                                "description": "Filter value, threshold, or parameter",
                            },
                            "params": {
                                "type": "object",
                                "description": "Additional operation parameters",
                            },
                        },
                        "required": ["op"],
                    },
                },
                "target_shards": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CIDs of data shards to process (leave empty for auto-discovery)",
                },
                "hardware_tier": {
                    "type": "string",
                    "enum": ["t1", "t2", "t3", "t4"],
                    "description": "Minimum hardware tier required (t1=mobile, t2=consumer, t3=high-end, t4=datacenter)",
                    "default": "t1",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS budget for execution (minimum 0.01)",
                    "minimum": 0.01,
                    "default": 5.0,
                },
            },
            "required": ["query", "instructions"],
        },
    ),
    Tool(
        name="prsm_dispatch_agent",
        description=(
            "Dispatch a previously created agent instruction manifest to the PRSM network. "
            "The agent will be sent to nodes holding the target data shards, executed in a "
            "WASM sandbox, and results aggregated.\n\n"
            "Use prsm_create_agent to build the instruction manifest, then prsm_dispatch_agent "
            "to execute it. Or use prsm_analyze for automatic end-to-end execution.\n\n"
            "IMPORTANT: Requires FTNS budget > 0 and a running PRSM node."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "instructions_json": {
                    "type": "string",
                    "description": "JSON instruction manifest from prsm_create_agent",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS budget for execution",
                    "minimum": 0.01,
                    "default": 5.0,
                },
            },
            "required": ["instructions_json"],
        },
    ),
    Tool(
        name="prsm_agent_status",
        description="Check the status of a dispatched mobile agent by its agent ID or job ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The job or agent ID to check"},
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_search_shards",
        description=(
            "Search for relevant data shards on the PRSM network by semantic similarity. "
            "Returns shards whose content is most relevant to your query, ranked by cosine similarity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to find relevant data shards"},
                "dataset_id": {"type": "string", "description": "Optional: limit search to a specific dataset"},
                "top_k": {"type": "integer", "description": "Number of results to return (default: 5)", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_upload_dataset",
        description=(
            "Upload a dataset to the PRSM network with semantic sharding and pricing. "
            "The dataset will be split into shards, distributed across nodes, and listed "
            "in the marketplace with your pricing terms. Revenue split: 80% to you (data owner), "
            "15% to compute providers, 5% to PRSM treasury."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Unique identifier for your dataset"},
                "title": {"type": "string", "description": "Human-readable title"},
                "description": {"type": "string", "description": "What this dataset contains"},
                "shard_count": {"type": "integer", "description": "Number of shards to split into", "default": 4},
                "base_access_fee": {"type": "number", "description": "FTNS fee per query against this dataset", "default": 1.0},
                "per_shard_fee": {"type": "number", "description": "Additional FTNS per shard accessed", "default": 0.1},
                "require_stake": {"type": "number", "description": "FTNS stake required for access (anti-scraping)", "default": 0},
            },
            "required": ["dataset_id", "title"],
        },
    ),
    Tool(
        name="prsm_yield_estimate",
        description=(
            "Estimate how much FTNS you would earn as a compute provider based on your hardware, "
            "hours of availability, and staking tier. Staking tiers: Casual (0 FTNS, 1x), "
            "Pledged (100 FTNS, 1.25x), Dedicated (1000 FTNS, 1.5x), Sentinel (10000 FTNS, 2x)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hours_per_day": {"type": "number", "description": "Hours available for compute per day", "default": 8},
                "stake_amount": {"type": "number", "description": "FTNS staked (determines yield boost tier)", "default": 0},
            },
        },
    ),
    Tool(
        name="prsm_stake",
        description=(
            "Preview or submit a FTNS stake on the running node. "
            "By default returns a tier preview without submitting. Pass execute=true "
            "to actually call POST /staking/stake. "
            "Tiers: Casual (0), Pledged (100+), Dedicated (1000+), Sentinel (10000+). "
            "Higher tiers earn proportionally more per compute job."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "FTNS to stake", "minimum": 0},
                "execute": {
                    "type": "boolean",
                    "description": "If true, actually submit the stake. Default false (preview only).",
                    "default": False,
                },
                "stake_type": {
                    "type": "string",
                    "description": "Stake type — passed through to the staking manager.",
                    "default": "general",
                },
            },
            "required": ["amount"],
        },
    ),
    Tool(
        name="prsm_revenue_split",
        description=(
            "Calculate how revenue would be distributed for a given payment. "
            "Default split: 80% data owner, 15% compute providers, 5% PRSM treasury. "
            "When no proprietary data is involved: 95% compute, 5% treasury."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "total_payment": {"type": "number", "description": "Total FTNS payment to split"},
                "has_data_owner": {"type": "boolean", "description": "Whether proprietary data is involved", "default": True},
                "compute_providers": {"type": "integer", "description": "Number of compute providers", "default": 1},
            },
            "required": ["total_payment"],
        },
    ),
    Tool(
        name="prsm_settlement_stats",
        description="Get FTNS settlement queue statistics — pending transfers, total settled, gas usage.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_privacy_status",
        description=(
            "Check the differential privacy budget status. Shows total epsilon spent, "
            "remaining budget, and recent privacy-consuming operations."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_training_status",
        description=(
            "Check NWTN training pipeline status — traces collected, corpus quality score, "
            "route coverage, and readiness for fine-tuning."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_inference",
        description=(
            "Run TEE-attested model inference on PRSM with verifiable receipts. "
            "Routes the prompt through PRSM's confidential-compute layer (Phase 2 TEE + "
            "Phase 7 content-tier gating) and returns the inference output along with a "
            "signed receipt that the caller can independently verify against the settling "
            "node's published Ed25519 public key.\n\n"
            "TWO LAYERS OF PRIVACY (per PRSM_Vision.md §7):\n"
            "- content_tier — encryption status of data being queried:\n"
            "    A = public content (default; no encryption)\n"
            "    B = encrypted-before-sharding (Phase 7-storage)\n"
            "    C = Tier B + Reed-Solomon erasure coding + Shamir-split keys\n"
            "- privacy_tier — TEE attestation + DP noise on activations:\n"
            "    none     = no DP noise\n"
            "    standard = ε=8.0 (default)\n"
            "    high     = ε=4.0\n"
            "    maximum  = ε=1.0\n\n"
            "End-to-end privacy for data-sensitive workloads requires both layers configured.\n\n"
            "IMPORTANT: privacy_tier other than 'none' requires a hardware-backed TEE "
            "(SGX / TDX / SEV-SNP / TrustZone / Apple Secure Enclave). On nodes with only "
            "software TEE, requests with non-none privacy_tier are rejected.\n\n"
            "REQUIRES FTNS budget > 0. Use prsm_quote first to estimate cost. "
            "Minimum budget: 0.01 FTNS."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt to send to the model",
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "Identifier of the model to run. Foundation-curated models for "
                        "Phase 3.x.1: mock-llama-3-8b, mock-mistral-7b, mock-phi-3 "
                        "(real model registry lands in Task 4)."
                    ),
                    "default": "mock-llama-3-8b",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "FTNS tokens to spend (REQUIRED, minimum 0.01).",
                    "minimum": 0.01,
                    "default": 1.0,
                },
                "privacy_tier": {
                    "type": "string",
                    "description": "Inference-layer privacy: none, standard (ε=8), high (ε=4), maximum (ε=1)",
                    "enum": ["none", "standard", "high", "maximum"],
                    "default": "standard",
                },
                "content_tier": {
                    "type": "string",
                    "description": "Content encryption tier: A (public), B (encrypted), C (encrypted+sharded)",
                    "enum": ["A", "B", "C"],
                    "default": "A",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate (model-dependent; default unbounded within budget)",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature 0.0-2.0 (default model-specific)",
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            },
            "required": ["prompt"],
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
    """Handle prsm_list_datasets tool call.

    Hits /content/search on the running node, optionally filtered by query
    string. Filters by max_price if the dataset's per-shard royalty rate is
    available in the index record. Returns up to 20 results by default.
    """
    search = (arguments.get("search") or "").strip()
    max_price = arguments.get("max_price")
    limit = int(arguments.get("limit") or 20)

    try:
        result = await _call_node_api("GET", f"/content/search?q={search}&limit={limit}")
    except Exception:
        # Fallback: surface the index stats so the user knows the node is reachable
        try:
            stats = await _call_node_api("GET", "/content/index/stats")
            return (
                f"Dataset listing failed but content index is reachable.\n"
                f"  Total entries: {stats.get('total_entries', 0)}\n"
                f"  Total bytes: {stats.get('total_bytes', 0)}\n"
                f"  Tip: publish data with prsm_upload_dataset."
            )
        except Exception:
            return "PRSM node not running. Start with: prsm node start"

    records = result.get("results", []) or []
    if max_price is not None:
        try:
            mp = float(max_price)
            records = [
                r for r in records
                if r.get("royalty_rate") is None or float(r.get("royalty_rate") or 0) <= mp
            ]
        except (TypeError, ValueError):
            pass

    if not records:
        return (
            f"No datasets found"
            + (f" matching '{search}'" if search else "")
            + ".\n  Use prsm_upload_dataset to publish data, or prsm storage upload via CLI."
        )

    lines = [f"Datasets ({len(records)} of {result.get('count', len(records))}):"]
    for r in records[:limit]:
        size_mb = (r.get("size_bytes") or 0) / (1024 * 1024)
        royalty = r.get("royalty_rate")
        royalty_str = f" royalty={royalty}" if royalty is not None else ""
        lines.append(
            f"  • {r.get('cid', '?')[:16]}…  {r.get('filename', '(unnamed)')}"
            f"  {size_mb:.2f} MB  by {r.get('creator_id', '?')[:12]}{royalty_str}"
        )
    return "\n".join(lines)


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


async def handle_prsm_create_agent(arguments: Dict[str, Any]) -> str:
    """Handle prsm_create_agent — build an instruction manifest."""
    query = arguments.get("query", "")
    instructions_raw = arguments.get("instructions", [])
    target_shards = arguments.get("target_shards", [])
    hardware_tier = arguments.get("hardware_tier", "t1")
    budget = arguments.get("budget_ftns", 5.0)

    if budget <= 0:
        return "FTNS budget required (minimum 0.01). Use prsm_quote to estimate costs."

    try:
        from prsm.compute.agents.instruction_set import (
            AgentOp, AgentInstruction, InstructionManifest,
        )

        instructions = []
        for inst in instructions_raw:
            op_str = inst.get("op", "count")
            try:
                op = AgentOp(op_str)
            except ValueError:
                return f"Unknown operation: {op_str}. Available: {[o.value for o in AgentOp]}"

            instructions.append(AgentInstruction(
                op=op,
                field=inst.get("field", ""),
                value=inst.get("value"),
                params=inst.get("params", {}),
            ))

        if not instructions:
            return "At least one instruction is required."

        manifest = InstructionManifest(
            query=query,
            instructions=instructions,
        )

        manifest_json = manifest.to_json()

        lines = [
            f"Agent Manifest Created",
            f"  Query: {query}",
            f"  Operations: {len(instructions)}",
        ]
        for i, inst in enumerate(instructions):
            field_str = f" on '{inst.field}'" if inst.field else ""
            value_str = f" = {inst.value}" if inst.value is not None else ""
            lines.append(f"    {i+1}. {inst.op.value}{field_str}{value_str}")

        lines.append(f"  Target shards: {target_shards or '(auto-discover)'}")
        lines.append(f"  Hardware tier: {hardware_tier}")
        lines.append(f"  Budget: {budget} FTNS")
        lines.append(f"")
        lines.append(f"  Manifest JSON (pass to prsm_dispatch_agent):")
        lines.append(f"  {manifest_json}")

        return "\n".join(lines)

    except Exception as e:
        return f"Agent creation failed: {str(e)}"


async def handle_prsm_dispatch_agent(arguments: Dict[str, Any]) -> str:
    """Handle prsm_dispatch_agent — dispatch an instruction manifest."""
    instructions_json = arguments.get("instructions_json", "")
    budget = arguments.get("budget_ftns", 5.0)

    if budget <= 0:
        return "FTNS budget required (minimum 0.01)."

    if not instructions_json:
        return "Missing instructions_json. Use prsm_create_agent first to build a manifest."

    try:
        from prsm.compute.agents.instruction_set import InstructionManifest

        manifest = InstructionManifest.from_json(instructions_json)

        # Try to dispatch via the node API
        try:
            result = await _call_node_api("POST", "/compute/forge", {
                "query": manifest.query,
                "budget_ftns": budget,
            })
            route = result.get("route", "unknown")
            response = result.get("response", str(result))
            return (
                f"Agent Dispatched\n"
                f"  Query: {manifest.query}\n"
                f"  Operations: {len(manifest.instructions)}\n"
                f"  Route: {route}\n"
                f"  Budget: {budget} FTNS\n\n"
                f"Result:\n{response}"
            )
        except Exception as e:
            return (
                f"Agent manifest valid ({len(manifest.instructions)} operations) "
                f"but dispatch failed: {str(e)}\n"
                f"Is your PRSM node running? (prsm node start)"
            )

    except Exception as e:
        return f"Invalid instruction manifest: {str(e)}"


async def handle_prsm_agent_status(arguments: Dict[str, Any]) -> str:
    job_id = arguments.get("job_id", "")
    try:
        result = await _call_node_api("GET", f"/compute/status/{job_id}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Could not check agent status: {e}. Is your PRSM node running?"


async def handle_prsm_search_shards(arguments: Dict[str, Any]) -> str:
    """Handle prsm_search_shards by querying the node's content index."""
    query = (arguments.get("query") or "").strip()
    top_k = int(arguments.get("top_k") or 5)

    if not query:
        return "Search requires a 'query' string."

    try:
        result = await _call_node_api("GET", f"/content/search?q={query}&limit={top_k}")
    except Exception as e:
        return f"Shard search requires a running PRSM node ({e}). Start with: prsm node start"

    records = result.get("results", []) or []
    if not records:
        return (
            f"No shards found for: '{query}'\n"
            f"  Use prsm_upload_dataset to publish data, or check the index with "
            f"prsm_list_datasets."
        )

    lines = [f"Shard search results for '{query}' (top {len(records)}):"]
    for r in records:
        providers = r.get("providers") or []
        size_mb = (r.get("size_bytes") or 0) / (1024 * 1024)
        lines.append(
            f"  • CID {r.get('cid', '?')}\n"
            f"      file={r.get('filename', '(unnamed)')}  size={size_mb:.2f} MB  "
            f"providers={len(providers)}  creator={r.get('creator_id', '?')[:12]}"
        )
    return "\n".join(lines)


async def handle_prsm_upload_dataset(arguments: Dict[str, Any]) -> str:
    dataset_id = arguments.get("dataset_id", "")
    title = arguments.get("title", "")
    description = arguments.get("description", "")
    shard_count = arguments.get("shard_count", 4)
    base_fee = arguments.get("base_access_fee", 1.0)
    per_shard = arguments.get("per_shard_fee", 0.1)
    require_stake = arguments.get("require_stake", 0)

    try:
        result = await _call_node_api("POST", "/content/upload/shard", {
            "dataset_id": dataset_id,
            "title": title,
            "description": description,
            "shard_count": shard_count,
            "base_access_fee": base_fee,
            "per_shard_fee": per_shard,
        })
        return (
            f"Dataset Published\n"
            f"  ID: {dataset_id}\n"
            f"  Title: {title}\n"
            f"  Shards: {result.get('shard_count', shard_count)}\n"
            f"  Base Fee: {base_fee} FTNS/query\n"
            f"  Per-Shard Fee: {per_shard} FTNS\n"
            f"  Revenue: 80% to you, 15% compute, 5% treasury"
        )
    except Exception as e:
        return f"Dataset upload failed: {e}. Is your PRSM node running?"


async def handle_prsm_yield_estimate(arguments: Dict[str, Any]) -> str:
    hours = arguments.get("hours_per_day", 8)
    stake = arguments.get("stake_amount", 0)
    try:
        from prsm.compute.wasm import HardwareProfiler
        from prsm.economy.pricing import PricingEngine, ProsumerTier
        profiler = HardwareProfiler()
        profile = profiler.detect()
        tier = ProsumerTier.from_stake(int(stake))
        engine = PricingEngine()
        est = engine.yield_estimate(
            hardware_tier=profile.compute_tier.value,
            tflops=profile.tflops_fp32,
            hours_per_day=hours,
            prosumer_tier=tier,
        )
        return (
            f"Yield Estimate\n"
            f"  Hardware: {profile.compute_tier.value.upper()} ({profile.tflops_fp32:.1f} TFLOPS)\n"
            f"  Stake: {stake:.0f} FTNS ({tier.name})\n"
            f"  Yield Boost: {est['yield_boost']}x\n"
            f"  Daily: {float(est['daily_ftns']):.2f} FTNS\n"
            f"  Monthly: {float(est['monthly_ftns']):.2f} FTNS"
        )
    except Exception as e:
        return f"Yield estimate failed: {e}"


async def handle_prsm_stake(arguments: Dict[str, Any]) -> str:
    """Handle prsm_stake — actually stake against the running node.

    If 'execute' is true, calls POST /staking/stake on the node. Otherwise
    returns a tier preview (the legacy info-only behavior). The execute flag
    defaults to False so naive callers can't accidentally lock tokens.
    """
    amount = arguments.get("amount", 0)
    execute = bool(arguments.get("execute", False))
    stake_type = arguments.get("stake_type", "general")

    try:
        from prsm.economy.pricing.models import ProsumerTier
        tier = ProsumerTier.from_stake(int(amount))
    except Exception as e:
        return f"Staking info failed: {e}"

    if not execute:
        return (
            f"Staking Preview (no transaction submitted)\n"
            f"  Amount: {amount} FTNS\n"
            f"  Tier: {tier.name}\n"
            f"  Yield Boost: {tier.yield_boost}x\n"
            f"  To actually stake, call prsm_stake again with execute=true."
        )

    if int(amount) <= 0:
        return "Cannot stake 0 FTNS. Provide a positive 'amount'."

    try:
        result = await _call_node_api(
            "POST",
            "/staking/stake",
            {"amount": float(amount), "stake_type": stake_type, "metadata": {}},
        )
    except Exception as e:
        return (
            f"Stake submission failed: {e}\n"
            f"  Is your node running? Tip: prsm node start"
        )

    return (
        f"Stake Submitted\n"
        f"  Stake ID: {result.get('stake_id', '?')}\n"
        f"  Amount: {result.get('amount', amount)} FTNS\n"
        f"  Type: {result.get('stake_type', stake_type)}\n"
        f"  Status: {result.get('status', 'unknown')}\n"
        f"  Tier: {tier.name} (yield boost {tier.yield_boost}x)\n"
        f"  Staked at: {result.get('staked_at', '')}"
    )


async def handle_prsm_revenue_split(arguments: Dict[str, Any]) -> str:
    total = arguments.get("total_payment", 0)
    has_data = arguments.get("has_data_owner", True)
    providers = arguments.get("compute_providers", 1)
    try:
        from decimal import Decimal
        from prsm.economy.pricing.revenue_split import RevenueSplitEngine
        engine = RevenueSplitEngine()
        provider_dict = {f"provider-{i}": 100.0/max(providers,1) for i in range(max(providers,1))}
        split = engine.calculate_split(
            total_payment=Decimal(str(total)),
            data_owner_id="data-owner" if has_data else "",
            compute_providers=provider_dict,
        )
        lines = [f"Revenue Split for {total} FTNS"]
        if has_data:
            lines.append(f"  Data Owner: {split.data_owner_amount} FTNS (80%)")
        lines.append(f"  Compute ({providers} providers): {sum(split.compute_amounts.values())} FTNS")
        lines.append(f"  Treasury: {split.treasury_amount} FTNS (5%)")
        return "\n".join(lines)
    except Exception as e:
        return f"Split calculation failed: {e}"


async def handle_prsm_settlement_stats(arguments: Dict[str, Any]) -> str:
    try:
        result = await _call_node_api("GET", "/settlement/stats")
        return json.dumps(result, indent=2)
    except Exception:
        return "Settlement stats require a running PRSM node. Start with: prsm node start"


async def handle_prsm_privacy_status(arguments: Dict[str, Any]) -> str:
    """Handle prsm_privacy_status — fetches the live DP budget audit report."""
    try:
        report = await _call_node_api("GET", "/privacy/budget")
    except Exception as e:
        return (
            f"Privacy Budget: Cannot reach node ({e}).\n"
            f"  Start a node with: prsm node start\n"
            f"  The privacy budget tracks cumulative differential privacy (ε) "
            f"spending across forge queries with privacy_level != 'none'."
        )

    max_eps = report.get("max_epsilon", 0)
    spent = report.get("total_spent", 0)
    remaining = report.get("remaining", 0)
    num_ops = report.get("num_operations", 0)
    spends = report.get("spends", []) or []

    pct = (spent / max_eps * 100.0) if max_eps else 0.0
    lines = [
        f"Differential Privacy Budget",
        f"  Total budget: {max_eps:.1f} ε",
        f"  Spent:        {spent:.3f} ε  ({pct:.1f}%)",
        f"  Remaining:    {remaining:.3f} ε",
        f"  Operations:   {num_ops}",
    ]
    if spends:
        lines.append("  Recent spends:")
        for s in spends[-5:]:
            lines.append(
                f"    - {s.get('operation', '?')}  ε={s.get('epsilon', 0):.3f}  "
                f"model={s.get('model_id', '') or '(none)'}"
            )
    return "\n".join(lines)


async def handle_prsm_training_status(arguments: Dict[str, Any]) -> str:
    try:
        result = await _call_node_api("GET", "/rings/status")
        forge = result.get("forge", {})
        traces = forge.get("traces_collected", 0)

        # Try to evaluate quality if we have traces
        from prsm.compute.nwtn.training.evaluation import TrainingEvaluator
        evaluator = TrainingEvaluator(min_traces=100)

        return (
            f"NWTN Training Pipeline Status\n"
            f"  Traces Collected: {traces}\n"
            f"  Minimum for Fine-tune: 100\n"
            f"  Ready: {'Yes' if traces >= 100 else 'No — need more queries'}\n"
            f"  Tip: Run diverse queries via prsm_analyze to build the training corpus."
        )
    except Exception:
        return (
            "NWTN Training Pipeline\n"
            "  The training pipeline collects AgentTrace data from every forge query.\n"
            "  Once enough traces are collected (100+), the NWTN model can be fine-tuned\n"
            "  for better task decomposition and WASM agent generation."
        )


async def handle_prsm_inference(arguments: Dict[str, Any]) -> str:
    """Handle prsm_inference — TEE-attested model inference with verifiable receipt.

    Builds an InferenceRequest, calls the node API at /compute/inference, and
    formats the response. Per Phase 3.x.1 design plan, the API endpoint is
    Task 5 (pending); when unavailable, a clear error is returned. The handler
    itself (this Task 6) ships now so MCP clients can begin exposing the tool
    surface and developers can wire it into their MCP configs.
    """
    prompt = arguments.get("prompt", "")
    model_id = arguments.get("model_id", "mock-llama-3-8b")
    budget = arguments.get("budget_ftns", 1.0)
    privacy_tier = arguments.get("privacy_tier", "standard")
    content_tier = arguments.get("content_tier", "A")
    max_tokens = arguments.get("max_tokens")
    temperature = arguments.get("temperature")

    if not prompt:
        return "Missing required 'prompt' argument."

    # Enforce minimum budget — same pattern as handle_prsm_analyze
    if budget <= 0:
        return (
            "PRSM inference requires an FTNS budget. "
            "Set budget_ftns to at least 0.01 FTNS.\n\n"
            "Tip: Use prsm_quote first to estimate the cost for your model + prompt."
        )
    if budget < MINIMUM_BUDGET_FTNS:
        return (
            f"Budget {budget} FTNS is below minimum ({MINIMUM_BUDGET_FTNS} FTNS). "
            f"Use prsm_quote to estimate the required budget."
        )

    request_payload: Dict[str, Any] = {
        "prompt": prompt,
        "model_id": model_id,
        "budget_ftns": budget,
        "privacy_tier": privacy_tier,
        "content_tier": content_tier,
    }
    if max_tokens is not None:
        request_payload["max_tokens"] = int(max_tokens)
    if temperature is not None:
        request_payload["temperature"] = float(temperature)

    try:
        result = await _call_node_api("POST", "/compute/inference", request_payload)
    except Exception as e:
        return (
            f"PRSM inference failed: {e}.\n"
            f"Possible causes:\n"
            f"  • PRSM node not running (start with: prsm node start)\n"
            f"  • /compute/inference endpoint not yet deployed (Phase 3.x.1 Task 5 pending; "
            f"see docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md)\n"
            f"  • Network connectivity issue between MCP server and node API"
        )

    # Surface API-level errors with helpful context
    if isinstance(result, dict) and result.get("error"):
        return f"Inference rejected: {result['error']}"
    if not isinstance(result, dict) or not result.get("success"):
        return (
            f"Inference failed: {result.get('error') if isinstance(result, dict) else 'unknown error'}"
        )

    # Format successful response with cost reconciliation footer per Phase 3.x.1
    # design plan §3.4 (per-tool billing visibility).
    output = result.get("output", "")
    receipt = result.get("receipt") or {}

    lines = [
        "PRSM Inference Result",
        "=====================",
        "",
        output,
        "",
        "—" * 60,
        f"Job ID:           {receipt.get('job_id', 'unknown')}",
        f"Model:            {receipt.get('model_id', model_id)}",
        f"Privacy tier:     {receipt.get('privacy_tier', privacy_tier)}"
        f" (ε={receipt.get('epsilon_spent', '?')})",
        f"Content tier:     {receipt.get('content_tier', content_tier)}",
        f"TEE backend:      {receipt.get('tee_type', 'unknown')}",
        f"Cost:             {receipt.get('cost_ftns', '?')} FTNS",
        f"Duration:         {receipt.get('duration_seconds', '?')}s",
        f"Settler:          {receipt.get('settler_node_id', 'unknown')}",
        "—" * 60,
    ]
    if receipt.get("settler_signature"):
        lines.append(
            "Receipt is signed. Verify with: prsm.compute.inference.verify_receipt("
            "receipt, public_key_b64=<settler_pubkey>)"
        )
    return "\n".join(lines)


# Tool dispatch map
TOOL_HANDLERS = {
    "prsm_analyze": handle_prsm_analyze,
    "prsm_quote": handle_prsm_quote,
    "prsm_list_datasets": handle_prsm_list_datasets,
    "prsm_node_status": handle_prsm_node_status,
    "prsm_hardware_benchmark": handle_prsm_hardware_benchmark,
    "prsm_create_agent": handle_prsm_create_agent,
    "prsm_dispatch_agent": handle_prsm_dispatch_agent,
    "prsm_agent_status": handle_prsm_agent_status,
    "prsm_search_shards": handle_prsm_search_shards,
    "prsm_upload_dataset": handle_prsm_upload_dataset,
    "prsm_yield_estimate": handle_prsm_yield_estimate,
    "prsm_stake": handle_prsm_stake,
    "prsm_revenue_split": handle_prsm_revenue_split,
    "prsm_settlement_stats": handle_prsm_settlement_stats,
    "prsm_privacy_status": handle_prsm_privacy_status,
    "prsm_training_status": handle_prsm_training_status,
    "prsm_inference": handle_prsm_inference,
}


# ── MCP Server ───────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the PRSM MCP server."""
    server = Server(
        name="prsm",
        version="0.39.0",
        instructions=(
            "PRSM is a decentralized AI compute network. Use these tools to "
            "submit analysis queries, run TEE-attested inference, estimate "
            "costs, browse datasets, and monitor node health. The network "
            "dispatches WASM mobile agents to edge nodes for distributed "
            "computation, and runs sharded inference under TEE attestation "
            "for verifiable confidential compute."
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
    """Entry point for the MCP server.

    CRITICAL: MCP stdio protocol requires that ONLY JSON-RPC messages
    go to stdout. All logging and print output must go to stderr.
    We capture stdout during PRSM imports to prevent structlog noise
    from corrupting the JSON-RPC stream.
    """
    import sys

    # Temporarily redirect stdout to stderr during imports
    # (structlog prints to stdout on module load)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    # Suppress all logging to stderr
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )
    for name in [
        "prsm", "prsm.core", "prsm.compute", "prsm.data", "prsm.economy",
        "prsm.node", "structlog", "httpx", "aiohttp",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Force-import ALL PRSM modules while stdout is captured
    # This ensures structlog's noisy output goes to stderr, not stdout
    _imports = [
        "prsm", "prsm.core", "prsm.core.config", "prsm.core.models",
        "prsm.compute.wasm.profiler", "prsm.compute.wasm.profiler_models",
        "prsm.compute.tee.platform_detect", "prsm.compute.tee.models",
        "prsm.economy.pricing", "prsm.economy.pricing.engine",
        "prsm.compute.nwtn.agent_forge",
    ]
    for mod in _imports:
        try:
            __import__(mod)
        except Exception:
            pass

    # Restore stdout for MCP protocol
    sys.stdout = real_stdout

    asyncio.run(run_server())


if __name__ == "__main__":
    main()
