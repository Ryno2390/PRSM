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
import inspect
import json
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)


# Streaming helper type — callable that emits progress notifications when the
# MCP client supplied a `progressToken` in its request meta. Phase 3.x.1 Task 8.
#
# Signature: emit(message: str, progress: float, total: float | None = None) -> None
# Handlers accepting an `emit_progress` keyword argument receive a real emitter
# during streaming requests, or None for non-streaming requests. They use it
# inline like:
#
#     async def handle_prsm_inference(args, *, emit_progress=None):
#         if emit_progress: await emit_progress("Submitting...", 1, 4)
#         result = await _call_node_api(...)
#         if emit_progress: await emit_progress("Inference complete.", 4, 4)
#         return format_response(result)
#
# Progress notifications are SIDE-CHANNEL — they do NOT replace the final
# TextContent response. Non-streaming clients see only the final return value;
# streaming clients see both the progress updates and the final response.
ProgressEmitter = Callable[[str, float, Optional[float]], Awaitable[None]]

logger = logging.getLogger(__name__)

# ── Tool Definitions ─────────────────────────────────────────────────────

# 2026-05-07 (canonical-workflow gap-list delta): tools were hidden
# end-to-end because their backends depended on the deleted Agent
# Forge. The Tool definitions remain in TOOLS below (so the
# call_tool dispatch table still works for explicit invocations)
# but list_tools() filters them so client-side tool discovery does
# not surface them.
#
# 2026-05-08 (B8 unhide pass 1): prsm_analyze re-exposed.
# /compute/forge now duck-type-dispatches on
# QueryOrchestrator.dispatch_query (replacing the deleted Agent
# Forge surface) — operators with PRSM_QUERY_ORCHESTRATOR_ENABLED=1
# get a working analyze path end-to-end.
#
# 2026-05-08 (B8 unhide pass 2): prsm_dispatch_agent re-exposed.
# Its handler already routes through /compute/forge with
# manifest.query — that path now works via the same QO dispatch.
# Caveat: the user-supplied InstructionManifest is pre-validated
# locally (catches malformed manifests early) but the
# QueryOrchestrator re-decomposes server-side; the manifest's
# instruction list is currently advisory rather than executed.
# A separate sprint can wire end-to-end manifest pass-through
# (add manifest= kwarg to QueryOrchestrator.dispatch_query +
# extend /compute/forge body schema) when that becomes
# load-bearing.
#
# 2026-05-08 (B8 unhide pass 3): prsm_agent_status re-exposed.
# /compute/status/{job_id} now reads from node._payment_escrow
# (the only per-job persistent state in the synchronous-from-
# caller-view forge pipeline) and returns the escrow's lifecycle:
# pending / released / refunded / disputed + amount + timing +
# provider winner. Coverage limitation: only jobs that locked an
# escrow (budget > 0) are retrievable; budget=0 jobs (test
# fixtures, free-tier dev mode) are not.
#
# All three originally-hidden tools now functional. The
# BROKEN_TOOLS_HIDDEN set is empty; the constant remains in case
# future endpoints need temporary hiding.
BROKEN_TOOLS_HIDDEN = frozenset()

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
        name="prsm_billing_status",
        description=(
            "Look up the FTNS billing state for any prior PRSM tool invocation by job_id. "
            "Returns escrow status (pending / released / refunded), amount locked, "
            "requester / provider identifiers, and on-chain transaction references "
            "if settlement reached the chain. Use this to reconcile costs across multiple "
            "tool calls or to investigate why a particular job's escrow did not release."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "The job ID returned by a previous PRSM tool call "
                        "(e.g. forge-abc123, infer-def456). Found in the cost-reconciliation "
                        "footer of any FTNS-consuming tool response."
                    ),
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_balance_check",
        description=(
            "Check FTNS token balance + USD equivalent for a wallet "
            "address. V1 reads on-chain via the node's "
            "OnChainFTNSLedger and converts to USD using the "
            "PRSM_FTNS_USD_RATE env var as a static placeholder until "
            "the Aerodrome USDC-FTNS pool is seeded (Vision §13 "
            "Phase 5 gantt: 2026-06-15). Defaults to the node's "
            "connected wallet when no address is supplied."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": (
                        "Optional 0x-prefixed Ethereum address. When "
                        "omitted, returns balance for the node's "
                        "connected wallet."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_arbitration_preview_resolution",
        description=(
            "Compose a dispute-resolution preview from the AI side "
            "panel. Composer-only — DOES NOT call queue.resolve(). "
            "Returns the would-be-applied resolution + conflict-"
            "with-existing detection so council members can confirm "
            "intent before signing on-chain governance proposals "
            "separately. Local-resolve auth model is pending council "
            "ratification; this composer is the safe surface until "
            "that's settled."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": "Arbitration record ID.",
                },
                "decision": {
                    "type": "string",
                    "enum": [
                        "upheld_parent",
                        "rejected_parent",
                        "insufficient",
                    ],
                    "description": (
                        "Council decision on the disputed attribution."
                    ),
                },
                "by_council": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Non-empty list of council member identifiers "
                        "endorsing this decision."
                    ),
                    "minItems": 1,
                },
            },
            "required": ["record_id", "decision", "by_council"],
        },
    ),
    Tool(
        name="prsm_arbitration_record_detail",
        description=(
            "Fetch full context for a single content-attribution "
            "dispute record by ID, including its current resolution "
            "state. Council members reviewing a flagged record use "
            "this to gather context (similarity, kind, flagged_at, "
            "resolution if any) before signing on-chain governance "
            "proposals. Backed by GET /content/arbitration/queue/"
            "{record_id}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": (
                        "The arbitration record ID returned by "
                        "prsm_arbitration_status (list view)."
                    ),
                },
            },
            "required": ["record_id"],
        },
    ),
    Tool(
        name="prsm_arbitration_status",
        description=(
            "List pending content-attribution disputes awaiting "
            "council adjudication. Surfaces records flagged in "
            "the disputed similarity band (PRSM-PROV-1 Item 6). "
            "Backed by GET /content/arbitration/queue. Operators "
            "watching for council-action items use this to track "
            "which content uploads are blocked pending arbitration."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_cleanup_stale_escrows",
        description=(
            "Force-cleanup expired PENDING escrows (refund to "
            "requester). Operators use this to immediately reclaim "
            "FTNS from stuck escrows without waiting for the "
            "10-min periodic cleanup loop. Backed by POST "
            "/compute/cleanup-stale. Returns the number of escrows "
            "refunded."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_spend_summary",
        description=(
            "Sum operator's FTNS spend on completed compute jobs "
            "over the last N days (default 30). Counts RELEASED "
            "escrows only — REFUNDED + PENDING are excluded. "
            "Backed by GET /wallet/spend. Useful for cost-tracking "
            "dashboards + budget reconciliation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Window in days (1..365). Default 30.",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30,
                },
                "address": {
                    "type": "string",
                    "description": (
                        "Optional 0x address override; defaults "
                        "to node's connected wallet."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_audit_summary",
        description=(
            "Aggregated bucketed counts over the audit ring "
            "buffer for quick ops dashboards. Returns status "
            "buckets (2xx/3xx/4xx/5xx), method counts, and "
            "top-N most-frequent paths. Faster operator triage "
            "than scrolling prsm_audit_recent. Backed by GET "
            "/audit/summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "top_paths": {
                    "type": "integer",
                    "description": (
                        "Number of top paths to surface (1..100). "
                        "Default 10."
                    ),
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
            },
        },
    ),
    Tool(
        name="prsm_audit_recent",
        description=(
            "Show recent state-changing API requests (POST/PUT/"
            "PATCH/DELETE) on this node from the in-memory audit "
            "ring buffer. Each entry: timestamp, method, path, "
            "requester, status_code, request_id. Useful for "
            "operator triage: 'what just happened on my node?' "
            "Optionally paginate via limit/offset OR filter by "
            "status code (exact like '404' or range like '4xx'/'5xx')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
                "status": {
                    "type": "string",
                    "description": (
                        "Optional status filter. Either exact code "
                        "('404') or HTTP range ('2xx', '3xx', '4xx', "
                        "'5xx'). Useful for drilling into errors."
                    ),
                },
                "requester": {
                    "type": "string",
                    "description": (
                        "Optional exact-match filter on requester "
                        "node/identity. Composes with status filter."
                    ),
                },
                "path_prefix": {
                    "type": "string",
                    "description": (
                        "Optional URL path-prefix filter. E.g. "
                        "'/compute/forge' matches both "
                        "/compute/forge AND /compute/forge/quote. "
                        "Composes with status + requester filters."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_webhook_history",
        description=(
            "Recent webhook dispatch attempts (success or failure). "
            "Useful for verifying webhook integration is firing as "
            "expected — operators see event names + URLs + success/"
            "failure + status_code + error. Backed by GET "
            "/admin/webhook-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_forge_submit",
        description=(
            "Submit a query through the full Ring 1-10 Agent Forge "
            "pipeline. End-to-end sovereign-edge AI: AgentForge "
            "decomposes the query, finds shards, quotes via "
            "PricingEngine, routes (DIRECT_LLM / SINGLE_AGENT / "
            "SWARM), aggregates, settles FTNS. Returns job_id + "
            "initial status. Use prsm_quote first to estimate cost. "
            "Backed by POST /compute/forge."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to submit.",
                },
                "budget_ftns": {
                    "type": "number",
                    "description": "Max FTNS to spend (default 10.0).",
                    "default": 10.0,
                },
                "shard_cids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of content CIDs to ground "
                        "the query in. Empty = LLM-only response."
                    ),
                },
                "privacy_level": {
                    "type": "string",
                    "enum": ["none", "standard", "high", "maximum"],
                    "description": "Privacy budget tier.",
                    "default": "standard",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="prsm_content_info",
        description=(
            "Look up a specific content record by CID. Returns "
            "filename, size, content_hash, creator_id, providers, "
            "royalty_rate, parent_cids. Use to verify on-chain "
            "registration + see provider list. Backed by GET "
            "/content/{cid}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "cid": {
                    "type": "string",
                    "description": "Content ID (CID).",
                },
            },
            "required": ["cid"],
        },
    ),
    Tool(
        name="prsm_my_content",
        description=(
            "Paginated list of content uploaded by this node. "
            "Each entry: content_id, filename, size, royalty_rate, "
            "access_count, total_royalties, provenance_tx_hash. "
            "Use to verify on-chain provenance registration + "
            "track royalty accruals. Backed by GET /content/mine."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0, "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_distribution_trigger",
        description=(
            "Manually trigger pull_and_distribute on-chain. Use when "
            "the PullAndDistributeScheduler has crashed / paused, or "
            "to force an emission round (e.g., after weight "
            "ratification) without waiting for the next scheduled "
            "tick. Permissionless on-chain; caller pays gas. Backed "
            "by POST /admin/distribution/trigger."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_heartbeat_trigger",
        description=(
            "Manually record an on-chain heartbeat. Use when the "
            "HeartbeatScheduler has crashed / paused / been "
            "disabled and the operator wants to avoid the slashing "
            "window opening. Returns tx_hash + status. Backed by "
            "POST /admin/heartbeat/trigger."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_distribution_history",
        description=(
            "Recent on-chain Distributed events observed by the "
            "CompensationDistributorWatcher. Each entry: timestamp, "
            "to_creator, to_operator, to_grant, total_distributed "
            "(all FTNS wei). Operators verify emission rounds are "
            "landing + tracks the 3-pool split over time. Backed "
            "by GET /admin/distribution-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0, "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_heartbeat_history",
        description=(
            "Recent on-chain HeartbeatRecorded events observed by "
            "the StorageSlashingWatcher. Operators verify their "
            "scheduler is landing transactions on-chain. Optional "
            "`provider` filter narrows to a single address. Backed "
            "by GET /admin/heartbeat-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1, "maximum": 1000, "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0, "default": 0,
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional address filter."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_slash_history",
        description=(
            "Recent on-chain slash events observed by the "
            "StorageSlashingWatcher. Two kinds: "
            "proof_failure_slashed (verification failed) and "
            "heartbeat_missing_slashed (operator missed window). "
            "Optional `provider` filter narrows to a single "
            "address. Backed by GET /admin/slash-history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..1000). Default 20.",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
                "provider": {
                    "type": "string",
                    "description": (
                        "Optional address filter. Omit to see "
                        "fleet-wide events."
                    ),
                },
            },
        },
    ),
    Tool(
        name="prsm_earnings_summary",
        description=(
            "Operator earnings dashboard. Aggregates 3 streams: "
            "(1) royalty.claimable_wei (RoyaltyDistributor), "
            "(2) heartbeat.last_heartbeat + grace_remaining + "
            "at_risk flag (StorageSlashing), (3) distribution."
            "last_distribution + seconds_since "
            "(CompensationDistributor). Each stream isolated — "
            "RPC failure on one doesn't take down others. "
            "Backed by GET /admin/earnings-summary."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_webhook_test",
        description=(
            "Smoke-test the operator's configured webhook URL. "
            "Synthesizes a webhook.test event + dispatches via "
            "the same deliverer the DaemonWatchdog uses; returns "
            "delivery success/failure with attempts + error fields. "
            "Use after configuring PRSM_WEBHOOK_URL to verify "
            "without waiting for a real daemon crash. Backed by "
            "POST /admin/webhook-test."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_canonical_check",
        description=(
            "Verify operator's wired contract addresses match the "
            "canonical pins in networks.py for the active "
            "PRSM_NETWORK. Purpose-built for post-migration "
            "verification: after a contract redeploy ceremony "
            "(e.g., A-08 v2 RoyaltyDistributor 2026-05-09), "
            "operators run this to confirm their node picked up "
            "the new pins without manually inspecting each "
            "subsystem. Renders PASS/FAIL summary + flags any "
            "mismatch with the action hint. Backed by GET "
            "/health/detailed (filters for canonical-match fields)."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_metrics_summary",
        description=(
            "Render the node's Prometheus /metrics exposition as "
            "a human-readable summary for AI side-panel triage. "
            "Distinct from prsm_node_health (subsystem readiness) "
            "— this surfaces actual operational gauge values "
            "(escrow counts, locked FTNS, history size, claimable "
            "royalties, arbitration pending)."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_node_health",
        description=(
            "One-shot operator diagnostic surfacing per-subsystem "
            "readiness: ftns_ledger, payment_escrow, job_history, "
            "royalty_distributor. Backed by GET /health/detailed. "
            "Top-level status is healthy / degraded / unhealthy. "
            "Distinct from prsm_node_status (which focuses on Ring "
            "activation) — use this for ops/troubleshooting when a "
            "subsystem is suspected of misbehaving."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="prsm_escrow_lookup",
        description=(
            "Direct-lookup detail view of a single escrow by "
            "escrow_id. Companion to prsm_escrow_summary (list "
            "view); operators investigating a specific escrow "
            "from logs / on-chain tx receipts use this to fetch "
            "full lifecycle metadata. Backed by GET /wallet/escrows/"
            "{escrow_id}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "escrow_id": {
                    "type": "string",
                    "description": (
                        "Escrow ID (the unique primary key, "
                        "distinct from job_id)."
                    ),
                },
            },
            "required": ["escrow_id"],
        },
    ),
    Tool(
        name="prsm_escrow_summary",
        description=(
            "List active FTNS escrows for the operator's wallet "
            "(or any address via override). Surfaces outstanding "
            "compute-budget commitments — the FTNS amounts locked "
            "in pending compute jobs awaiting settlement. Backed "
            "by GET /wallet/escrows. Default returns PENDING only; "
            "pass include_terminal=true for RELEASED + REFUNDED "
            "audit view."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": (
                        "Optional 0x-prefixed address override. "
                        "Defaults to the node's connected wallet."
                    ),
                },
                "include_terminal": {
                    "type": "boolean",
                    "description": (
                        "When true, returns RELEASED + REFUNDED "
                        "escrows in addition to PENDING. Default "
                        "false (PENDING only)."
                    ),
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="prsm_jobs_list",
        description=(
            "List recent /compute/forge jobs from JobHistoryStore. "
            "Backed by GET /compute/jobs. Most-recent-first by "
            "started_at. Optional status filter (in_progress | "
            "completed | failed | cancelled). Pagination via "
            "offset + limit (max 100/page). Useful for operator "
            "dashboards + 'find my last failed job' workflows."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [
                        "in_progress", "completed", "failed", "cancelled",
                    ],
                    "description": "Optional filter by job status.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Page size (1..100). Default 20.",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "minimum": 0,
                    "default": 0,
                },
            },
        },
    ),
    Tool(
        name="prsm_unstake",
        description=(
            "Request to unstake FTNS tokens. Backed by POST "
            "/staking/unstake. Creates an unstake request that "
            "becomes available for withdrawal after the unstaking "
            "period (default 7 days). amount is optional — omit to "
            "unstake the full stake balance. Use prsm_staking_status "
            "first to find your stake_id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stake_id": {
                    "type": "string",
                    "description": "ID of the stake to unstake.",
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "Optional amount to unstake. Omit to unstake "
                        "the full stake balance. Must be positive."
                    ),
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["stake_id"],
        },
    ),
    Tool(
        name="prsm_subsystem_stats",
        description=(
            "Stats for a chosen operator subsystem. Backed by GET "
            "/settler/stats, /storage/stats, or /compute/stats "
            "depending on the `subsystem` selector. Useful for "
            "operators checking single-subsystem health without "
            "loading the full /health/detailed response."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "subsystem": {
                    "type": "string",
                    "enum": ["settler", "storage", "compute"],
                    "description": "Which subsystem to probe.",
                },
            },
            "required": ["subsystem"],
        },
    ),
    Tool(
        name="prsm_staking_status",
        description=(
            "Render the user's full staking dashboard. Backed by "
            "GET /staking/status. Shows active stakes (id, amount, "
            "type, status, rewards earned/claimed), pending unstake "
            "requests with their available_at timestamps, and "
            "totals (staked + earned + claimed). Useful for "
            "stakers tracking positions without grepping the local "
            "staking manager state."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_agents",
        description=(
            "List or search PRSM agents. Without `capability`, "
            "calls GET /agents (with optional `local_only` filter). "
            "With `capability`, routes to GET /agents/search filtered "
            "by that capability string. Useful for discovering "
            "which agents the operator can dispatch jobs to."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": (
                        "Optional capability string. When provided, "
                        "calls /agents/search; otherwise lists all."
                    ),
                    "maxLength": 256,
                },
                "local_only": {
                    "type": "boolean",
                    "description": (
                        "When listing (no capability), restrict to "
                        "locally-registered agents. Default false."
                    ),
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Max results when searching (1..100). "
                        "Default 20."
                    ),
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
        },
    ),
    Tool(
        name="prsm_agent_spending",
        description=(
            "Aggregate spending dashboard across all local agents. "
            "Backed by GET /agents/spending. Returns per-agent spent "
            "+ allowance plus totals. Useful for operators tracking "
            "agent budget burn before granting more FTNS via "
            "/agents/{agent_id}/allowance."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_peers",
        description=(
            "List currently-connected peers (outbound + inbound). "
            "Backed by GET /peers. Useful for verifying bootstrap "
            "connectivity — degraded mode is typically caused by "
            "no peers reaching the canonical wss:// bootstrap. "
            "Shows direction (outbound/inbound), peer_id, address, "
            "display_name per peer."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_transactions",
        description=(
            "Render the node's FTNS transaction history. Backed by "
            "GET /transactions. Returns tx_id, type, from/to wallet, "
            "amount, description, timestamp per record. Limit defaults "
            "to 50; capped server-side at 200. Useful for end-users "
            "tracking FTNS flows without grepping the local ledger."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of records (1..200). Default 50.",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 50,
                },
            },
        },
    ),
    Tool(
        name="prsm_info",
        description=(
            "Render static node metadata: node_id, api_version, "
            "network, chain_id, rpc_host, operator_address, "
            "agent_forge_wired, query_orchestrator state/error, "
            "and the full canonical_addresses dict (FTNS token + "
            "ProvenanceRegistry V1/V2 + RoyaltyDistributor + "
            "Foundation Safe + audit-bundle + Phase 7/8 contracts). "
            "Useful for verifying which chain/contracts a node is "
            "pinned to without parsing /health/detailed."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="prsm_cancel_job",
        description=(
            "Cancel a submitted /compute/forge job by job_id. "
            "Backed by POST /compute/cancel/{job_id}. Marks the "
            "JobHistoryStore record CANCELLED and refunds the "
            "PENDING escrow. v1 caveat: in-flight Python "
            "coroutines are NOT interrupted — the release-side "
            "race loses against the now-REFUNDED escrow (correct "
            "outcome). Useful when prsm_status_stream shows a job "
            "stuck IN_PROGRESS beyond expected duration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "Job ID returned by prsm_forge_submit or "
                        "prsm_inference."
                    ),
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_status_stream",
        description=(
            "Stream live status transitions for a submitted job. "
            "Backed by GET /compute/status/{job_id}/stream (Server-"
            "Sent Events). Blocks until the server emits a terminal "
            "event (completed / history_terminal / escrow_terminal / "
            "timeout) OR max_wait_sec elapses (default 60, clamped to "
            "[1, 600]). Returns a rendered trajectory of unique "
            "status snapshots + the terminal reason + the final "
            "status. Closes the gap referenced in prsm_forge_submit's "
            "idempotent-replay hint — end-users no longer need to "
            "hand-poll prsm_agent_status to track progress."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": (
                        "Job ID returned by prsm_forge_submit or "
                        "prsm_inference."
                    ),
                },
                "max_wait_sec": {
                    "type": "number",
                    "description": (
                        "Max seconds to block waiting for terminal "
                        "event. Clamped to [1, 600]. Default 60."
                    ),
                    "default": 60,
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="prsm_royalty_claim",
        description=(
            "Claim accumulated FTNS royalties from RoyaltyDistributor. "
            "Closes the loop on the offramp claim_required path: when "
            "coinbase_offramp_initiate reports `Prerequisite: Claim X "
            "FTNS in royalties`, this tool executes the claim. "
            "Defaults to dry_run=true (returns the artifact + claimable "
            "amount without on-chain action). Pass dry_run=false to "
            "actually execute the on-chain claim() call. v1 caveat: "
            "operator authorization is implicit via running the node "
            "with the configured FTNS private key."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": (
                        "When true (default), returns the claimable "
                        "amount + artifact without on-chain action. "
                        "When false, executes the claim() on-chain "
                        "and returns the tx hash."
                    ),
                    "default": True,
                },
            },
        },
    ),
    Tool(
        name="coinbase_offramp_initiate",
        description=(
            "Compose a pre-flight transaction summary for cashing out "
            "FTNS to USD via the Aerodrome USDC-FTNS pool + Coinbase "
            "CDP off-ramp. V1 returns the artifact described in "
            "Vision §13 Phase 5 step 2 ('Gemini presents an Artifact "
            "in your side panel'); does NOT initiate any on-chain "
            "swap or fiat off-ramp. Status is PENDING_COMMISSION "
            "until the CDP integration commissions (gates on "
            "Aerodrome pool seeding per Vision gantt 2026-06-15). "
            "Use prsm_balance_check first to confirm sufficient "
            "balance before quoting larger amounts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "usd_amount": {
                    "type": "number",
                    "description": (
                        "USD amount to off-ramp. Must be positive. "
                        "Tool returns 422 if exceeds available balance."
                    ),
                    "minimum": 0.01,
                },
                "bank_account_alias": {
                    "type": "string",
                    "description": (
                        "Optional bank-account nickname (e.g. 'primary', "
                        "'savings'). Defaults to 'primary'."
                    ),
                    "default": "primary",
                },
            },
            "required": ["usd_amount"],
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
            "IMPORTANT: privacy_tier other than 'none' is intended to require a "
            "hardware-backed TEE (SGX / TDX / SEV-SNP / TrustZone / Apple Secure Enclave). "
            "Phase 3.x.1 ships the privacy-budget gate (DP-ε accounting) but defers the "
            "hardware-TEE enforcement gate to Phase 3.x.1 Task 3. On the current mock "
            "executor, software TEE accepts all privacy tiers and the receipt records "
            "tee_type=software — verify TEE type before relying on confidentiality "
            "guarantees.\n\n"
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


async def _call_node_api(
    method: str, path: str, data: Dict = None,
    *, raw_text: bool = False,
):
    """Call the PRSM node API.

    By default returns the response as parsed JSON (Dict).
    Pass ``raw_text=True`` for endpoints that emit text/plain
    bodies (e.g., /metrics Prometheus exposition).
    """
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
                if raw_text:
                    return await resp.text()
                return await resp.json()
        else:
            async with session.post(
                f"{url}{path}",
                json=data or {},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if raw_text:
                    return await resp.text()
                return await resp.json()


# ──────────────────────────────────────────────────────────────────────
# Phase 3.x.8.1 Task 3 — SSE streaming client for /compute/inference/stream
# ──────────────────────────────────────────────────────────────────────


class InferenceError(RuntimeError):
    """Structured error from a streaming inference dispatch.

    Raised by ``_call_node_api_streaming`` when the SSE stream
    terminates with an ``event: error`` frame. ``code`` carries the
    machine-readable error code (e.g. ``EXECUTION_FAILURE``,
    ``INTERNAL_ERROR``) and ``message`` is the human-readable
    description. The MCP tool handler maps this to an MCP-friendly
    error response — same surface as a non-success unary response.
    """

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message


async def _parse_sse(response: Any) -> AsyncIterator[Tuple[str, str]]:
    """Minimal Server-Sent Events parser. Yields ``(event_type,
    data)`` tuples per W3C SSE spec.

    Frame structure:
      event: <type>\\n
      data: <payload>\\n
      \\n   <- blank line terminates the frame

    Multi-line ``data:`` is concatenated with literal newlines
    between lines. ``event:`` defaults to ``"message"`` when absent
    (per spec). Comment lines (starting with ``:``) and unknown
    fields are silently ignored. The parser does NOT try to handle
    every edge case in the SSE spec — it handles the framing PRSM
    emits, which is the strict ``event:``/``data:``/blank-line
    pattern. Anything more exotic from a peer is ignored at the
    field level rather than crashing the parser.
    """
    event_type = "message"
    data_lines: List[str] = []
    # aiohttp's response.content is an asyncio.StreamReader-shaped
    # object. iter_any() yields raw bytes chunks; we accumulate +
    # split on newlines so a chunk-boundary mid-frame doesn't break
    # parsing.
    buffer = ""
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8", errors="replace")
        # Process complete lines; keep the trailing partial line in
        # the buffer for the next chunk.
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")  # tolerate CRLF
            if line == "":
                # Blank line — frame terminator.
                if data_lines:
                    yield event_type, "\n".join(data_lines)
                    data_lines = []
                    event_type = "message"
            elif line.startswith(":"):
                # SSE comment — ignored.
                continue
            elif line.startswith("event:"):
                # ``event: <type>`` — strip the prefix + optional
                # leading space (per SSE spec's "leading space
                # after the colon is consumed if present").
                value = line[len("event:"):]
                if value.startswith(" "):
                    value = value[1:]
                event_type = value
            elif line.startswith("data:"):
                value = line[len("data:"):]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
            # Other fields (id:, retry:) are silently ignored.
    # If the connection closed mid-line (no trailing newline at all),
    # process the remaining buffered text as a final line. Then if
    # data_lines accumulated anything (with or without a trailing
    # blank-line terminator), flush them as a final frame —
    # defensive against servers that forget the trailing blank line.
    if buffer:
        line = buffer.rstrip("\r")
        if line.startswith("data:"):
            value = line[len("data:"):]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
        elif line.startswith("event:"):
            value = line[len("event:"):]
            if value.startswith(" "):
                value = value[1:]
            event_type = value
        # Other unterminated fields ignored (id:, retry:, comment).
    if data_lines:
        yield event_type, "\n".join(data_lines)


async def _call_node_api_streaming(
    path: str,
    data: Dict[str, Any],
    emit_progress: ProgressEmitter,
) -> Dict[str, Any]:
    """Open an SSE connection to a node-API endpoint, forward token
    events to ``emit_progress``, return the final result dict on
    terminal ``event: result``.

    Raises:
      ``InferenceError`` when the stream terminates with an
        ``event: error`` frame. The ``code`` attribute carries the
        server-side error code.
      ``RuntimeError`` when the stream closes without a terminal
        ``result`` or ``error`` event (server crash / connection
        drop / etc.).

    Phase 3.x.8.1 Task 3 — wires the
    ``POST /compute/inference/stream`` endpoint (Task 2) to the MCP
    progress-event surface. Caller is responsible for catching
    ``InferenceError`` and formatting an MCP-friendly error
    response.
    """
    import aiohttp
    import json as _json

    url = await _get_node_api_url()
    api_key = os.environ.get("PRSM_NODE_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    sequence_count = 0

    # No total timeout — streaming inference can run for minutes.
    # Per-chunk timeout via sock_read keeps a hung server detectable
    # without bounding the overall stream length.
    timeout = aiohttp.ClientTimeout(total=None, sock_read=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{url}{path}",
            json=data or {},
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for event_type, event_data in _parse_sse(response):
                if event_type == "token":
                    try:
                        payload = _json.loads(event_data)
                    except _json.JSONDecodeError as exc:
                        # Malformed token event — surface as an
                        # InferenceError rather than crashing the
                        # iterator. Token frames carry user-visible
                        # text so a parse failure here is a server
                        # bug worth surfacing.
                        raise InferenceError(
                            f"malformed token event: {exc}",
                            code="MALFORMED_RESPONSE",
                        )
                    text_delta = payload.get("text_delta", "")
                    sequence_count += 1
                    await emit_progress(
                        text_delta,
                        float(sequence_count),
                        None,
                    )
                elif event_type == "result":
                    try:
                        return _json.loads(event_data)
                    except _json.JSONDecodeError as exc:
                        raise InferenceError(
                            f"malformed result event: {exc}",
                            code="MALFORMED_RESPONSE",
                        )
                elif event_type == "error":
                    try:
                        err = _json.loads(event_data)
                    except _json.JSONDecodeError:
                        # Even the error event is malformed — surface
                        # as InferenceError with the raw payload.
                        raise InferenceError(
                            f"malformed error event: {event_data!r}",
                            code="MALFORMED_RESPONSE",
                        )
                    raise InferenceError(
                        err.get("error", "unknown inference error"),
                        code=err.get("code"),
                    )
                # Unknown event types are ignored — forward-compat
                # with future server-side event additions.

    # Stream closed without a terminal result/error event.
    raise RuntimeError(
        "SSE stream ended without a 'result' or 'error' event "
        "(server crashed or connection dropped mid-stream)"
    )


MINIMUM_BUDGET_FTNS = 0.01


async def handle_prsm_analyze(
    arguments: Dict[str, Any],
    *,
    emit_progress: Optional[ProgressEmitter] = None,
) -> str:
    """Handle prsm_analyze tool call.

    Streaming-aware (Phase 3.x.1 Task 8): if the MCP client included a
    progressToken, intermediate stages emit as progress notifications.
    Non-streaming clients see only the final return value (unchanged
    backwards-compatible behavior).
    """
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

    if emit_progress:
        await emit_progress("Submitting query to PRSM gateway...", 1.0, 4.0)

    try:
        if emit_progress:
            await emit_progress("Dispatching agents to swarm...", 2.0, 4.0)

        result = await _call_node_api("POST", "/compute/forge", {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        })

        if emit_progress:
            await emit_progress("Aggregating swarm results...", 3.0, 4.0)

        response = result.get("response", "")
        route = result.get("route", "unknown")
        job_id = result.get("job_id", "")

        if emit_progress:
            await emit_progress("Analysis complete.", 4.0, 4.0)

        # Cost reconciliation footer (Phase 3.x.1 Task 7).
        # The /compute/forge response includes job_id but doesn't currently
        # surface the actual settled cost in the response shape — fall back
        # to budget_ftns until the API exposes settled cost as a top-level
        # field. (Recoverable separately via prsm_billing_status.)
        footer = _format_cost_footer(
            job_id=job_id or "unknown",
            cost_ftns=result.get("cost_ftns"),
            budget_ftns=budget,
            extra_fields={
                "Route": route,
                "Privacy level": privacy,
            },
        )

        return (
            f"PRSM Analysis Result\n"
            f"====================\n\n"
            f"{response}\n"
            f"{footer}"
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
        lines.append(f"")
        lines.append(f"  Manifest JSON (pass to prsm_dispatch_agent):")
        lines.append(f"  {manifest_json}")

        # Cost reconciliation footer (Phase 3.x.1 Task 7).
        # prsm_create_agent does not itself consume FTNS — it builds the
        # manifest. Spend happens at prsm_dispatch_agent time, so the footer
        # carries the planned budget rather than a settled cost. There is no
        # job_id yet (one will be allocated at dispatch).
        footer = _format_cost_footer(
            job_id="(none — assigned at dispatch)",
            budget_ftns=budget,
            extra_fields={
                "Hardware tier": hardware_tier,
                "Operations": str(len(instructions)),
            },
            note="ℹ️  Manifest only — no FTNS consumed yet. "
                 "Cost will be charged when dispatched via prsm_dispatch_agent.",
        )
        return "\n".join(lines) + "\n" + footer

    except Exception as e:
        return f"Agent creation failed: {str(e)}"


async def handle_prsm_dispatch_agent(arguments: Dict[str, Any]) -> str:
    """Handle prsm_dispatch_agent — dispatch an instruction manifest.

    Flow (post-B8 unhide pass 2):
      1. Parse the user-supplied InstructionManifest JSON locally
         (early validation — malformed manifests rejected without
         spending FTNS or hitting the node).
      2. Forward ``manifest.query`` to /compute/forge with the
         requested budget.
      3. /compute/forge duck-type-dispatches on
         ``node.agent_forge.dispatch_query`` (QueryOrchestrator) →
         decomposes the query server-side → finds shards →
         fans out → aggregates → returns.

    Honest scope: the QueryOrchestrator currently RE-DECOMPOSES
    the natural-language query rather than consuming the user's
    pre-built manifest verbatim. The local manifest serves as a
    structured precondition (validates op set, budget hint) but
    its instruction list is not executed verbatim. A future
    sprint may wire end-to-end manifest pass-through.
    """
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
            job_id = result.get("job_id", "unknown")

            footer = _format_cost_footer(
                job_id=job_id,
                cost_ftns=result.get("cost_ftns"),
                budget_ftns=budget,
                extra_fields={
                    "Route": route,
                    "Operations": str(len(manifest.instructions)),
                },
            )
            return (
                f"Agent Dispatched\n"
                f"  Query: {manifest.query}\n\n"
                f"Result:\n{response}\n"
                f"{footer}"
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


async def handle_prsm_inference(
    arguments: Dict[str, Any],
    *,
    emit_progress: Optional[ProgressEmitter] = None,
) -> str:
    """Handle prsm_inference — TEE-attested model inference with verifiable receipt.

    Builds an InferenceRequest, calls the node API, and formats the
    response with cost-reconciliation footer.

    Phase 3.x.8.1 Task 4 — routing: streaming-capable MCP clients
    (those that supplied a ``progressToken``, surfaced as a non-None
    ``emit_progress`` callback) hit the SSE
    ``POST /compute/inference/stream`` endpoint and receive
    incremental token output as MCP progress events. Non-streaming
    clients hit the existing unary ``POST /compute/inference``
    endpoint and see only the final formatted response.

    Both paths produce the same final TextContent shape (output +
    cost-reconciliation footer with model / privacy tier / content
    tier / TEE backend / duration / settler / signature note). The
    only caller-observable difference is the per-token progress
    stream on the streaming path.
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

    # Branch on streaming-capable client. The streaming path emits
    # one progress event per StreamToken (Phase 3.x.8.1 Task 3
    # _call_node_api_streaming forwards). The unary path keeps the
    # original Phase 3.x.1 Task 8 4-stage progress milestones.
    if emit_progress is not None:
        try:
            result = await _call_node_api_streaming(
                "/compute/inference/stream",
                request_payload,
                emit_progress,
            )
        except InferenceError as e:
            # Server-side rejection (budget, model, tier, etc.) —
            # surface the structured error directly.
            return f"Inference rejected: {e.message}"
        except Exception as e:  # noqa: BLE001
            return (
                f"PRSM streaming inference failed: {e}.\n"
                f"Possible causes:\n"
                f"  • PRSM node not running (start with: prsm node start)\n"
                f"  • /compute/inference/stream endpoint not deployed "
                f"(Phase 3.x.8.1 Task 2 — verify with `curl -N` against "
                f"the node)\n"
                f"  • Network connectivity issue between MCP server and "
                f"node API"
            )
    else:
        if emit_progress:  # pragma: no cover — defensive; emit_progress is None here
            await emit_progress(
                f"Building inference request for model {model_id}...",
                1.0, 4.0,
            )
        try:
            result = await _call_node_api(
                "POST", "/compute/inference", request_payload,
            )
        except Exception as e:  # noqa: BLE001
            return (
                f"PRSM inference failed: {e}.\n"
                f"Possible causes:\n"
                f"  • PRSM node not running (start with: prsm node start)\n"
                f"  • /compute/inference endpoint not yet deployed (Phase 3.x.1 Task 5 pending; "
                f"see docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md)\n"
                f"  • Network connectivity issue between MCP server and node API"
            )

    # Surface API-level errors with helpful context (only reachable
    # on the unary path — the streaming path raises InferenceError
    # for these cases, handled above).
    if isinstance(result, dict) and result.get("error"):
        return f"Inference rejected: {result['error']}"
    if not isinstance(result, dict) or not result.get("success"):
        return (
            f"Inference failed: {result.get('error') if isinstance(result, dict) else 'unknown error'}"
        )

    # Format successful response with cost reconciliation footer
    # (Phase 3.x.1 Task 7 — uses shared _format_cost_footer helper).
    output = result.get("output", "")
    receipt = result.get("receipt") or {}

    extra: Dict[str, str] = {
        "Model": str(receipt.get("model_id", model_id)),
        "Privacy tier": f"{receipt.get('privacy_tier', privacy_tier)} (ε={receipt.get('epsilon_spent', '?')})",
        "Content tier": str(receipt.get("content_tier", content_tier)),
        "TEE backend": str(receipt.get("tee_type", "unknown")),
        "Duration": f"{receipt.get('duration_seconds', '?')}s",
        "Settler": str(receipt.get("settler_node_id", "unknown")),
    }
    note = None
    if receipt.get("settler_signature"):
        note = (
            "Receipt is signed. Verify with: "
            "prsm.compute.inference.verify_receipt(receipt, "
            "public_key_b64=<settler_pubkey>)"
        )

    footer = _format_cost_footer(
        job_id=str(receipt.get("job_id", result.get("job_id", "unknown"))),
        cost_ftns=receipt.get("cost_ftns"),
        budget_ftns=budget,
        extra_fields=extra,
        note=note,
    )

    return (
        f"PRSM Inference Result\n"
        f"=====================\n\n"
        f"{output}\n"
        f"{footer}"
    )


def _format_cost_footer(
    *,
    job_id: str,
    cost_ftns: Optional[Any] = None,
    budget_ftns: Optional[Any] = None,
    extra_fields: Optional[Dict[str, str]] = None,
    note: Optional[str] = None,
) -> str:
    """Build a uniform cost-reconciliation footer for FTNS-consuming tool responses.

    Phase 3.x.1 Task 7 — extracts the pattern from prsm_inference (Task 6) into a
    shared helper applied to all four FTNS-consuming tools (prsm_analyze,
    prsm_inference, prsm_create_agent, prsm_dispatch_agent).

    Args:
        job_id: required — the FTNS job identifier the LLM should pass to
            prsm_billing_status if it wants to reconcile later.
        cost_ftns: actual settled cost (post-execution). Falls back to "?" when
            the underlying API didn't surface it.
        budget_ftns: prepaid budget from the call. Used when cost_ftns isn't
            available, to communicate the upper-bound spend.
        extra_fields: per-tool fields (route, model, privacy tier, etc.).
            Inserted between the standard rows.
        note: optional trailing line below the rule (e.g. "Manifest only — no
            FTNS consumed yet" for prsm_create_agent).

    Returns the footer block as a single string ready to append to the response.
    """
    rule = "—" * 60
    lines = ["", rule, f"Job ID:           {job_id or 'unknown'}"]

    if extra_fields:
        for label, value in extra_fields.items():
            lines.append(f"{(label + ':'):<18}{value}")

    if cost_ftns is not None:
        lines.append(f"Cost:             {cost_ftns} FTNS")
    elif budget_ftns is not None:
        lines.append(f"Budget reserved:  {budget_ftns} FTNS")

    lines.append(f"Reconcile via:    prsm_billing_status(job_id=\"{job_id}\")")
    lines.append(rule)

    if note:
        lines.append(note)
    return "\n".join(lines)


async def handle_prsm_billing_status(arguments: Dict[str, Any]) -> str:
    """Handle prsm_billing_status — query escrow state for a prior job_id.

    Phase 3.x.1 Task 7. Calls /billing/{job_id} on the node API; formats
    the response as a structured billing report.
    """
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' argument."

    try:
        result = await _call_node_api("GET", f"/billing/{job_id}")
    except Exception as e:
        return (
            f"Failed to query billing for job_id={job_id}: {e}\n"
            f"  • Is your PRSM node running? (prsm node start)"
        )

    if isinstance(result, dict) and result.get("detail"):
        # FastAPI surfaces 404 as {"detail": "..."} — pass through as-is for
        # the LLM to read and explain to the user.
        return f"Billing query for {job_id}: {result['detail']}"

    if not isinstance(result, dict):
        return f"Unexpected billing response shape for {job_id}: {result!r}"

    lines = [
        f"PRSM Billing Status — {result.get('job_id', job_id)}",
        "=" * 60,
        f"Escrow ID:        {result.get('escrow_id', 'unknown')}",
        f"Status:           {result.get('status', 'unknown')}",
        f"Amount locked:    {result.get('amount_ftns', '?')} FTNS",
        f"Requester:        {result.get('requester_id', 'unknown')}",
    ]
    if result.get("provider_winner"):
        lines.append(f"Provider:         {result['provider_winner']}")
    if result.get("tx_lock"):
        lines.append(f"Lock tx:          {result['tx_lock']}")
    if result.get("tx_release"):
        lines.append(f"Release tx:       {result['tx_release']}")
    if result.get("created_at"):
        lines.append(f"Created at:       {result['created_at']}")
    if result.get("completed_at"):
        lines.append(f"Completed at:     {result['completed_at']}")
    return "\n".join(lines)


async def handle_prsm_balance_check(arguments: Dict[str, Any]) -> str:
    """Handle prsm_balance_check tool call.

    V1 scope: GET /balance/onchain via the node API; format the
    response as user-facing text. Closes the explicit Vision §13
    Phase 5 stand-in.
    """
    address = arguments.get("address")
    path = "/balance/onchain"
    if address:
        path = f"{path}?address={address}"

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    # 503 fallback path — endpoint returned a `detail` envelope rather
    # than a balance. Common cause: ftns_ledger not initialized
    # because PRSM_ONCHAIN_FTNS or FTNS_TOKEN_ADDRESS is unset.
    if "balance_ftns" not in result:
        detail = result.get("detail", "unknown error")
        return (
            f"On-chain FTNS not configured on this node.\n"
            f"  Detail: {detail}\n"
            f"  Set PRSM_ONCHAIN_FTNS=1 + FTNS_TOKEN_ADDRESS to enable."
        )

    addr = result["address"]
    balance_ftns = result["balance_ftns"]
    usd_rate = result["usd_rate"]
    usd_equivalent = result["usd_equivalent"]

    # Display: short address (first 10 chars) + full FTNS amount +
    # USD equivalent (with explicit rate so users see the conversion
    # they're trusting).
    short_addr = (
        addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    )

    # Aggregate-source breakdown (audit-prep §7.23 honest-scope
    # closure): when v2 fields are present in the response, render
    # the multi-source breakdown. Backwards-compat: if the endpoint
    # returns a v1-only response (no `total_ftns`), fall through to
    # the legacy single-line format.
    if "total_ftns" in result:
        claimable_ftns = result["claimable_royalties_ftns"]
        escrowed_ftns = result["escrowed_ftns"]
        total_ftns = result["total_ftns"]
        total_usd = result["total_usd_equivalent"]
        sources = result.get("sources", {})

        # Show source breakdown only when at least one extra source
        # has non-zero balance OR is wired (available=true). When
        # all extras are zero+unwired, the breakdown is just noise;
        # render legacy format.
        has_extras = (
            claimable_ftns > 0 or escrowed_ftns > 0
            or sources.get("claimable_royalties", {}).get("available", False)
            or sources.get("escrowed", {}).get("available", False)
        )
        if has_extras:
            claimable_avail = sources.get(
                "claimable_royalties", {},
            ).get("available", False)
            escrowed_avail = sources.get("escrowed", {}).get("available", False)
            claimable_marker = "" if claimable_avail else " (unavailable)"
            escrowed_marker = "" if escrowed_avail else " (unavailable)"
            return (
                f"PRSM Wallet Balance (aggregate)\n"
                f"  Address:               {short_addr}\n"
                f"  On-chain:              {balance_ftns:.6f} FTNS\n"
                f"  Claimable royalties:   "
                f"{claimable_ftns:.6f} FTNS{claimable_marker}\n"
                f"  Escrowed (pending):    "
                f"{escrowed_ftns:.6f} FTNS{escrowed_marker}\n"
                f"  ─────────────────────\n"
                f"  Total:                 {total_ftns:.6f} FTNS\n"
                f"  USD (total):           ${total_usd:,.2f}  "
                f"(@ {usd_rate} USD/FTNS)\n"
                f"  Source:                aggregate"
            )

    # Legacy single-line format (v1 response shape OR v2 with no
    # extra sources wired).
    return (
        f"PRSM Wallet Balance\n"
        f"  Address:  {short_addr}\n"
        f"  Balance:  {balance_ftns:.6f} FTNS\n"
        f"  USD:      ${usd_equivalent:,.2f}  "
        f"(@ {usd_rate} USD/FTNS)\n"
        f"  Source:   {result['source']}"
    )


async def handle_prsm_arbitration_preview_resolution(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_arbitration_preview_resolution: composer-only
    dry-run of a resolution proposal."""
    record_id = arguments.get("record_id")
    decision = arguments.get("decision")
    by_council = arguments.get("by_council") or []
    if not record_id or not decision or not by_council:
        return (
            "Missing required arguments. Need record_id, decision, "
            "by_council. Use prsm_arbitration_status to list pending "
            "records first."
        )

    body = {
        "record_id": record_id,
        "decision": decision,
        "by_council": list(by_council),
    }
    try:
        result = await _call_node_api(
            "POST", "/content/arbitration/preview-resolution", body,
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "status" not in result:
        detail = result.get("detail", "unknown error")
        if "404" in detail or "No arbitration record" in detail:
            return (
                f"Record not found: {record_id}\n  Detail: {detail}"
            )
        return f"Preview composer failed.\n  Detail: {detail}"

    record = result["record"]
    proposed = result["proposed"]
    current = result.get("current_resolution")
    conflict = result.get("conflict_with_existing", False)

    lines = [
        f"PRSM Arbitration Resolution Preview (DRY_RUN)",
        f"  Record ID:        {record_id}",
        f"",
        f"  Disputed CID:     {record.get('new_cid', '')}",
        f"  Candidate parent: {record.get('candidate_parent_cid', '')}",
        f"  Similarity:       {record.get('similarity', 0.0):.4f}",
        f"  Fingerprint kind: {record.get('fingerprint_kind', '?')}",
        f"",
        f"  Proposed:",
        f"    Decision:    {proposed['decision']}",
        f"    By council:  {', '.join(proposed['by_council'])}",
    ]
    if current is not None:
        lines.append("")
        lines.append("  Current resolution (already on record):")
        lines.append(f"    Decision:    {current.get('decision', '?')}")
        cur_council = current.get("by_council", [])
        lines.append(f"    By council:  {', '.join(cur_council) or '(none)'}")
    if conflict:
        lines.append("")
        lines.append(
            "  [!] CONFLICT: proposed decision differs from existing "
            "resolution. Re-applying via queue.resolve() would be "
            "rejected (the queue raises ValueError on conflicting "
            "re-resolve). Reconcile with council before signing the "
            "on-chain proposal."
        )
    elif current is not None:
        lines.append("")
        lines.append(
            "  [-] No conflict (proposed matches existing resolution). "
            "Re-applying via queue.resolve() would be a no-op."
        )
    lines.append("")
    lines.append(
        "  Note: composer-only artifact; does NOT call queue.resolve(). "
        "Council member signs on-chain governance proposal separately."
    )
    return "\n".join(lines)


async def handle_prsm_arbitration_record_detail(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_arbitration_record_detail: fetch full context
    for a single record + its resolution state."""
    record_id = arguments.get("record_id")
    if not record_id:
        return (
            "Missing required argument: record_id.\n"
            "Use prsm_arbitration_status to list pending records first."
        )

    try:
        result = await _call_node_api(
            "GET", f"/content/arbitration/queue/{record_id}",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "record" not in result:
        detail = result.get("detail", "unknown error")
        if "No arbitration record" in detail or "404" in detail:
            return (
                f"Record not found: {record_id}\n"
                f"  Detail: {detail}\n"
                f"  List pending records via prsm_arbitration_status."
            )
        return f"Detail fetch failed.\n  Detail: {detail}"

    record = result["record"]
    resolution = result.get("resolution")
    status = result.get("status", "?")

    lines = [
        f"PRSM Arbitration Record Detail",
        f"  Record ID:        {record_id}",
        f"  Status:           {status.upper()}",
        f"",
        f"  Disputed CID:     {record.get('new_cid', '')}",
        f"  Disputing creator: {record.get('new_creator', '')}",
        f"  Candidate parent: {record.get('candidate_parent_cid', '')}",
        f"  Parent creator:   {record.get('candidate_parent_creator', '')}",
        f"  Similarity:       {record.get('similarity', 0.0):.4f}",
        f"  Fingerprint kind: {record.get('fingerprint_kind', '?')}",
        f"  Flagged at:       {record.get('flagged_at', 0)} (unix)",
    ]
    proposal_id = record.get("proposal_id")
    if proposal_id:
        lines.append(f"  Proposal ID:      {proposal_id}")
    if resolution is not None:
        lines.append("")
        lines.append("  Resolution:")
        lines.append(f"    Decision:    {resolution.get('decision', '?')}")
        signers = resolution.get("by_council", [])
        lines.append(f"    By council:  {', '.join(signers) or '(none)'}")
    return "\n".join(lines)


async def handle_prsm_arbitration_status(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_arbitration_status: render pending arbitration
    records."""
    try:
        result = await _call_node_api(
            "GET", "/content/arbitration/queue",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "pending" not in result:
        detail = result.get("detail", "unknown error")
        return f"Arbitration query failed.\n  Detail: {detail}"

    pending = result["pending"]
    total = result["total"]
    if total == 0:
        return "No pending arbitration disputes."

    lines = [
        f"PRSM Arbitration Queue ({total} pending)",
        "  CID                    Similarity  Kind     Proposal",
        "  " + "-" * 60,
    ]
    for r in pending[:20]:  # cap at 20 for sanity
        cid = r.get("new_cid", "")[:20]
        sim = r.get("similarity", 0.0)
        kind = r.get("fingerprint_kind", "?")
        prop = r.get("proposal_id") or "(no-proposal)"
        lines.append(
            f"  {cid:<22} {sim:>6.4f}     {kind:<8} {prop}"
        )
    if total > 20:
        lines.append(f"  ... ({total - 20} more not shown)")
    return "\n".join(lines)


async def handle_prsm_cleanup_stale_escrows(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_cleanup_stale_escrows: force-cleanup expired
    escrows + return refunded count."""
    try:
        result = await _call_node_api("POST", "/compute/cleanup-stale")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "cleaned" not in result:
        detail = result.get("detail", "unknown error")
        return f"Cleanup failed.\n  Detail: {detail}"

    cleaned = result["cleaned"]
    if cleaned == 0:
        return "No stale escrows. Nothing to clean."
    return (
        f"Cleaned up {cleaned} stale escrow(s). "
        f"FTNS refunded to requester(s)."
    )


async def handle_prsm_spend_summary(arguments: Dict[str, Any]) -> str:
    """Handle prsm_spend_summary tool call: aggregate operator's
    FTNS spend over the last N days from RELEASED escrows."""
    params = []
    days = arguments.get("days", 30)
    params.append(f"days={days}")
    if "address" in arguments:
        params.append(f"address={arguments['address']}")
    path = "/wallet/spend?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "total_spent_ftns" not in result:
        detail = result.get("detail", "unknown error")
        return f"Spend summary failed.\n  Detail: {detail}"

    addr = result.get("address", "")
    short = addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    days_v = result["days"]
    total = result["total_spent_ftns"]
    count = result["escrows_count"]

    return (
        f"PRSM Spend Summary\n"
        f"  Address:        {short}\n"
        f"  Window:         last {days_v} day(s)\n"
        f"  Total spent:    {total:.6f} FTNS\n"
        f"  Released jobs:  {count}\n"
        f"  Avg / job:      "
        f"{(total / count if count else 0):.6f} FTNS"
    )


async def handle_prsm_audit_summary(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_audit_summary: render bucketed audit counts."""
    params = []
    if "top_paths" in arguments:
        params.append(f"top_paths={arguments['top_paths']}")
    path = "/audit/summary"
    if params:
        path += "?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "status_buckets" not in result:
        detail = result.get("detail", "unknown error")
        return f"Audit summary failed.\n  Detail: {detail}"

    total = result.get("total", 0)
    status = result.get("status_buckets", {})
    methods = result.get("method_buckets", {})
    top = result.get("top_paths", [])

    lines = [
        f"PRSM Audit Summary (buffer total: {total}):",
        f"",
        f"  Status buckets:",
    ]
    for bucket in ("2xx", "3xx", "4xx", "5xx", "other"):
        if bucket in status:
            lines.append(f"    {bucket}:    {status[bucket]}")
    if not status:
        lines.append("    (empty)")

    lines.append("")
    lines.append("  Methods:")
    for method, count in sorted(
        methods.items(), key=lambda kv: kv[1], reverse=True,
    ):
        lines.append(f"    {method:<8}  {count}")
    if not methods:
        lines.append("    (empty)")

    lines.append("")
    lines.append("  Top paths:")
    for entry in top:
        lines.append(
            f"    {entry['count']:>4}  {entry['path']}"
        )
    if not top:
        lines.append("    (empty)")

    return "\n".join(lines)


async def handle_prsm_audit_recent(
    arguments: Dict[str, Any],
) -> str:
    """Render recent state-changing requests from the audit ring."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    if "status" in arguments and arguments["status"]:
        params.append(f"status={arguments['status']}")
    if "requester" in arguments and arguments["requester"]:
        params.append(f"requester={arguments['requester']}")
    if "path_prefix" in arguments and arguments["path_prefix"]:
        params.append(f"path_prefix={arguments['path_prefix']}")
    path = "/audit/recent?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        return f"Audit fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    total_matched = result.get("total_matched")  # only present with filter
    status_filter = result.get("status_filter")
    requester_filter = result.get("requester_filter")
    path_prefix_filter = result.get("path_prefix_filter")
    filters_applied = []
    if status_filter:
        filters_applied.append(f"status={status_filter}")
    if requester_filter:
        filters_applied.append(f"requester={requester_filter}")
    if path_prefix_filter:
        filters_applied.append(f"path_prefix={path_prefix_filter}")
    filter_str = ", ".join(filters_applied) if filters_applied else None
    if not entries:
        if filter_str:
            return (
                f"No state-changing requests matched filter "
                f"({filter_str}) (buffer total: {total})."
            )
        return (
            f"No state-changing requests recorded in audit ring "
            f"(buffer total: {total})."
        )

    header_parts = [f"PRSM Audit Log (showing {len(entries)}"]
    if total_matched is not None:
        header_parts.append(
            f" of {total_matched} matched, {total} total"
        )
        if filter_str:
            header_parts.append(f", filter={filter_str}")
        header_parts.append(")")
    else:
        header_parts.append(f" of {total})")
    lines = [
        "".join(header_parts) + ":",
        f"  Time  Method  Status  Path",
        f"  " + "-" * 70,
    ]
    import datetime
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(
                ts,
            ).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        lines.append(
            f"  {t}  {e.get('method', '?'):<6}  "
            f"{e.get('status_code', 0):>3}     "
            f"{e.get('path', '')}"
        )
    return "\n".join(lines)


async def handle_prsm_forge_submit(
    arguments: Dict[str, Any],
) -> str:
    """Submit a query through /compute/forge."""
    body: Dict[str, Any] = {"query": arguments["query"]}
    if "budget_ftns" in arguments:
        body["budget_ftns"] = arguments["budget_ftns"]
    if "shard_cids" in arguments:
        body["shard_cids"] = arguments["shard_cids"]
    if "privacy_level" in arguments:
        body["privacy_level"] = arguments["privacy_level"]
    try:
        result = await _call_node_api(
            "POST", "/compute/forge", data=body,
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "detail" in result and "job_id" not in result:
        detail = result["detail"]
        if "agent_forge" in detail.lower() or "not available" in detail.lower():
            return (
                f"Agent Forge not enabled on this node.\n"
                f"  Detail: {detail}\n"
                f"  Operator must set "
                f"PRSM_QUERY_ORCHESTRATOR_ENABLED=1 to enable."
            )
        return f"Forge submit failed.\n  Detail: {detail}"
    if result.get("status") == "idempotent_replay":
        return (
            f"Idempotent replay (cached result).\n"
            f"  job_id: {result.get('job_id', '?')}\n"
            f"  Use prsm_jobs_list / prsm_status_stream to "
            f"observe progress."
        )
    return (
        f"Query submitted to Agent Forge.\n"
        f"  job_id: {result.get('job_id', '?')}\n"
        f"  status: {result.get('status', '?')}\n"
        f"  Use prsm_jobs_list to track progress."
    )


async def handle_prsm_content_info(
    arguments: Dict[str, Any],
) -> str:
    """Render content record by CID."""
    cid = arguments.get("cid", "").strip()
    if not cid:
        return "Missing required 'cid'."
    try:
        result = await _call_node_api("GET", f"/content/{cid}")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "cid" not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in detail.lower():
            return f"Content not found: {cid}"
        if "not initialized" in detail.lower():
            return f"Content index not configured.\n  Detail: {detail}"
        return f"Content lookup failed.\n  Detail: {detail}"

    providers = result.get("providers", [])
    parent_cids = result.get("parent_cids", [])
    lines = [
        f"PRSM Content: {result['cid']}",
        f"  Filename:     {result.get('filename', '?')}",
        f"  Size:         {result.get('size_bytes', 0)} bytes",
        f"  Hash:         {result.get('content_hash', '?')}",
        f"  Creator:      {result.get('creator_id', '?')}",
        f"  Royalty:      {result.get('royalty_rate', 0):.4f}",
        f"  Providers:    {len(providers)}"
        + (f" ({', '.join(providers[:3])}...)" if len(providers) > 3
           else f" ({', '.join(providers)})" if providers else ""),
    ]
    if parent_cids:
        lines.append(f"  Parents:      {len(parent_cids)} citation(s)")
    return "\n".join(lines)


async def handle_prsm_my_content(
    arguments: Dict[str, Any],
) -> str:
    """Render content uploaded by this node."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/content/mine?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"ContentUploader not configured.\n"
                f"  Detail: {detail}"
            )
        return f"My-content fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No uploaded content (total: {total}). "
            f"Upload via /content/upload or /content/upload/shard."
        )

    lines = [
        f"PRSM My Content (showing {len(entries)} of {total}):",
        f"  Content ID                   File           Royalties  Hits",
        "  " + "-" * 70,
    ]
    for e in entries:
        cid = e.get("content_id", "?")
        if len(cid) > 28:
            cid = cid[:14] + ".." + cid[-12:]
        fn = e.get("filename", "?")
        if len(fn) > 14:
            fn = fn[:11] + ".."
        royalties = e.get("total_royalties", 0.0)
        hits = e.get("access_count", 0)
        prov_marker = (
            "[chain]" if e.get("provenance_tx_hash") else "[off]"
        )
        lines.append(
            f"  {cid:<28}  {fn:<14}  {royalties:>9.6f}  "
            f"{hits:>4}  {prov_marker}"
        )
    return "\n".join(lines)


async def handle_prsm_distribution_trigger(
    arguments: Dict[str, Any],
) -> str:
    """Manually trigger pull_and_distribute."""
    try:
        result = await _call_node_api(
            "POST", "/admin/distribution/trigger",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "tx_hash" not in result:
        detail = result.get("detail", "unknown error")
        if "not wired" in detail.lower():
            return (
                f"CompensationDistributorClient not wired.\n"
                f"  Detail: {detail}"
            )
        return f"Distribution trigger failed.\n  Detail: {detail}"
    return (
        f"pull_and_distribute submitted on-chain.\n"
        f"  tx_hash: {result['tx_hash']}\n"
        f"  status:  {result.get('status', '?')}\n"
        f"  Use prsm_distribution_history to confirm landing."
    )


async def handle_prsm_heartbeat_trigger(
    arguments: Dict[str, Any],
) -> str:
    """Manually trigger an on-chain heartbeat record."""
    try:
        result = await _call_node_api(
            "POST", "/admin/heartbeat/trigger",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "tx_hash" not in result:
        detail = result.get("detail", "unknown error")
        if "not wired" in detail.lower():
            return (
                f"StorageSlashingClient not wired.\n"
                f"  Detail: {detail}"
            )
        return f"Heartbeat trigger failed.\n  Detail: {detail}"
    return (
        f"Heartbeat recorded on-chain.\n"
        f"  tx_hash: {result['tx_hash']}\n"
        f"  status:  {result.get('status', '?')}"
    )


async def handle_prsm_distribution_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent Distributed events."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/admin/distribution-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Distribution log not configured.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED=1."
            )
        return f"Distribution history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No distributions recorded "
            f"(buffer total: {total})."
        )

    import datetime
    lines = [
        f"PRSM Distributions (showing {len(entries)} of {total}):",
        f"  Time      Creator       Operator       Grant         Total",
        "  " + "-" * 60,
    ]
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        creator = e.get("to_creator", 0) / 1e18
        operator = e.get("to_operator", 0) / 1e18
        grant = e.get("to_grant", 0) / 1e18
        total_d = e.get("total_distributed", 0) / 1e18
        lines.append(
            f"  {t}  {creator:>10.4f}  {operator:>10.4f}  "
            f"{grant:>10.4f}  {total_d:>10.4f}"
        )
    return "\n".join(lines)


async def handle_prsm_heartbeat_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent on-chain HeartbeatRecorded events."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    if "provider" in arguments and arguments["provider"]:
        params.append(f"provider={arguments['provider']}")
    path = "/admin/heartbeat-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Heartbeat log not configured.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_STORAGE_SLASHING_WATCHER_ENABLED=1 "
                f"to enable."
            )
        return f"Heartbeat history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No heartbeats recorded "
            f"(buffer total: {total}). "
            f"Either the watcher hasn't started or no heartbeats "
            f"have landed yet."
        )

    import datetime
    lines = [
        f"PRSM Heartbeats (showing {len(entries)} of {total}):",
        f"  Observed     On-chain TS    Provider",
        "  " + "-" * 60,
    ]
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            obs = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except Exception:
            obs = "????"
        ots = e.get("onchain_timestamp", 0)
        try:
            on = datetime.datetime.fromtimestamp(ots).strftime("%H:%M:%S")
        except Exception:
            on = "????"
        provider = e.get("provider", "?")
        if len(provider) > 26:
            provider = provider[:8] + ".." + provider[-14:]
        lines.append(
            f"  {obs}     {on}      {provider}"
        )
    return "\n".join(lines)


async def handle_prsm_slash_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent on-chain slash events."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    if "provider" in arguments and arguments["provider"]:
        params.append(f"provider={arguments['provider']}")
    path = "/admin/slash-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Slash event log not configured.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_STORAGE_SLASHING_WATCHER_ENABLED=1 "
                f"+ slashing client env to enable."
            )
        return f"Slash history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No slash events recorded "
            f"(buffer total: {total})."
        )

    import datetime
    lines = [
        f"PRSM Slash Events (showing {len(entries)} of {total}):",
        f"  Time      Kind                          Provider           Slash ID",
        "  " + "-" * 80,
    ]
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        provider = e.get("provider", "?")
        if len(provider) > 18:
            provider = provider[:8] + ".." + provider[-6:]
        slash_id = e.get("slash_id", "?")
        if len(slash_id) > 18:
            slash_id = slash_id[:10] + "..."
        lines.append(
            f"  {t}  {e.get('kind', '?'):<28}  "
            f"{provider:<18}  {slash_id}"
        )
    return "\n".join(lines)


async def handle_prsm_earnings_summary(
    arguments: Dict[str, Any],
) -> str:
    """Render aggregated operator earnings dashboard."""
    try:
        result = await _call_node_api("GET", "/admin/earnings-summary")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    lines = ["PRSM Operator Earnings Summary"]
    op_addr = result.get("operator_address")
    lines.append(f"  Operator: {op_addr or '(PRSM_OPERATOR_ADDRESS unset)'}")
    lines.append("")

    royalty = result.get("royalty", {})
    if royalty.get("available"):
        wei = royalty.get("claimable_wei", 0)
        ftns = wei / 1e18
        lines.append(f"  Royalty:      {ftns:.6f} FTNS claimable")
    else:
        err = royalty.get("error")
        if err:
            lines.append(f"  Royalty:      [!] error: {err}")
        else:
            lines.append(f"  Royalty:      not wired")

    hb = result.get("heartbeat", {})
    if hb.get("available"):
        if hb.get("never_recorded"):
            lines.append(
                f"  Heartbeat:    [!] never recorded — "
                f"node will be slashed at next epoch"
            )
        elif hb.get("expired"):
            lines.append(
                f"  Heartbeat:    [!] EXPIRED — last at "
                f"{hb['last_heartbeat']}, slashing window open"
            )
        elif hb.get("at_risk"):
            lines.append(
                f"  Heartbeat:    [!] at-risk — only "
                f"{hb['grace_remaining']}s grace remaining"
            )
        else:
            lines.append(
                f"  Heartbeat:    ok — {hb['grace_remaining']}s "
                f"grace remaining (of {hb['grace_seconds']}s)"
            )
    else:
        err = hb.get("error")
        if err:
            lines.append(f"  Heartbeat:    [!] error: {err}")
        else:
            lines.append(
                f"  Heartbeat:    not wired (set "
                f"PRSM_OPERATOR_ADDRESS + StorageSlashing env)"
            )

    dist = result.get("distribution", {})
    if dist.get("available"):
        if dist.get("never_distributed"):
            lines.append(f"  Distribution: never run yet")
        else:
            secs = dist.get("seconds_since", 0)
            hours = secs // 3600
            lines.append(
                f"  Distribution: last run {hours}h ago "
                f"(timestamp {dist['last_distribution']})"
            )
    else:
        err = dist.get("error")
        if err:
            lines.append(f"  Distribution: [!] error: {err}")
        else:
            lines.append(f"  Distribution: not wired")

    return "\n".join(lines)


async def handle_prsm_webhook_history(
    arguments: Dict[str, Any],
) -> str:
    """Render recent webhook dispatch attempts."""
    params = []
    limit = arguments.get("limit", 20)
    params.append(f"limit={limit}")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/admin/webhook-history?" + "&".join(params)
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "entries" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"Webhook log not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_WEBHOOK_URL env var to enable."
            )
        return f"Webhook history fetch failed.\n  Detail: {detail}"

    entries = result["entries"]
    total = result["total"]
    if not entries:
        return (
            f"No webhook dispatches recorded "
            f"(buffer total: {total})."
        )

    lines = [
        f"PRSM Webhook History (showing {len(entries)} of {total}):",
        f"  Time      Event                  Status  Result",
        f"  " + "-" * 60,
    ]
    import datetime
    for e in entries:
        ts = e.get("timestamp", 0)
        try:
            t = datetime.datetime.fromtimestamp(
                ts,
            ).strftime("%H:%M:%S")
        except Exception:
            t = "????"
        result_marker = "[ok]" if e.get("success") else "[!]"
        lines.append(
            f"  {t}  {e.get('event', '?'):<22}  "
            f"{e.get('status_code', '?'):<5}  "
            f"{result_marker} {e.get('error', '') if not e.get('success') else 'delivered'}"
        )
    return "\n".join(lines)


async def handle_prsm_webhook_test(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_webhook_test: smoke-test the configured
    webhook URL via POST /admin/webhook-test."""
    try:
        result = await _call_node_api("POST", "/admin/webhook-test")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "success" not in result:
        detail = result.get("detail", "unknown error")
        if "not configured" in detail.lower():
            return (
                f"Webhook not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_WEBHOOK_URL env var to enable."
            )
        return f"Webhook test failed.\n  Detail: {detail}"

    success = result.get("success", False)
    status_code = result.get("status_code")
    attempts = result.get("attempts", 0)
    error = result.get("error")

    if success:
        return (
            f"PRSM Webhook Test\n"
            f"  Result:       PASS\n"
            f"  Status code:  {status_code}\n"
            f"  Attempts:     {attempts}\n"
            f"  webhook.test event delivered successfully."
        )
    return (
        f"PRSM Webhook Test\n"
        f"  Result:       FAIL\n"
        f"  Status code:  {status_code}\n"
        f"  Attempts:     {attempts}\n"
        f"  Error:        {error}\n"
        f"  Operator action: verify webhook URL reachable + "
        f"accepts POST + returns 2xx."
    )


async def handle_prsm_canonical_check(
    arguments: Dict[str, Any],
) -> str:
    """Filter /health/detailed for canonical-match fields and
    render a pass/fail summary. Designed for post-migration
    verification (e.g., after A-08 v2 RoyaltyDistributor deploy)."""
    try:
        result = await _call_node_api("GET", "/health/detailed")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    subsystems = result.get("subsystems", {})
    if not subsystems:
        return "Endpoint returned no subsystems; cannot verify canonical pins."

    matched: list = []
    mismatched: list = []
    skipped: list = []
    for name, info in subsystems.items():
        if "canonical_match" not in info:
            # Subsystem either has no on-chain contract or the
            # canonical-match check isn't implemented for it.
            if info.get("available", False):
                skipped.append((name, "no canonical pin"))
            continue
        if info["canonical_match"]:
            matched.append((name, info.get("wired_address", "")))
        else:
            mismatched.append((
                name,
                info.get("wired_address", ""),
                info.get("canonical_address", ""),
            ))

    lines = ["PRSM Canonical-Pin Check"]
    if not mismatched:
        lines.append(f"  Result: ALL {len(matched)} PIN(S) MATCH")
    else:
        lines.append(
            f"  Result: {len(mismatched)} MISMATCH(ES) "
            f"({len(matched)} match)"
        )
    lines.append("")
    if matched:
        lines.append("  Matched:")
        for name, addr in matched:
            short = addr[:10] + "..." + addr[-4:] if len(addr) > 14 else addr
            lines.append(f"    [ok]  {name:<22}  {short}")
    if mismatched:
        lines.append("")
        lines.append("  Mismatched:")
        for name, wired, canonical in mismatched:
            lines.append(f"    [!]   {name}")
            lines.append(f"            wired:     {wired}")
            lines.append(f"            canonical: {canonical}")
        lines.append("")
        lines.append(
            "  Operator action: update PRSM_*_ADDRESS env override "
            "to canonical address(es), OR remove the env override "
            "to fall through to networks.py defaults."
        )
    if skipped:
        lines.append("")
        lines.append("  Skipped (no canonical pin available):")
        for name, reason in skipped:
            lines.append(f"    [-]   {name:<22}  ({reason})")
    return "\n".join(lines)


async def handle_prsm_metrics_summary(
    arguments: Dict[str, Any],
) -> str:
    """Handle prsm_metrics_summary: parse /metrics text and
    render gauges as a side-panel summary."""
    try:
        body = await _call_node_api("GET", "/metrics", raw_text=True)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if not isinstance(body, str):
        return f"Unexpected metrics response type: {type(body).__name__}"

    gauges: list = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "metric_name VALUE" — split on whitespace.
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        name, value = parts
        gauges.append((name, value))

    if not gauges:
        return "Metrics endpoint returned no parseable gauges."

    lines = ["PRSM Metrics Summary:"]
    for name, value in gauges:
        # Strip the "prsm_" prefix for readability.
        label = name[5:] if name.startswith("prsm_") else name
        lines.append(f"  {label:<32}  {value}")
    return "\n".join(lines)


async def handle_prsm_node_health(arguments: Dict[str, Any]) -> str:
    """Handle prsm_node_health tool call: render structured
    per-subsystem readiness from /health/detailed."""
    try:
        result = await _call_node_api("GET", "/health/detailed")
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    status = result.get("status", "unknown")
    node_id = result.get("node_id", "unknown")
    subsystems = result.get("subsystems", {})

    lines = [
        f"PRSM Node Health",
        f"  Node ID:     {node_id}",
        f"  Status:      {status.upper()}",
        f"",
        f"  Subsystems:",
    ]
    for name, info in subsystems.items():
        avail = info.get("available", False)
        marker = "[ok]" if avail else "[--]"
        sub_status = info.get("status", "?")
        line = f"    {marker} {name:<22}  {sub_status}"
        if not avail and "error" in info:
            line += f"  (error: {info['error']})"
        elif name == "payment_escrow" and "pending_count" in info:
            line += f"  (pending: {info['pending_count']})"
            # Surface cleanup-task crash explicitly with [!] marker
            # since it's a high-sev silent failure.
            if info.get("cleanup_task_running") is False:
                line += "  [!] cleanup_task CRASHED"
            elif info.get("cleanup_task_running") is True:
                line += "  cleanup_task: ok"
        elif name == "job_history" and "count" in info:
            persisted = info.get("persisted", False)
            line += (
                f"  (count: {info['count']}, "
                f"persisted: {'yes' if persisted else 'no'})"
            )
        elif name == "ftns_ledger" and info.get("connected_address"):
            addr = info["connected_address"]
            short = addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
            line += f"  ({short})"
        elif name == "royalty_distributor" and "claimable_wei" in info:
            line += f"  (claimable: {info['claimable_wei']} wei)"
        lines.append(line)
        # Canonical-match indicator (shipped post-A-08 ceremony):
        # surface mismatches loudly so operators see stale env
        # overrides without scrolling. Match=True is shown subtly;
        # Match=False is the load-bearing signal.
        if "canonical_match" in info:
            if info["canonical_match"]:
                lines.append(
                    f"      -> canonical pin matches "
                    f"({info.get('wired_address', '?')[:10]}...)"
                )
            else:
                wired = info.get("wired_address", "?")
                canon = info.get("canonical_address", "?")
                lines.append(
                    f"      [!] canonical MISMATCH: wired={wired}, "
                    f"canonical={canon}"
                )
                lines.append(
                    f"        (operator action: update "
                    f"PRSM_*_ADDRESS env override or accept canonical)"
                )
    return "\n".join(lines)


async def handle_prsm_escrow_lookup(arguments: Dict[str, Any]) -> str:
    """Handle prsm_escrow_lookup: direct lookup by escrow_id."""
    escrow_id = arguments.get("escrow_id")
    if not escrow_id:
        return (
            "Missing required argument: escrow_id.\n"
            "Use prsm_escrow_summary to list active escrows first."
        )
    try:
        result = await _call_node_api(
            "GET", f"/wallet/escrows/{escrow_id}",
        )
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )
    if "escrow_id" not in result:
        detail = result.get("detail", "unknown error")
        if "404" in detail or "No escrow record" in detail:
            return (
                f"Escrow not found: {escrow_id}\n"
                f"  Detail: {detail}"
            )
        return f"Escrow lookup failed.\n  Detail: {detail}"

    lines = [
        f"PRSM Escrow Detail",
        f"  Escrow ID:        {result['escrow_id']}",
        f"  Job ID:           {result.get('job_id', '?')}",
        f"  Requester:        {result.get('requester_id', '?')}",
        f"  Amount (FTNS):    {result.get('amount_ftns', 0):.6f}",
        f"  Status:           {result.get('status', '?').upper()}",
    ]
    if result.get("provider_winner"):
        lines.append(f"  Provider winner:  {result['provider_winner']}")
    if result.get("tx_lock"):
        lines.append(f"  Lock tx:          {result['tx_lock']}")
    if result.get("tx_release"):
        lines.append(f"  Release tx:       {result['tx_release']}")
    if result.get("created_at"):
        lines.append(f"  Created at:       {result['created_at']}")
    if result.get("completed_at"):
        lines.append(f"  Completed at:     {result['completed_at']}")
    return "\n".join(lines)


async def handle_prsm_escrow_summary(arguments: Dict[str, Any]) -> str:
    """Handle prsm_escrow_summary tool call: enumerate operator's
    active escrows."""
    params = []
    if "address" in arguments:
        params.append(f"address={arguments['address']}")
    if arguments.get("include_terminal"):
        params.append("include_terminal=true")
    path = "/wallet/escrows"
    if params:
        path += "?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "escrows" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"PaymentEscrow not configured on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_escrow_summary failed.\n  Detail: {detail}"

    escrows = result["escrows"]
    total = result["total"]
    locked = result["total_locked_ftns"]
    addr = result.get("address", "")
    short_addr = (
        addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    )

    if not escrows:
        return (
            f"PRSM Escrow Summary\n"
            f"  Address:  {short_addr}\n"
            f"  No active escrows."
        )

    lines = [
        f"PRSM Escrow Summary",
        f"  Address:        {short_addr}",
        f"  Active escrows: {total}",
        f"  Locked (PENDING): {locked:.6f} FTNS",
        f"",
        f"  Job ID            Amount        Status",
        f"  " + "-" * 50,
    ]
    for e in escrows:
        lines.append(
            f"  {e['job_id']:<16}  "
            f"{e['amount_ftns']:>10.6f}  "
            f"{e['status']}"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Sprint 209 — prsm_status_stream
# Consumes /compute/status/{job_id}/stream SSE feed and renders
# status-transition trajectory for end-users polling job progress
# via MCP.
# ──────────────────────────────────────────────────────────────────────


async def _consume_status_stream(
    job_id: str, *, max_wait_sec: float,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Open a GET-SSE connection to /compute/status/{job_id}/stream,
    collect unique status snapshots, return when terminal event or
    max_wait_sec elapses.

    Returns ``(snapshots, terminal_reason, last_status)``:
      - ``snapshots`` — ordered list of unique status-dict snapshots
      - ``terminal_reason`` — one of "completed"/"history_terminal"/
        "escrow_terminal"/"timeout" (server-side) or "client_timeout"
        (max_wait_sec elapsed before server emitted terminal)
      - ``last_status`` — best-effort final status string

    Raises ``RuntimeError`` on network failure (caller renders).
    """
    import aiohttp
    import asyncio as _asyncio

    url = await _get_node_api_url()
    api_key = os.environ.get("PRSM_NODE_API_KEY", "")
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    snapshots: List[Dict[str, Any]] = []
    terminal_reason: str = "client_timeout"
    last_status: Optional[str] = None

    async def _read():
        nonlocal terminal_reason, last_status
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/compute/status/{job_id}/stream",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=max_wait_sec + 10),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"HTTP {resp.status}: {body[:200]}"
                    )
                async for event_type, data in _parse_sse(resp):
                    try:
                        payload = json.loads(data) if data else {}
                    except json.JSONDecodeError:
                        payload = {"raw": data}
                    if event_type == "status":
                        # Dedup by JSON equality against prior tail
                        # snapshot — same shape the SSE emitter uses.
                        if not snapshots or snapshots[-1] != payload:
                            snapshots.append(payload)
                            last_status = (
                                payload.get("status")
                                or payload.get("history", {}).get("status")
                                or last_status
                            )
                    elif event_type == "terminal":
                        terminal_reason = (
                            payload.get("reason") or "completed"
                        )
                        # Capture any final-status field if present.
                        last_status = (
                            payload.get("status") or last_status
                        )
                        return
                    elif event_type == "error":
                        terminal_reason = "error"
                        last_status = (
                            payload.get("error") or last_status
                        )
                        return

    try:
        await _asyncio.wait_for(_read(), timeout=max_wait_sec)
    except _asyncio.TimeoutError:
        terminal_reason = "client_timeout"
    return snapshots, terminal_reason, last_status


async def handle_prsm_status_stream(
    arguments: Dict[str, Any],
    *, emit_progress: Optional[Any] = None,
) -> str:
    """Stream job-status transitions for a given job_id.

    Blocks until the server emits a terminal SSE event OR
    max_wait_sec elapses. Returns a rendered trajectory of unique
    status snapshots + the terminal reason.
    """
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' (non-empty)."

    raw_wait = arguments.get("max_wait_sec", 60)
    try:
        max_wait_sec = float(raw_wait)
    except (TypeError, ValueError):
        max_wait_sec = 60.0
    # Clamp to [1, 600]. Sub-second polling wastes worker; 10-min
    # ceiling avoids accidentally hanging the LLM session forever.
    if max_wait_sec < 1:
        max_wait_sec = 1.0
    if max_wait_sec > 600:
        max_wait_sec = 600.0

    try:
        snapshots, terminal_reason, last_status = (
            await _consume_status_stream(
                job_id, max_wait_sec=max_wait_sec,
            )
        )
    except Exception as e:
        return (
            f"prsm_status_stream failed for job_id={job_id}: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )

    # Emit one progress per unique transition (optional).
    if emit_progress is not None:
        try:
            for i, snap in enumerate(snapshots):
                s = (
                    snap.get("status")
                    or snap.get("history", {}).get("status")
                    or "?"
                )
                await emit_progress(
                    progress=i + 1,
                    total=max(1, len(snapshots)),
                    message=f"status={s}",
                )
        except Exception:  # noqa: BLE001
            # Progress is best-effort; don't fail the call.
            pass

    lines = [
        f"Status stream for job_id={job_id}:",
    ]
    if not snapshots:
        lines.append(
            f"  (no status frames received before "
            f"{terminal_reason}; final={last_status or 'unknown'})"
        )
    else:
        for i, snap in enumerate(snapshots):
            s = (
                snap.get("status")
                or snap.get("history", {}).get("status")
                or "?"
            )
            lines.append(f"  {i+1}. status={s}")
    lines.append(
        f"  terminal_reason={terminal_reason}; "
        f"final_status={last_status or 'unknown'}"
    )
    return "\n".join(lines)


async def handle_prsm_unstake(arguments: Dict[str, Any]) -> str:
    """Sprint 217 — request to unstake FTNS tokens.

    Local-side validation: stake_id required, amount must be
    positive + finite when provided. Body-guard middleware on
    the api side catches Infinity at the wire layer too (sprint
    201), but we validate locally for friendlier UX.
    """
    import math as _math
    stake_id = (arguments.get("stake_id") or "").strip()
    if not stake_id:
        return "Missing required 'stake_id' (non-empty)."
    body: Dict[str, Any] = {"stake_id": stake_id}
    if "amount" in arguments and arguments["amount"] is not None:
        raw_amt = arguments["amount"]
        try:
            amount = float(raw_amt)
        except (TypeError, ValueError):
            return (
                f"amount must be a positive finite number; "
                f"got {raw_amt!r}."
            )
        if not _math.isfinite(amount) or amount <= 0:
            return f"amount must be a positive finite number; got {amount}."
        body["amount"] = amount
    try:
        result = await _call_node_api("POST", "/staking/unstake", body)
    except Exception as e:
        return (
            f"prsm_unstake failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    if "request_id" not in result:
        detail = result.get("detail", "unknown error")
        if "not found" in detail.lower():
            return f"Unstake refused: {detail}"
        return f"Unstake refused: {detail}"
    return (
        f"Unstake requested.\n"
        f"  request_id:    {result.get('request_id', '?')}\n"
        f"  stake_id:      {result.get('stake_id', '?')}\n"
        f"  amount:        {result.get('amount', '?')} FTNS\n"
        f"  status:        {result.get('status', '?')}\n"
        f"  requested_at:  {result.get('requested_at', '?')}\n"
        f"  available_at:  {result.get('available_at', '?')}\n"
        f"  Use prsm_unstake_finalize to withdraw when available_at "
        f"is reached, or to cancel before then."
    )


_SUBSYSTEM_STATS_PATHS = {
    "settler": "/settler/stats",
    "storage": "/storage/stats",
    "compute": "/compute/stats",
}


async def handle_prsm_subsystem_stats(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 216 — render stats for settler/storage/compute
    subsystems via a single MCP tool selector."""
    subsystem = (arguments.get("subsystem") or "").strip().lower()
    if not subsystem:
        return (
            f"Missing required 'subsystem'. Must be one of "
            f"{sorted(_SUBSYSTEM_STATS_PATHS)}."
        )
    if subsystem not in _SUBSYSTEM_STATS_PATHS:
        return (
            f"subsystem must be one of "
            f"{sorted(_SUBSYSTEM_STATS_PATHS)}; got {subsystem!r}."
        )
    path = _SUBSYSTEM_STATS_PATHS[subsystem]
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_subsystem_stats({subsystem}) failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    lines = [f"PRSM {subsystem.title()} Stats:"]
    if not result:
        lines.append("  (empty response)")
    else:
        for k, v in result.items():
            lines.append(f"  {k:<24} {v}")
    return "\n".join(lines)


async def handle_prsm_staking_status(arguments: Dict[str, Any]) -> str:
    """Sprint 215 — render GET /staking/status user dashboard."""
    try:
        result = await _call_node_api("GET", "/staking/status")
    except Exception as e:
        return (
            f"prsm_staking_status failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    user_id = result.get("user_id", "?")
    total_staked = result.get("total_staked", 0)
    earned = result.get("total_rewards_earned", 0)
    claimed = result.get("total_rewards_claimed", 0)
    stakes = result.get("active_stakes") or []
    pending = result.get("pending_unstake_requests") or []

    lines = [
        f"PRSM Staking Status (user={user_id}):",
        f"  Total staked: {total_staked} FTNS",
        f"  Rewards earned: {earned} FTNS  "
        f"(claimed: {claimed}; "
        f"unclaimed: {float(earned) - float(claimed)})",
    ]
    if stakes:
        lines.append(f"  Active stakes ({len(stakes)}):")
        for s in stakes:
            lines.append(
                f"    {s.get('stake_id', '?'):<12}  "
                f"{s.get('amount', '?')!s:>10} FTNS  "
                f"type={s.get('stake_type', '?'):<12}  "
                f"rewards={s.get('rewards_earned', 0)}"
            )
    else:
        lines.append("  No active stakes.")
    if pending:
        lines.append(f"  Pending unstake requests ({len(pending)}):")
        for r in pending:
            lines.append(
                f"    {r.get('request_id', '?'):<12}  "
                f"amount={r.get('amount', '?')}  "
                f"available_at={r.get('available_at', '?')}"
            )
    return "\n".join(lines)


async def handle_prsm_agents(arguments: Dict[str, Any]) -> str:
    """Sprint 214 — list or search agents.

    If `capability` is provided, routes to GET /agents/search;
    otherwise GET /agents with optional `local_only` filter.
    """
    capability = (arguments.get("capability") or "").strip()
    if capability:
        if len(capability) > 256:
            return f"capability must be <= 256 chars; got {len(capability)}."
        raw_limit = arguments.get("limit", 20)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return f"limit must be an integer; got {raw_limit!r}."
        if limit < 1 or limit > 100:
            return f"limit must be in [1, 100]; got {limit}."
        path = (
            f"/agents/search?capability={capability}&limit={limit}"
        )
    else:
        local_only = bool(arguments.get("local_only", False))
        path = (
            f"/agents?local_only={'true' if local_only else 'false'}"
        )
    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"prsm_agents failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    agents = result.get("agents") or []
    count = result.get("count", len(agents))
    header = (
        f"PRSM Agents matching capability='{capability}' "
        f"(count={count}):"
        if capability
        else f"PRSM Agents (count={count}):"
    )
    if not agents:
        return f"{header}\n  (none)"
    lines = [header]
    for a in agents:
        lines.append(
            f"  {a.get('agent_id', '?'):<16}  "
            f"{(a.get('display_name') or ''):<24}  "
            f"status={a.get('status', '?')}"
        )
    return "\n".join(lines)


async def handle_prsm_agent_spending(
    arguments: Dict[str, Any],
) -> str:
    """Sprint 214 — render GET /agents/spending aggregate dashboard."""
    try:
        result = await _call_node_api("GET", "/agents/spending")
    except Exception as e:
        return (
            f"prsm_agent_spending failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    agents = result.get("agents") or []
    total_spent = result.get("total_spent", 0)
    total_allow = result.get("total_allowance", 0)
    lines = [
        f"PRSM Agent Spending (total {total_spent} FTNS of "
        f"{total_allow} FTNS allowance):",
    ]
    if not agents:
        lines.append("  (no agents with spending records)")
    else:
        for a in agents:
            lines.append(
                f"  {a.get('agent_id', '?'):<16}  "
                f"spent={a.get('spent', '?')} / "
                f"allowance={a.get('allowance', '?')}"
            )
    return "\n".join(lines)


async def handle_prsm_peers(arguments: Dict[str, Any]) -> str:
    """Sprint 213 — render GET /peers connected-peer list."""
    try:
        result = await _call_node_api("GET", "/peers")
    except Exception as e:
        return (
            f"prsm_peers failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    peers = result.get("connected") or []
    count = result.get("connected_count", len(peers))
    if not peers:
        return (
            f"No peers connected (count={count}). If degraded, check "
            f"PRSM_BOOTSTRAP_ENDPOINT + /info network."
        )
    lines = [f"PRSM Connected Peers (count={count}):"]
    for p in peers:
        direction = "outbound" if p.get("outbound") else "inbound "
        peer_id = (p.get("peer_id") or "?")[:14]
        addr = (p.get("address") or "?")[:60]
        name = p.get("display_name") or ""
        lines.append(
            f"  [{direction}] {peer_id:<14}  {addr}  {name}"
        )
    return "\n".join(lines)


async def handle_prsm_transactions(arguments: Dict[str, Any]) -> str:
    """Sprint 212 — render GET /transactions history.

    Local-side limit validation: server caps at [1, 200]; reject
    out-of-range before round-trip so users get an instant error
    instead of a 422.
    """
    raw_limit = arguments.get("limit", 50)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError):
        return f"limit must be an integer in [1, 200]; got {raw_limit!r}."
    if limit < 1 or limit > 200:
        return f"limit must be in [1, 200]; got {limit}."

    try:
        result = await _call_node_api(
            "GET", f"/transactions?limit={limit}",
        )
    except Exception as e:
        return (
            f"prsm_transactions failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    txs = result.get("transactions") or []
    count = result.get("count", len(txs))
    if not txs:
        return f"No transactions in history (count={count})."
    lines = [f"PRSM Transactions (count={count}, showing {len(txs)}):"]
    for tx in txs:
        ts = tx.get("timestamp")
        ts_str = (
            f"{int(ts) % 86400 // 3600:02d}:"
            f"{int(ts) % 3600 // 60:02d}:{int(ts) % 60:02d}"
            if isinstance(ts, (int, float)) else "????"
        )
        lines.append(
            f"  {tx.get('tx_id', '?')[:18]:<18} "
            f"{tx.get('type', '?'):<18} "
            f"{tx.get('amount', '?')!s:>10} FTNS  "
            f"{(tx.get('from') or '?')[:10]}→{(tx.get('to') or '?')[:10]}  "
            f"@~{ts_str}"
        )
    return "\n".join(lines)


async def handle_prsm_info(arguments: Dict[str, Any]) -> str:
    """Sprint 211 — render GET /info static node metadata.

    Useful for verifying which chain/contracts the node is pinned
    to without parsing /health/detailed."""
    try:
        result = await _call_node_api("GET", "/info")
    except Exception as e:
        return (
            f"prsm_info failed: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    lines = ["PRSM Node Info:"]
    lines.append(f"  node_id:     {result.get('node_id', '?')}")
    lines.append(f"  api_version: {result.get('api_version', '?')}")
    if "network" in result:
        lines.append(f"  network:     {result['network']}")
    if "chain_id" in result:
        lines.append(f"  chain_id:    {result['chain_id']}")
    if "rpc_host" in result:
        lines.append(f"  rpc_host:    {result['rpc_host']}")
    if "operator_address" in result:
        lines.append(
            f"  operator:    {result['operator_address']}"
        )
    if "agent_forge_wired" in result:
        lines.append(
            f"  agent_forge_wired: {result['agent_forge_wired']}"
        )
    if "query_orchestrator_state" in result:
        lines.append(
            f"  qo_state:    {result['query_orchestrator_state']}"
        )
    if "query_orchestrator_error" in result:
        lines.append(
            f"  qo_error:    {result['query_orchestrator_error']}"
        )
    canonical = result.get("canonical_addresses") or {}
    if canonical:
        lines.append("  canonical_addresses:")
        for fld, addr in canonical.items():
            lines.append(f"    {fld:<26} {addr}")
    return "\n".join(lines)


async def handle_prsm_cancel_job(arguments: Dict[str, Any]) -> str:
    """Sprint 210 — cancel a submitted job by job_id via
    POST /compute/cancel/{job_id}. Marks history CANCELLED and
    refunds PENDING escrow (v1 caveat: in-flight Python coroutines
    not interrupted but their release-side race-loses against the
    now-REFUNDED escrow)."""
    job_id = (arguments.get("job_id") or "").strip()
    if not job_id:
        return "Missing required 'job_id' (non-empty)."
    try:
        result = await _call_node_api(
            "POST", f"/compute/cancel/{job_id}",
        )
    except Exception as e:
        return (
            f"prsm_cancel_job failed for job_id={job_id}: {e}\n"
            f"Is your PRSM node running? (prsm node start)"
        )
    # 404 path — node returns {"detail": "..."}.
    if "detail" in result and "status" not in result:
        return (
            f"Cancellation refused for job_id={job_id}: "
            f"{result.get('detail', '?')}"
        )
    return (
        f"Job {job_id} cancellation requested.\n"
        f"  status: {result.get('status', '?')}\n"
        f"  history_marked: {result.get('history_marked', '?')}\n"
        f"  escrow_refunded: {result.get('escrow_refunded', '?')}\n"
        f"  Note: in-flight Python coroutines are not interrupted; "
        f"the release-side race loses against the REFUNDED escrow."
    )


async def handle_prsm_jobs_list(arguments: Dict[str, Any]) -> str:
    """Handle prsm_jobs_list tool call: enumerate /compute/forge
    jobs with optional filter + pagination."""
    params = []
    if "status" in arguments:
        params.append(f"status={arguments['status']}")
    if "limit" in arguments:
        params.append(f"limit={arguments['limit']}")
    else:
        params.append("limit=20")
    if "offset" in arguments:
        params.append(f"offset={arguments['offset']}")
    path = "/compute/jobs"
    if params:
        path += "?" + "&".join(params)

    try:
        result = await _call_node_api("GET", path)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "jobs" not in result:
        detail = result.get("detail", "unknown error")
        if "not initialized" in detail.lower():
            return (
                f"JobHistoryStore not configured on this node.\n"
                f"  Detail: {detail}"
            )
        return f"prsm_jobs_list failed.\n  Detail: {detail}"

    jobs = result["jobs"]
    total = result.get("total", 0)
    offset = result.get("offset", 0)
    limit = result.get("limit", 0)

    if not jobs:
        return f"No jobs match the filter (total={total})."

    lines = [f"PRSM Jobs (showing {offset+1}–{offset+len(jobs)} of {total}):"]
    for j in jobs:
        ts = j.get("started_at")
        ts_str = (
            f"{int(ts) % 86400 // 3600:02d}:"
            f"{int(ts) % 3600 // 60:02d}:{int(ts) % 60:02d}"
            if isinstance(ts, (int, float)) else "????"
        )
        lines.append(
            f"  {j['job_id']:<16}  "
            f"{j['status']:<12}  "
            f"started @ ~{ts_str}  "
            f"{(j.get('query') or '')[:40]}"
        )
    if offset + len(jobs) < total:
        lines.append(
            f"  ... pass offset={offset+limit} to see next page"
        )
    return "\n".join(lines)


async def handle_prsm_royalty_claim(arguments: Dict[str, Any]) -> str:
    """Handle prsm_royalty_claim tool call.

    Closes the loop on the offramp claim_required path. Defaults
    to dry_run=true; pass dry_run=false to execute the on-chain
    claim() call.
    """
    dry_run = bool(arguments.get("dry_run", True))
    body = {"dry_run": dry_run}

    try:
        result = await _call_node_api("POST", "/wallet/royalty/claim", body)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    if "status" not in result:
        detail = result.get("detail", "unknown error")
        if "not wired" in detail.lower() or "distributor" in detail.lower():
            return (
                f"RoyaltyDistributor not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_ROYALTY_DISTRIBUTOR_ADDRESS + "
                f"FTNS_TOKEN_ADDRESS to enable."
            )
        return f"Royalty claim failed.\n  Detail: {detail}"

    status = result["status"]
    claimable = result.get("claimable_ftns", 0.0)

    if status == "DRY_RUN":
        return (
            f"PRSM Royalty Claim (dry-run)\n"
            f"  Claimable:    {claimable:.6f} FTNS\n"
            f"  Status:       DRY_RUN  (no on-chain action)\n"
            f"\n"
            f"  Pass dry_run=false to execute the on-chain claim().\n"
            f"  Example: prsm_royalty_claim {{\"dry_run\": false}}"
        )
    if status == "SKIPPED_ZERO":
        return (
            f"PRSM Royalty Claim\n"
            f"  Claimable:    0.000000 FTNS\n"
            f"  Status:       SKIPPED_ZERO\n"
            f"  Note: {result.get('note', 'No claimable balance.')}"
        )
    if status == "EXECUTED":
        return (
            f"PRSM Royalty Claim (executed)\n"
            f"  Claimed:      {result['amount_claimed_ftns']:.6f} FTNS\n"
            f"  Tx hash:      {result['tx_hash']}\n"
            f"  Status:       EXECUTED  "
            f"({result.get('transfer_status', 'OK')})"
        )
    return f"Royalty claim returned unknown status: {status}"


async def handle_coinbase_offramp_initiate(arguments: Dict[str, Any]) -> str:
    """Handle coinbase_offramp_initiate tool call.

    V1 scope: pre-flight quote composer per Vision §13 Phase 5
    step 2. Calls POST /wallet/offramp/quote and formats the
    response as a transaction-summary artifact. Does NOT initiate
    any on-chain or fiat-side action — actual execution gates on
    CDP commission per Vision gantt 2026-06-15.
    """
    if "usd_amount" not in arguments:
        return (
            "Missing required argument: usd_amount.\n"
            "Example: {\"usd_amount\": 500.0}"
        )

    body = {
        "usd_amount": arguments["usd_amount"],
        "bank_account_alias": arguments.get("bank_account_alias", "primary"),
    }

    try:
        result = await _call_node_api("POST", "/wallet/offramp/quote", body)
    except Exception as e:
        return (
            f"Cannot reach PRSM node: {str(e)}\n"
            f"Start with: prsm node start"
        )

    # 4xx/503 fallback path — endpoint returned a `detail` envelope
    # rather than a quote.
    if "quote" not in result:
        detail = result.get("detail", "unknown error")
        # Distinguish insufficient-balance (422) from misconfig (503).
        if "insufficient" in detail.lower() or "balance" in detail.lower():
            return (
                f"Insufficient balance for off-ramp.\n"
                f"  Detail: {detail}\n"
                f"  Use prsm_balance_check to verify available funds."
            )
        if "not initialized" in detail.lower() or "ftns_ledger" in detail.lower():
            return (
                f"On-chain FTNS not configured on this node.\n"
                f"  Detail: {detail}\n"
                f"  Set PRSM_ONCHAIN_FTNS=1 + FTNS_TOKEN_ADDRESS to enable."
            )
        return f"Off-ramp quote failed.\n  Detail: {detail}"

    quote = result["quote"]
    addr = result["source_address"]
    short_addr = (
        addr[:10] + "…" + addr[-4:] if len(addr) > 14 else addr
    )

    # Aggregate-source claim-required prerequisite block (v2 endpoint
    # field). When on-chain alone insufficient but claimable royalties
    # bridge the gap, surface the required claim before the quote so
    # the operator knows the eventual swap depends on it.
    prereq_block = ""
    if result.get("claim_required"):
        claim_amount = result.get("claim_amount_ftns", 0.0)
        available_ftns = result.get("available_ftns", 0.0)
        available_usd = result.get("available_usd", 0.0)
        claimable = result.get("claimable_royalties_ftns", 0.0)
        prereq_block = (
            f"\n"
            f"  Prerequisite: Claim {claim_amount:.6f} FTNS in royalties "
            f"before swap can execute\n"
            f"    Available (aggregate):  {available_ftns:.6f} FTNS  "
            f"(${available_usd:,.2f})\n"
            f"    On-chain:               "
            f"{result['source_balance_ftns']:.6f} FTNS\n"
            f"    Claimable royalties:    {claimable:.6f} FTNS\n"
        )

    return (
        f"PRSM Cash-Out Pre-Flight\n"
        f"  Requested:    ${result['requested_usd']:,.2f} USD\n"
        f"  Source:       {short_addr}\n"
        f"  Balance:      {result['source_balance_ftns']:.6f} FTNS  "
        f"(${result['source_balance_usd']:,.2f} @ "
        f"{result['usd_rate']} USD/FTNS)\n"
        f"{prereq_block}"
        f"\n"
        f"  Quote:\n"
        f"    Swap:       {quote['ftns_to_swap']:.6f} FTNS  "
        f"→  {quote['usdc_received']:,.2f} USDC  (via {quote['swap_route']})\n"
        f"    Off-ramp:   {quote['usdc_received']:,.2f} USDC  "
        f"→  ${quote['usd_settled']:,.2f} USD  "
        f"(via {quote['offramp_route']})\n"
        f"    Bank:       {quote['bank_account_alias']}\n"
        f"\n"
        f"  Status:       {result['status']}\n"
        f"\n"
        f"  Note: {result['commission_gate_note']}"
    )


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
    "prsm_billing_status": handle_prsm_billing_status,
    "prsm_balance_check": handle_prsm_balance_check,
    "prsm_arbitration_preview_resolution": handle_prsm_arbitration_preview_resolution,
    "prsm_arbitration_record_detail": handle_prsm_arbitration_record_detail,
    "prsm_arbitration_status": handle_prsm_arbitration_status,
    "prsm_audit_recent": handle_prsm_audit_recent,
    "prsm_audit_summary": handle_prsm_audit_summary,
    "prsm_canonical_check": handle_prsm_canonical_check,
    "prsm_forge_submit": handle_prsm_forge_submit,
    "prsm_content_info": handle_prsm_content_info,
    "prsm_my_content": handle_prsm_my_content,
    "prsm_distribution_trigger": handle_prsm_distribution_trigger,
    "prsm_heartbeat_trigger": handle_prsm_heartbeat_trigger,
    "prsm_distribution_history": handle_prsm_distribution_history,
    "prsm_heartbeat_history": handle_prsm_heartbeat_history,
    "prsm_slash_history": handle_prsm_slash_history,
    "prsm_earnings_summary": handle_prsm_earnings_summary,
    "prsm_webhook_history": handle_prsm_webhook_history,
    "prsm_webhook_test": handle_prsm_webhook_test,
    "prsm_metrics_summary": handle_prsm_metrics_summary,
    "prsm_cleanup_stale_escrows": handle_prsm_cleanup_stale_escrows,
    "prsm_node_health": handle_prsm_node_health,
    "prsm_spend_summary": handle_prsm_spend_summary,
    "prsm_escrow_lookup": handle_prsm_escrow_lookup,
    "prsm_escrow_summary": handle_prsm_escrow_summary,
    "prsm_jobs_list": handle_prsm_jobs_list,
    "prsm_status_stream": handle_prsm_status_stream,
    "prsm_cancel_job": handle_prsm_cancel_job,
    "prsm_info": handle_prsm_info,
    "prsm_transactions": handle_prsm_transactions,
    "prsm_peers": handle_prsm_peers,
    "prsm_agents": handle_prsm_agents,
    "prsm_staking_status": handle_prsm_staking_status,
    "prsm_subsystem_stats": handle_prsm_subsystem_stats,
    "prsm_unstake": handle_prsm_unstake,
    "prsm_agent_spending": handle_prsm_agent_spending,
    "prsm_royalty_claim": handle_prsm_royalty_claim,
    "coinbase_offramp_initiate": handle_coinbase_offramp_initiate,
}


# ── MCP Server ───────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the PRSM MCP server.

    Server version reads from installed package metadata so it
    stays in sync with pyproject.toml across releases (parallel
    to /api-info + /openapi.json + prsm_build_info gauge).
    """
    try:
        from importlib.metadata import version as _pkg_version
        _server_version = _pkg_version("prsm-network")
    except Exception:  # noqa: BLE001
        _server_version = "unknown"
    server = Server(
        name="prsm",
        version=_server_version,
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
        # Hide currently-broken tools from client-side discovery.
        # See BROKEN_TOOLS_HIDDEN above for rationale + lift-gate
        # conditions. PRSM_EXPOSE_BROKEN_TOOLS=1 forces visibility for
        # operators reconstructing the data-query path.
        expose_broken = os.getenv(
            "PRSM_EXPOSE_BROKEN_TOOLS", "",
        ).lower() in ("1", "true", "yes")
        if expose_broken:
            return TOOLS
        return [t for t in TOOLS if t.name not in BROKEN_TOOLS_HIDDEN]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            # Streaming opt-in: handlers that accept an `emit_progress` keyword
            # parameter receive a real emitter when the MCP client supplied a
            # progressToken in its request meta. Per Phase 3.x.1 Task 8.
            kwargs: Dict[str, Any] = {}
            if _handler_accepts_emit_progress(handler):
                emitter = _build_progress_emitter(server)
                # Pass the emitter even when None — handlers should treat None
                # as "client didn't ask for streaming." Passing it consistently
                # keeps the handler's kwargs surface stable across calls.
                kwargs["emit_progress"] = emitter

            result_text = await handler(arguments or {}, **kwargs)
            return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


def _handler_accepts_emit_progress(handler: Callable) -> bool:
    """True iff `handler` accepts an `emit_progress` keyword parameter.

    Used to opt handlers into streaming without changing the dispatcher
    contract for the other 15 tools.
    """
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    return "emit_progress" in sig.parameters


def _build_progress_emitter(server: Server) -> Optional[ProgressEmitter]:
    """Build a ProgressEmitter from the current MCP request context.

    Returns None when:
      - No request_context is active (server isn't currently handling a request)
      - Client did not provide a progressToken in request meta

    Returns a callable when the client opted into progress streaming. The
    callable safely no-ops if the underlying session call raises (we don't
    want a failed progress notification to break the tool response).
    """
    try:
        ctx = server.request_context
    except Exception:
        return None

    if ctx is None or ctx.meta is None:
        return None
    progress_token = getattr(ctx.meta, "progressToken", None)
    if progress_token is None:
        return None

    session = ctx.session

    async def _emit(message: str, progress: float, total: Optional[float] = None) -> None:
        try:
            await session.send_progress_notification(
                progress_token=progress_token,
                progress=progress,
                total=total,
                message=message,
            )
        except Exception as exc:
            # Log but don't propagate — a dropped progress notification
            # should not break the tool response.
            logger.warning(f"send_progress_notification failed: {exc}")

    return _emit


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
