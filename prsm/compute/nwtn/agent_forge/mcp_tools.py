"""
MCP Tool Definitions for Agent Forge
=====================================

Exposes the forge pipeline as MCP tools that any LLM can call.
"""

from typing import Any, Dict, List


FORGE_MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "prsm_analyze",
        "description": "Submit a natural language query to the PRSM network for distributed analysis. Automatically decomposes, dispatches agents, and returns aggregated results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The analysis query"},
                "budget_ftns": {"type": "number", "description": "Max FTNS to spend", "default": 10.0},
                "dataset_id": {"type": "string", "description": "Optional: specific dataset to query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "prsm_quote",
        "description": "Get a cost estimate for a query before committing. Returns compute cost, data cost, network fee, and total in FTNS.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to estimate costs for"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "prsm_list_datasets",
        "description": "Browse available datasets on the PRSM network with pricing information.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category"},
                "max_price": {"type": "number", "description": "Max base access fee in FTNS"},
            },
        },
    },
    {
        "name": "prsm_dispatch_agent",
        "description": "Low-level: dispatch a pre-built WASM agent to specific data shards.",
        "parameters": {
            "type": "object",
            "properties": {
                "wasm_url": {"type": "string", "description": "IPFS URL of WASM binary"},
                "required_data": {"type": "array", "items": {"type": "string"}, "description": "CIDs of data shards"},
                "min_hardware_tier": {"type": "string", "enum": ["t1", "t2", "t3", "t4"], "default": "t1"},
                "budget_ftns": {"type": "number", "default": 5.0},
            },
            "required": ["wasm_url", "required_data"],
        },
    },
    {
        "name": "prsm_swarm_status",
        "description": "Check the status of a running swarm job.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The swarm job ID"},
            },
            "required": ["job_id"],
        },
    },
]


def get_forge_tools() -> List[Dict[str, Any]]:
    """Return MCP tool definitions for the agent forge."""
    return FORGE_MCP_TOOLS
