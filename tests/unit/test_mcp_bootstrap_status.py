"""Sprint 266 — prsm_bootstrap_status MCP wrapper."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_bootstrap_status,
)


def test_tool_in_handlers():
    assert "prsm_bootstrap_status" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_healthy_render():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "configured_nodes": [
                "wss://bootstrap1.prsm-network.com:8765",
            ],
            "attempted_nodes": [],
            "failed_nodes": [],
            "success_node": "wss://bootstrap1.prsm-network.com:8765",
            "connected_count": 1,
            "degraded_mode": False,
            "retry_attempts": 0,
            "fallback_enabled": True,
            "fallback_activated": False,
            "fallback_succeeded": False,
            "addresses_rejected": 0,
            "source_policy": "primary_only",
            "bootstrap_client_active": True,
        }),
    ):
        result = await handle_prsm_bootstrap_status({})
    assert "✓ healthy" in result
    assert "bootstrap1.prsm-network.com" in result
    assert "connected_count:        1" in result


@pytest.mark.asyncio
async def test_degraded_render_surfaces_failed_nodes():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "configured_nodes": ["wss://b1", "wss://b2"],
            "attempted_nodes": ["wss://b1", "wss://b2"],
            "failed_nodes": ["wss://b1", "wss://b2"],
            "success_node": None,
            "connected_count": 0,
            "degraded_mode": True,
            "retry_attempts": 5,
            "fallback_enabled": True,
            "fallback_activated": True,
            "fallback_succeeded": False,
            "addresses_rejected": 4,
            "source_policy": "primary_then_fallback",
            "bootstrap_client_active": False,
        }),
    ):
        result = await handle_prsm_bootstrap_status({})
    assert "⚠ degraded" in result
    assert "failed_nodes" in result
    assert "wss://b1" in result


@pytest.mark.asyncio
async def test_disconnected_render():
    """connected_count=0 but degraded_mode=False = freshly
    disconnected (not yet flipped to degraded)."""
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "configured_nodes": [],
            "attempted_nodes": [],
            "failed_nodes": [],
            "success_node": None,
            "connected_count": 0,
            "degraded_mode": False,
            "retry_attempts": 0,
            "fallback_enabled": True,
            "fallback_activated": False,
            "fallback_succeeded": False,
            "addresses_rejected": 0,
            "source_policy": "primary_only",
            "bootstrap_client_active": False,
        }),
    ):
        result = await handle_prsm_bootstrap_status({})
    assert "⚠ disconnected" in result


@pytest.mark.asyncio
async def test_503_friendly():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "Peer discovery not initialized.",
        }),
    ):
        result = await handle_prsm_bootstrap_status({})
    assert "not wired" in result.lower()


@pytest.mark.asyncio
async def test_network_error():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(side_effect=RuntimeError("conn refused")),
    ):
        result = await handle_prsm_bootstrap_status({})
    assert "running" in result.lower() or "failed" in result.lower()
