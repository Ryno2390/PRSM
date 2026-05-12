"""Sprint 325 — Fix prsm_bootstrap_status MCP handler.

The MCP handler was authored against legacy PeerDiscovery's
payload shape (`connected_count`, `degraded_mode`,
`success_node`, `retry_attempts`, `bootstrap_client_active`,
`fallback_*`). Libp2pDiscovery — the canonical default per
sprint 164 — returns a different shape (`connected`,
`degraded`, `attempted`, `bootstrap_nodes`,
`discovered_peer_count`, plus sprint-324 counters
`peer_join_events`/`peer_leave_events`/`stale_evictions`/
`reconnect_attempts`/`reconnect_successes`/`client_state`).

Pre-sprint-325 the handler hit `"connected_count" not in
result` and emitted "Peer discovery not wired" — confusingly
wrong on a healthy node. Sprint 325 reworks the handler to:
  - Detect Libp2pDiscovery shape (presence of `client_state`
    OR `peer_join_events`)
  - Render sprint-319-through-324 counters + client_state
  - Keep legacy-shape rendering for nodes still using
    PeerDiscovery
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import handle_prsm_bootstrap_status


# ── Libp2pDiscovery shape rendering ───────────────────────


def test_renders_libp2p_shape_healthy_connected():
    payload = {
        "attempted": 1,
        "connected": 1,
        "degraded": False,
        "bootstrap_nodes": ["wss://bootstrap1.prsm-network.com:8765"],
        "discovered_peer_count": 3,
        "peer_join_events": 5,
        "peer_leave_events": 2,
        "stale_evictions": 1,
        "reconnect_attempts": 0,
        "reconnect_successes": 0,
        "client_state": "connected",
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_bootstrap_status({}))
    # Healthy header
    assert "healthy" in out.lower()
    # Sprint 324 counters surfaced
    assert "peer_join_events" in out
    assert "peer_leave_events" in out
    assert "stale_evictions" in out
    assert "reconnect_attempts" in out
    assert "client_state" in out
    # Concrete values rendered
    assert "5" in out  # peer_join_events
    assert "connected" in out  # client_state


def test_renders_libp2p_shape_degraded():
    payload = {
        "attempted": 1, "connected": 0, "degraded": True,
        "bootstrap_nodes": ["wss://bootstrap1.prsm-network.com:8765"],
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 3,
        "reconnect_successes": 0, "client_state": "dead",
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_bootstrap_status({}))
    # Degraded warning marker present
    assert "degraded" in out.lower()
    # Sentinel state surfaced
    assert "dead" in out
    # Reconnect attempts surfaced
    assert "reconnect_attempts" in out
    assert "3" in out


def test_renders_libp2p_shape_no_bootstrap_nodes():
    """Empty bootstrap_nodes list should not crash; should
    still render the rest of the payload."""
    payload = {
        "attempted": 0, "connected": 0, "degraded": False,
        "bootstrap_nodes": [],
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "none",
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_bootstrap_status({}))
    assert "client_state" in out
    assert "none" in out


# ── Legacy PeerDiscovery shape kept rendering ─────────────


def test_renders_legacy_shape_when_present():
    """If a node is still wired with the legacy PeerDiscovery,
    its payload shape (`connected_count` etc.) must still
    render correctly."""
    payload = {
        "connected_count": 1,
        "degraded_mode": False,
        "success_node": "wss://bootstrap1.prsm-network.com:8765",
        "retry_attempts": 0,
        "bootstrap_client_active": True,
        "fallback_enabled": True,
        "fallback_activated": False,
        "fallback_succeeded": False,
        "addresses_rejected": 0,
        "source_policy": "legacy",
        "configured_nodes": ["wss://bootstrap1.prsm-network.com:8765"],
        "failed_nodes": [],
    }
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_bootstrap_status({}))
    assert "connected_count" in out
    assert "bootstrap_client_active" in out


# ── Error path ────────────────────────────────────────────


def test_node_unreachable_message():
    """If the node isn't reachable, the handler should print
    a clear actionable error."""
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(side_effect=ConnectionError(
                   "Connection refused"))):
        out = asyncio.run(handle_prsm_bootstrap_status({}))
    assert "failed" in out.lower() or "refused" in out.lower()
    # Still gives operator a hint about what to do
    assert "node" in out.lower()


def test_discovery_not_initialized_error():
    """If the node returns the not-initialized 503 detail,
    render it cleanly."""
    payload = {"detail": "Peer discovery not initialized."}
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_bootstrap_status({}))
    assert "not wired" in out.lower() or "not initialized" in out.lower()
