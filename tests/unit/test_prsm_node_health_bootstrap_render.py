"""Sprint 331 — prsm_node_health renders bootstrap_discovery
with client_state prominent.

Sprint 329 added `bootstrap_discovery` subsystem to
/health/detailed. Sprint 331 extends `prsm_node_health` MCP
to render its load-bearing fields (client_state + connected +
discovered_peer_count) inline so AI-assisted operators
triaging via the MCP tool see the discovery state without
having to drill into /bootstrap/status.

Other subsystems already have per-subsystem-name special
rendering (payment_escrow shows pending_count, job_history
shows count+persisted, ftns_ledger shows shortened address).
This sprint extends the same pattern.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import handle_prsm_node_health


def _health_with_bootstrap(bs_info):
    return {
        "status": "healthy",
        "node_id": "test-node",
        "subsystems": {
            "ftns_ledger": {"available": True, "status": "ok"},
            "payment_escrow": {"available": True, "status": "ok"},
            "bootstrap_discovery": bs_info,
        },
    }


def test_renders_client_state_when_bootstrap_ok():
    payload = _health_with_bootstrap({
        "available": True,
        "status": "ok",
        "client_state": "connected",
        "connected": 1,
        "discovered_peer_count": 5,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "bootstrap_discovery" in out
    # client_state surfaced inline on the subsystem line
    assert "client_state=connected" in out
    # peer count surfaced
    assert "peers=5" in out


def test_renders_client_state_when_bootstrap_degraded():
    payload = _health_with_bootstrap({
        "available": False,
        "status": "degraded",
        "client_state": "dead",
        "connected": 0,
        "discovered_peer_count": 0,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "bootstrap_discovery" in out
    # Dead client surfaced visibly
    assert "client_state=dead" in out
    # The degraded status still appears
    assert "degraded" in out


def test_renders_not_wired_cleanly_without_extra_decoration():
    """When bootstrap_discovery is not_wired, we don't have
    client_state or peer counts to surface — render the line
    without extra decoration."""
    payload = _health_with_bootstrap({
        "available": False, "status": "not_wired",
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "bootstrap_discovery" in out
    assert "not_wired" in out
    # No spurious client_state / peers fields
    assert "client_state=" not in out


def test_renders_error_with_reason_when_bootstrap_errors():
    """If bootstrap_discovery returns status=error, surface
    the error reason like other subsystems do."""
    payload = _health_with_bootstrap({
        "available": False,
        "status": "error",
        "error": "discovery raised RuntimeError(boom)",
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    # The generic error-rendering path surfaces the reason —
    # don't double-render it
    assert "boom" in out
