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


# ── Sprint 376 — active_url rendering ────────────────


def test_renders_active_url_when_set():
    """Sprint 376: prsm_node_health renders the sprint-375
    active_url inline so operators see which bootstrap host
    is in use during triage."""
    payload = _health_with_bootstrap({
        "available": True,
        "status": "ok",
        "client_state": "connected",
        "connected": 1,
        "discovered_peer_count": 3,
        "active_url": (
            "wss://bootstrap-eu.prsm-network.com:8765"
        ),
        "fallback_enabled": True,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    # Host:port surfaced without wss:// scheme noise
    assert "bootstrap-eu.prsm-network.com:8765" in out
    assert "active=" in out
    # Standard fields still rendered
    assert "client_state=connected" in out
    assert "peers=3" in out


def test_renders_without_active_url_when_none():
    """When active_url is None (all candidates failed), the
    line falls back to client_state+peers only — no
    spurious active= field."""
    payload = _health_with_bootstrap({
        "available": False,
        "status": "degraded",
        "client_state": "dead",
        "connected": 0,
        "discovered_peer_count": 0,
        "active_url": None,
        "fallback_enabled": True,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "client_state=dead" in out
    assert "peers=0" in out
    # No spurious active= when None
    assert "active=" not in out


def test_renders_active_url_strips_scheme():
    """The renderer strips ws:// or wss:// so the host:port
    is what operators see at a glance."""
    payload = _health_with_bootstrap({
        "available": True,
        "status": "ok",
        "client_state": "connected",
        "connected": 1,
        "discovered_peer_count": 1,
        "active_url": "wss://example.com:8765",
        "fallback_enabled": True,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "active=example.com:8765" in out
    assert "active=wss://" not in out
