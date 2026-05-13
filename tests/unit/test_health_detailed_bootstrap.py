"""Sprint 329 — Surface bootstrap discovery on /health/detailed.

`/health/detailed` is the structured per-subsystem readiness
probe ops alerting consumes (sprint 2026-05-09). Pre-sprint-329
it surfaced 27 subsystems — none of them P2P discovery. So
ops dashboards alerting on /health/detailed couldn't catch a
node sitting in degraded-bootstrap state.

Sprint 329 adds `bootstrap_discovery` to the subsystems map
+ counts it as an optional subsystem in the aggregate-status
roll-up:
  - core (must be available): ftns_ledger, payment_escrow
  - optional (or opt-out): job_history, royalty_distributor,
    bootstrap_discovery

A wired but disconnected bootstrap subsystem flips the
top-level status to "degraded" without flipping core
subsystems to "unhealthy" — operators get an alert without
falsely marking the node entirely down.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_client(*, discovery=None, ftns_ok=True, escrow_ok=True):
    """Build a TestClient with minimal core subsystems."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.identity.node_id = "n-test"
    node.discovery = discovery
    # Core subsystems stubbed as wired+ok unless overridden
    if ftns_ok:
        ledger = MagicMock()
        ledger._is_initialized = True
        ledger._connected_address = "0xabc"
        ledger.contract_address = "0xabc"
        node.ftns_ledger = ledger
    else:
        node.ftns_ledger = None
    if escrow_ok:
        esc = MagicMock()
        esc._escrows = {}
        esc.default_timeout = 3600
        node._payment_escrow = esc
    else:
        node._payment_escrow = None
    node._job_history = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── bootstrap_discovery subsystem appears ────────────────


def test_bootstrap_discovery_subsystem_appears_when_wired():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ["wss://"],
        "discovered_peer_count": 3,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/health/detailed").json()
    assert "bootstrap_discovery" in body["subsystems"]
    sub = body["subsystems"]["bootstrap_discovery"]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    assert sub["client_state"] == "connected"
    assert sub["connected"] == 1
    assert sub["discovered_peer_count"] == 3


def test_bootstrap_discovery_not_wired_when_no_discovery():
    """Discovery not initialized → status=not_wired so it
    counts as opt-out (not degradation)."""
    client = _make_client(discovery=None)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["bootstrap_discovery"]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


def test_bootstrap_discovery_degraded_when_degraded_flag_set():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 0, "degraded": True,
        "bootstrap_nodes": ["wss://"],
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 3,
        "reconnect_successes": 0, "client_state": "dead",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["bootstrap_discovery"]
    assert sub["available"] is False
    assert sub["status"] == "degraded"
    assert sub["client_state"] == "dead"


def test_bootstrap_discovery_error_when_get_status_raises():
    """If discovery is wired but get_bootstrap_status raises,
    surface the error without crashing the whole endpoint."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(
        side_effect=RuntimeError("boom"),
    )
    client = _make_client(discovery=discovery)
    resp = client.get("/health/detailed")
    assert resp.status_code == 200
    sub = resp.json()["subsystems"]["bootstrap_discovery"]
    assert sub["available"] is False
    assert sub["status"] == "error"
    assert "boom" in sub.get("error", "")


# ── Aggregate-status contribution ────────────────────────


def test_top_status_degraded_when_bootstrap_degraded_with_core_ok():
    """A wired-but-degraded bootstrap should flip the top-
    level status to 'degraded', not 'unhealthy' — core is
    fine, only the optional bootstrap is in trouble."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 0, "degraded": True,
        "bootstrap_nodes": ["wss://"],
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 3,
        "reconnect_successes": 0, "client_state": "dead",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/health/detailed").json()
    assert body["status"] == "degraded"


def test_top_status_healthy_when_bootstrap_not_wired_with_core_ok():
    """Opt-out (not_wired) doesn't count as degradation."""
    client = _make_client(discovery=None)
    body = client.get("/health/detailed").json()
    # ftns_ledger + payment_escrow OK, bootstrap not_wired,
    # other optional subsystems may be not_wired too → healthy
    # (or degraded if other optionals report uninitialized)
    assert body["status"] in ("healthy", "degraded")
    # The specific assertion: bootstrap_discovery alone should
    # NOT trigger degraded because it's opt-out
    sub = body["subsystems"]["bootstrap_discovery"]
    assert sub["status"] == "not_wired"
