"""Sprint 328 — Export bootstrap counters as Prometheus metrics.

Sprint 324 added 5 cumulative counters + `client_state` to
`get_bootstrap_status()`. Sprints 325-327 surfaced them via
operator API + MCP. But ops dashboards (Grafana / Prometheus /
Datadog) consume the `/metrics` endpoint, where the bootstrap
counters were absent. Sprint 328 exports them so dashboards
can alert on:

  - peer_leave rate (sudden spike = mass-disconnect event)
  - reconnect rate vs success ratio (degraded server)
  - stale eviction rate (network-partition signal)
  - connected gauge (0 = disconnected from bootstrap right now)
  - degraded gauge (1 = all bootstrap nodes unreachable)

Counter-typed: peer_join_events, peer_leave_events,
stale_evictions, reconnect_attempts, reconnect_successes
(cumulative-monotonic, only-resets-on-restart semantics).

Gauge-typed: connected, discovered_peer_count, degraded
(point-in-time snapshots).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_client(*, discovery=None):
    """Build /metrics TestClient with optional discovery stub."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.identity.node_id = "n-test"
    node.discovery = discovery
    node.ftns_ledger = None
    node._operator_address = None
    node.agent_forge = None
    node._query_orchestrator_state = None
    node._query_orchestrator_error = None
    # Disable the other /metrics probes so we test the new
    # bootstrap block in isolation
    node._payment_escrow = None
    node._job_history = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_metrics_emits_bootstrap_counters_when_discovery_present():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ["wss://"],
        "discovered_peer_count": 3,
        "peer_join_events": 5,
        "peer_leave_events": 2,
        "stale_evictions": 1,
        "reconnect_attempts": 0,
        "reconnect_successes": 0,
        "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/metrics").text

    # All 5 counter-type metrics present with correct values
    assert "prsm_bootstrap_peer_join_events_total 5" in body
    assert "prsm_bootstrap_peer_leave_events_total 2" in body
    assert "prsm_bootstrap_stale_evictions_total 1" in body
    assert "prsm_bootstrap_reconnect_attempts_total 0" in body
    assert "prsm_bootstrap_reconnect_successes_total 0" in body

    # Counter-type declarations present
    for name in (
        "prsm_bootstrap_peer_join_events_total",
        "prsm_bootstrap_peer_leave_events_total",
        "prsm_bootstrap_stale_evictions_total",
        "prsm_bootstrap_reconnect_attempts_total",
        "prsm_bootstrap_reconnect_successes_total",
    ):
        assert f"# TYPE {name} counter" in body


def test_metrics_emits_bootstrap_gauges_when_discovery_present():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ["wss://"],
        "discovered_peer_count": 7,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/metrics").text

    # connected gauge
    assert "prsm_bootstrap_connected 1" in body
    assert "# TYPE prsm_bootstrap_connected gauge" in body
    # discovered_peer_count gauge
    assert "prsm_bootstrap_discovered_peer_count 7" in body
    # degraded gauge — boolean rendered as 0
    assert "prsm_bootstrap_degraded 0" in body


def test_metrics_degraded_gauge_renders_true_as_one():
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
    body = client.get("/metrics").text
    assert "prsm_bootstrap_degraded 1" in body
    assert "prsm_bootstrap_connected 0" in body


def test_metrics_omits_bootstrap_block_when_no_discovery():
    """No discovery wired → no bootstrap metrics emitted, BUT
    /metrics still returns 200 with other gauges (per /metrics
    fail-soft semantics)."""
    client = _make_client(discovery=None)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "prsm_bootstrap_peer_join_events_total" not in resp.text
    # Node-up gauge still emitted
    assert "prsm_node_up 1" in resp.text


def test_metrics_fail_soft_when_get_bootstrap_status_raises():
    """If get_bootstrap_status raises, the bootstrap block is
    omitted but /metrics still returns 200 — fail-soft per the
    other /metrics probes."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(
        side_effect=RuntimeError("boom"),
    )
    client = _make_client(discovery=discovery)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "prsm_bootstrap_peer_join_events_total" not in resp.text
    assert "prsm_node_up 1" in resp.text


# ── Sprint 377 — multi-bootstrap Prometheus surface ──


def test_metrics_emits_active_url_gauge_with_label():
    """Sprint 377: active_url surfaces as a labeled gauge.
    Prometheus pattern for current-string-state: emit value 1
    with the string as a label. Operators can then `count by
    (url)` across the fleet to graph bootstrap distribution.
    """
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 2, "connected": 1, "degraded": False,
        "bootstrap_nodes": [
            "wss://bootstrap1.prsm-network.com:8765",
        ],
        "bootstrap_fallback_nodes": [
            "wss://bootstrap-eu.prsm-network.com:8765",
        ],
        "bootstrap_fallback_enabled": True,
        "active_url": (
            "wss://bootstrap-eu.prsm-network.com:8765"
        ),
        "discovered_peer_count": 3,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/metrics").text
    # Labeled gauge — operators alert on absence of "bootstrap1"
    # label across the fleet (= primary down).
    assert (
        'prsm_bootstrap_active{url="'
        'wss://bootstrap-eu.prsm-network.com:8765"} 1'
    ) in body


def test_metrics_emits_fallback_enabled_gauge():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ["wss://"],
        "bootstrap_fallback_nodes": [],
        "bootstrap_fallback_enabled": True,
        "active_url": "wss://",
        "discovered_peer_count": 1,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/metrics").text
    assert "prsm_bootstrap_fallback_enabled 1" in body


def test_metrics_fallback_enabled_zero_when_disabled():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ["wss://"],
        "bootstrap_fallback_nodes": [],
        "bootstrap_fallback_enabled": False,
        "active_url": "wss://",
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/metrics").text
    assert "prsm_bootstrap_fallback_enabled 0" in body


def test_metrics_omits_active_url_when_none():
    """When active_url is None (all candidates failed), the
    labeled gauge is omitted — Prometheus absence is the
    canonical 'no current value' signal, much cleaner than
    emitting an empty-label gauge."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 2, "connected": 0, "degraded": True,
        "bootstrap_nodes": ["wss://"],
        "bootstrap_fallback_nodes": ["wss://eu/"],
        "bootstrap_fallback_enabled": True,
        "active_url": None,
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 2,
        "reconnect_successes": 0, "client_state": "dead",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/metrics").text
    assert "prsm_bootstrap_active{" not in body
    # Degraded gauge still fires
    assert "prsm_bootstrap_degraded 1" in body


def test_metrics_active_url_label_escapes_quotes():
    """Defensive: if active_url contains a quote character
    (shouldn't, but be safe), it must be escaped to keep the
    Prometheus exposition format valid."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ['wss://he"llo:8765'],
        "bootstrap_fallback_nodes": [],
        "bootstrap_fallback_enabled": True,
        "active_url": 'wss://he"llo:8765',
        "discovered_peer_count": 0,
        "peer_join_events": 0, "peer_leave_events": 0,
        "stale_evictions": 0, "reconnect_attempts": 0,
        "reconnect_successes": 0, "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    resp = client.get("/metrics")
    # Endpoint stays 200 + label-value escapes the embedded
    # quote (Prometheus exposition format: \" is the escape).
    assert resp.status_code == 200
    assert r'\"' in resp.text or 'he\\"llo' in resp.text
