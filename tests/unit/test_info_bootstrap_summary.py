"""Sprint 327 — Surface compact bootstrap summary on /info.

Operators' canonical quick-check is `/info`. Pre-sprint-327 it
surfaced node_id, api_version, operator_address, network, chain
id, canonical_addresses, agent_forge_wired/state — but NOT
bootstrap connectivity state. So "is this node actually
discovering peers?" required a separate /bootstrap/status hit.

Sprint 327 adds a compact `bootstrap` subdict surfacing the
most operator-relevant signals from sprint 324's telemetry:
  - client_state (most important — instant triage)
  - connected (0 or 1 — are we registered?)
  - degraded (boolean — alarm signal)
  - discovered_peer_count (size of visible network)

Full sprint-324 cumulative counters stay on /bootstrap/status
for deep diagnosis; /info gets the digest.

Fail-soft: if discovery isn't initialized, the `bootstrap`
key is omitted (same pattern as `canonical_addresses` and
`operator_address`).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_client(*, discovery=None):
    """Build a TestClient against create_api_app with a minimal
    node stub and optional discovery object."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.identity.node_id = "n-test"
    node.discovery = discovery
    node.ftns_ledger = None
    node._operator_address = None
    node.agent_forge = None
    node._query_orchestrator_state = None
    node._query_orchestrator_error = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_info_surfaces_bootstrap_when_discovery_present():
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
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
    })
    client = _make_client(discovery=discovery)
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "bootstrap" in body
    bs = body["bootstrap"]
    # Compact digest — must include these 4 signals
    assert bs["client_state"] == "connected"
    assert bs["connected"] == 1
    assert bs["degraded"] is False
    assert bs["discovered_peer_count"] == 3


def test_info_omits_bootstrap_when_no_discovery():
    """If discovery isn't initialized, the bootstrap key is
    omitted — operators see absence as 'discovery not wired'
    rather than confusing zero/false values."""
    client = _make_client(discovery=None)
    resp = client.get("/info")
    body = resp.json()
    assert "bootstrap" not in body


def test_info_fail_soft_when_get_bootstrap_status_raises():
    """If get_bootstrap_status raises, /info must still return
    200 with the rest of the body — discovery failure
    shouldn't 500 the quick-check endpoint."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(
        side_effect=RuntimeError("discovery boom"),
    )
    client = _make_client(discovery=discovery)
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    # bootstrap key absent (fail-soft) but rest of body present
    assert "bootstrap" not in body
    assert body["node_id"] == "n-test"


def test_info_does_not_leak_full_counters_in_bootstrap_subdict():
    """The /info `bootstrap` subdict is intentionally COMPACT —
    full sprint-324 counters belong on /bootstrap/status only.
    Including all of them on /info would bloat the quick-check
    endpoint that integration code polls frequently."""
    discovery = MagicMock()
    discovery.get_bootstrap_status = MagicMock(return_value={
        "attempted": 1, "connected": 1, "degraded": False,
        "bootstrap_nodes": ["wss://"],
        "discovered_peer_count": 0,
        "peer_join_events": 99, "peer_leave_events": 88,
        "stale_evictions": 77, "reconnect_attempts": 66,
        "reconnect_successes": 55,
        "client_state": "connected",
    })
    client = _make_client(discovery=discovery)
    body = client.get("/info").json()
    bs = body["bootstrap"]
    # Cumulative counters absent — they live on /bootstrap/status
    for k in (
        "peer_join_events", "peer_leave_events",
        "stale_evictions", "reconnect_attempts",
        "reconnect_successes",
    ):
        assert k not in bs, (
            f"/info bootstrap subdict must not include "
            f"cumulative counter {k} — keep it compact"
        )


def test_info_bootstrap_includes_degraded_when_degraded():
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
    body = client.get("/info").json()
    bs = body["bootstrap"]
    assert bs["degraded"] is True
    assert bs["client_state"] == "dead"
    assert bs["connected"] == 0
