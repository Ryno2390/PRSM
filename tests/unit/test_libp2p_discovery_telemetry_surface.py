"""Sprint 324 — telemetry surface for the P2P discovery layer.

Sprints 319-323 added significant new state to
`Libp2pDiscovery`:
  - Reconnect attempts + successes (321)
  - Stale-peer evictions (323)
  - peer_join / peer_leave announcement counts (320)
  - Sentinel state — is the client currently dead? (321)

None of this was visible on `/bootstrap/status`. Operators
trying to diagnose a misbehaving node had to attach a
debugger or scrape logs. Sprint 324 threads counters into
`_bootstrap_status` at the existing increment sites and
extends `get_bootstrap_status()` to surface them.

Cumulative counters never reset within a process lifetime —
operators reading them at T₂ minus T₁ get a rate. Reset
only on process restart (which is the natural reset boundary
since the discovery layer is recreated).
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

from prsm.bootstrap.client import BootstrapPeer
from prsm.node.discovery import PeerInfo
from prsm.node.libp2p_discovery import (
    Libp2pDiscovery,
    _DeadBootstrapSentinel,
)


def _make_discovery():
    transport = MagicMock()
    transport.identity.node_id = "self-node"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)
    return Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://example.com:8765"],
    )


# ── New counters appear on get_bootstrap_status() ──────────


def test_default_counters_zero_at_init():
    d = _make_discovery()
    status = d.get_bootstrap_status()
    for k in (
        "peer_join_events",
        "peer_leave_events",
        "stale_evictions",
        "reconnect_attempts",
        "reconnect_successes",
    ):
        assert status[k] == 0, f"{k} should be 0 at init"


def test_client_state_field_reflects_sentinel():
    """`client_state` must surface whether the
    BootstrapClient slot holds a live client, the dead
    sentinel, or is None entirely. Operators triage on this
    field."""
    d = _make_discovery()
    # No client installed
    assert d.get_bootstrap_status()["client_state"] == "none"

    # Sentinel installed (reconnect failed)
    d._bootstrap_client = _DeadBootstrapSentinel()
    assert d.get_bootstrap_status()["client_state"] == "dead"

    # Live client
    live = MagicMock()
    live.is_connected = True
    d._bootstrap_client = live
    assert d.get_bootstrap_status()["client_state"] == "connected"

    # Live-but-disconnected client (heartbeat detected drop)
    half = MagicMock()
    half.is_connected = False
    d._bootstrap_client = half
    assert d.get_bootstrap_status()["client_state"] == "disconnected"


# ── peer_join / peer_leave counters increment ──────────────


def test_peer_join_announcement_increments_counter():
    d = _make_discovery()
    client = MagicMock()
    client._observed_announcements = [
        {"announcement_type": "peer_join", "peer_id": "p1",
         "peer_endpoint": "10.0.0.1:9001"},
        {"announcement_type": "peer_join", "peer_id": "p2",
         "peer_endpoint": "10.0.0.2:9001"},
    ]
    d._consume_bootstrap_announcements(client)
    status = d.get_bootstrap_status()
    assert status["peer_join_events"] == 2


def test_peer_leave_announcement_increments_counter():
    d = _make_discovery()
    d._capability_index["p1"] = PeerInfo(
        node_id="p1", address="x:1", last_seen=time.time(),
    )
    client = MagicMock()
    client._observed_announcements = [
        {"announcement_type": "peer_leave", "peer_id": "p1"},
    ]
    d._consume_bootstrap_announcements(client)
    assert d.get_bootstrap_status()["peer_leave_events"] == 1


def test_peer_event_counters_accumulate_across_calls():
    d = _make_discovery()
    client = MagicMock()
    client._observed_announcements = [
        {"announcement_type": "peer_join", "peer_id": "a",
         "peer_endpoint": "1:1"},
    ]
    d._consume_bootstrap_announcements(client)
    # Re-fill with another join
    client._observed_announcements = [
        {"announcement_type": "peer_join", "peer_id": "b",
         "peer_endpoint": "1:2"},
        {"announcement_type": "peer_leave", "peer_id": "a"},
    ]
    d._consume_bootstrap_announcements(client)
    s = d.get_bootstrap_status()
    assert s["peer_join_events"] == 2
    assert s["peer_leave_events"] == 1


# ── stale_evictions counter ───────────────────────────────


def test_stale_evictions_counter_tracks_sweep_result():
    d = _make_discovery()
    now = time.time()
    for i in range(3):
        d._capability_index[f"s{i}"] = PeerInfo(
            node_id=f"s{i}", address="x:1",
            last_seen=now - 9999.0,
        )
    d._sweep_stale_peers(threshold_seconds=60.0)
    assert d.get_bootstrap_status()["stale_evictions"] == 3
    # Subsequent no-op sweep leaves counter alone
    d._sweep_stale_peers(threshold_seconds=60.0)
    assert d.get_bootstrap_status()["stale_evictions"] == 3
