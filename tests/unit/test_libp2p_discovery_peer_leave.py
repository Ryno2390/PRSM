"""Sprint 320 — Libp2pDiscovery must consume `peer_leave` announcements.

Sprint 319 plumbed BootstrapClient to buffer server-pushed
announcements into `_observed_announcements`. Sprint 320 closes
the loop: Libp2pDiscovery consumes that buffer on every poll
tick so:

  - `peer_join` announcements eagerly hydrate `_capability_index`
    (faster than waiting for the next poll's peer_list).
  - `peer_leave` announcements REMOVE the departed peer from
    `_capability_index` — without this, departed peers stick
    around forever as stale entries because
    `_hydrate_peers_from_bootstrap` only ADDS, never REMOVES.

The peer_leave gap was the load-bearing finding — pre-fix every
operator's `_capability_index` grew monotonically with every
peer who ever registered, accumulating dead entries that
downstream code (find_peers_by_capability, find_peers_with_gpu)
would surface to QueryOrchestrator-style consumers as live
candidates.

Live dogfood 2026-05-12 confirmed the canonical bootstrap server
emits both `peer_join` and `peer_leave` announcements with
`announcement_type` field on each.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_discovery():
    transport = MagicMock()
    transport.identity.node_id = "self-node"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)
    return Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=[],
    )


def _fake_client(announcements):
    """Build a stub BootstrapClient surrogate exposing only the
    surface Libp2pDiscovery's announcement consumer touches."""
    c = MagicMock()
    c._observed_announcements = list(announcements)
    return c


# ── peer_leave eviction ───────────────────────────────────


def test_peer_leave_evicts_peer_from_capability_index():
    """Departed peer must be removed from _capability_index.
    Pre-sprint-320 the index only grew — peer_leave was a no-op.
    """
    d = _make_discovery()
    # Pre-seed the index with a peer we will then evict
    from prsm.node.discovery import PeerInfo
    d._capability_index["dead-peer"] = PeerInfo(
        node_id="dead-peer",
        address="10.0.0.5:9999",
        last_seen=time.time(),
    )
    assert "dead-peer" in d._capability_index

    client = _fake_client([
        {
            "announcement_id": "a1",
            "announcement_type": "peer_leave",
            "peer_id": "dead-peer",
            "peer_endpoint": "10.0.0.5:9999",
        },
    ])
    d._consume_bootstrap_announcements(client)

    assert "dead-peer" not in d._capability_index


def test_peer_leave_is_idempotent_when_peer_already_absent():
    """A peer_leave for a peer we never knew about must not raise."""
    d = _make_discovery()
    client = _fake_client([
        {
            "announcement_id": "a1",
            "announcement_type": "peer_leave",
            "peer_id": "never-seen",
        },
    ])
    d._consume_bootstrap_announcements(client)  # no exception
    assert "never-seen" not in d._capability_index


def test_announcement_buffer_cleared_after_consume():
    """After consume, the client's announcement buffer must be
    emptied so the next poll tick doesn't re-process old events.
    """
    d = _make_discovery()
    client = _fake_client([
        {"announcement_type": "peer_join", "peer_id": "x",
         "peer_endpoint": "1.2.3.4:9000"},
        {"announcement_type": "peer_leave", "peer_id": "x"},
    ])
    d._consume_bootstrap_announcements(client)
    assert client._observed_announcements == []


# ── peer_join hydration ───────────────────────────────────


def test_peer_join_adds_peer_to_capability_index():
    """A peer_join announcement eagerly populates the index
    without waiting for the next bootstrap poll."""
    d = _make_discovery()
    client = _fake_client([
        {
            "announcement_id": "a1",
            "announcement_type": "peer_join",
            "peer_id": "new-peer",
            "peer_endpoint": "10.0.0.7:9001",
        },
    ])
    d._consume_bootstrap_announcements(client)
    assert "new-peer" in d._capability_index
    pi = d._capability_index["new-peer"]
    assert "10.0.0.7" in pi.address
    assert "9001" in pi.address


def test_peer_join_for_self_is_ignored():
    """peer_join announcements for our own node_id must not
    pollute the index (self-edge avoidance, sprint 165 invariant)."""
    d = _make_discovery()  # self-node = 'self-node'
    client = _fake_client([
        {
            "announcement_type": "peer_join",
            "peer_id": "self-node",
            "peer_endpoint": "127.0.0.1:9001",
        },
    ])
    d._consume_bootstrap_announcements(client)
    assert "self-node" not in d._capability_index


def test_unknown_announcement_type_is_ignored():
    """Forward-compat: unknown announcement types do not raise."""
    d = _make_discovery()
    client = _fake_client([
        {"announcement_type": "system_notice", "msg": "hi"},
        {"announcement_type": "future_event_v3", "peer_id": "x"},
    ])
    d._consume_bootstrap_announcements(client)  # no exception


def test_malformed_announcement_does_not_raise():
    """A malformed announcement (missing fields) must be
    skipped so one bad entry doesn't poison the whole batch."""
    d = _make_discovery()
    client = _fake_client([
        {"announcement_type": "peer_join"},  # no peer_id
        {"announcement_type": "peer_join", "peer_id": "good",
         "peer_endpoint": "10.0.0.1:9001"},
    ])
    d._consume_bootstrap_announcements(client)
    assert "good" in d._capability_index


# ── Wiring into poll loop ─────────────────────────────────


def test_consume_runs_after_poll_tick():
    """Sprint 320 — the bootstrap poll loop must invoke the
    announcement consumer alongside its peer_list hydration so
    operators see joins/leaves at poll cadence rather than never.
    """
    d = _make_discovery()
    fake_client = _fake_client([
        {"announcement_type": "peer_leave", "peer_id": "gone"},
    ])
    fake_client.get_peers = AsyncMock(return_value=[])
    d._bootstrap_client = fake_client

    # Pre-seed a peer we expect to be evicted by the poll tick
    from prsm.node.discovery import PeerInfo
    d._capability_index["gone"] = PeerInfo(
        node_id="gone", address="x:1", last_seen=time.time(),
    )

    # Run one poll iteration by hand. The conftest autouse
    # fixture replaces asyncio.sleep with an instant-yield
    # stub so we can't drive the loop via wall-time sleeps
    # here — instead we hand-yield enough times to step the
    # task through `sleep → get_peers → consume`.
    import os
    os.environ["PRSM_BOOTSTRAP_POLL_SECONDS"] = "0.05"
    try:
        async def runner():
            task = asyncio.create_task(d._bootstrap_poll_loop())
            # Give the task ample scheduler ticks to step
            # through its first iteration (sleep → get_peers
            # → hydrate → consume).
            for _ in range(20):
                await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        asyncio.run(runner())
    finally:
        os.environ.pop("PRSM_BOOTSTRAP_POLL_SECONDS", None)

    assert "gone" not in d._capability_index, (
        "peer_leave announcement should have evicted 'gone' "
        "during the poll tick"
    )
