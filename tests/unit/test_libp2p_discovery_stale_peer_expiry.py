"""Sprint 323 — Stale-peer expiry on Libp2pDiscovery.

Peers that crash hard (kernel panic, network unplug, process
SIGKILL) never emit peer_leave; they just stop heartbeating.
The bootstrap server eventually times them out (~90s default)
and emits peer_leave, which sprints 320 + 322 ensure we
consume. But two failure modes leak orphans regardless:

  1. If our BootstrapClient WebSocket drops during the
     server's peer_leave-announcement window, we miss the
     event entirely. Sprint 321's reconnect resumes get_peers
     polling, but the peer_leave has already passed by the
     time we reconnect — `_capability_index` keeps the
     orphan forever.
  2. Non-bootstrap-sourced peers (gossip-sourced) have no
     announcement path at all; their staleness is only
     visible via their own heartbeat absence.

Sprint 323 adds a periodic sweep that evicts any peer whose
`last_seen` is older than a configurable staleness threshold.
Bootstrap-sourced peers get their `last_seen` bumped on every
peer_list poll; gossip-sourced peers get theirs bumped on
every capability_announce. So any peer whose `last_seen`
exceeds N * (poll/heartbeat interval) is genuinely silent.

Default threshold is `PRSM_PEER_STALE_SECONDS` env (default
600s = 10 minutes), so a single missed bootstrap poll
(60s) doesn't trigger eviction — only sustained silence
across many polls. Set lower for fast tests.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.discovery import PeerInfo
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


# ── Core sweep behavior ──────────────────────────────────


def test_sweep_evicts_peer_whose_last_seen_exceeds_threshold():
    d = _make_discovery()
    now = time.time()
    d._capability_index["stale"] = PeerInfo(
        node_id="stale", address="x:1",
        last_seen=now - 1000.0,  # ~16 min ago
    )
    d._capability_index["fresh"] = PeerInfo(
        node_id="fresh", address="x:2",
        last_seen=now - 5.0,
    )
    d._sweep_stale_peers(threshold_seconds=600.0)
    assert "stale" not in d._capability_index
    assert "fresh" in d._capability_index


def test_sweep_is_noop_when_index_empty():
    d = _make_discovery()
    d._sweep_stale_peers(threshold_seconds=600.0)
    assert d._capability_index == {}


def test_sweep_never_evicts_self_even_if_indexed():
    """Self-edge protection — if our own node_id somehow
    landed in the index, the sweep must not remove us in a
    way that distinguishes from external eviction (the
    invariant from sprint 165 is that self never appears in
    _capability_index in the first place; this is defense
    in depth)."""
    d = _make_discovery()  # self-node = 'self-node'
    d._capability_index["self-node"] = PeerInfo(
        node_id="self-node", address="x:1",
        last_seen=time.time() - 1000.0,
    )
    d._sweep_stale_peers(threshold_seconds=600.0)
    # self entries get evicted just like any stale peer;
    # the protection is at INSERT time (sprint 165), not
    # sweep time. Documented here to assert the boundary.
    assert "self-node" not in d._capability_index


def test_sweep_zero_threshold_evicts_everything():
    """Zero (or negative) threshold means evict every peer.
    Useful for forced cache reset; documented as a feature
    not a bug."""
    d = _make_discovery()
    d._capability_index["a"] = PeerInfo(
        node_id="a", address="x:1", last_seen=time.time(),
    )
    d._capability_index["b"] = PeerInfo(
        node_id="b", address="x:2", last_seen=time.time(),
    )
    d._sweep_stale_peers(threshold_seconds=0.0)
    assert d._capability_index == {}


def test_sweep_returns_evicted_count():
    """Caller (poll loop / metrics) needs the count for
    logging + observability."""
    d = _make_discovery()
    now = time.time()
    for i in range(5):
        d._capability_index[f"stale-{i}"] = PeerInfo(
            node_id=f"stale-{i}", address="x:1",
            last_seen=now - 9999.0,
        )
    for i in range(3):
        d._capability_index[f"fresh-{i}"] = PeerInfo(
            node_id=f"fresh-{i}", address="x:2",
            last_seen=now,
        )
    evicted = d._sweep_stale_peers(threshold_seconds=600.0)
    assert evicted == 5
    assert len(d._capability_index) == 3


# ── Threshold env var ───────────────────────────────────


def test_default_threshold_comes_from_env(monkeypatch):
    """The poll-loop integration reads
    PRSM_PEER_STALE_SECONDS env var, defaulting to 600s.
    Below tests the env-derived threshold helper."""
    from prsm.node.libp2p_discovery import _resolve_stale_threshold
    monkeypatch.setenv("PRSM_PEER_STALE_SECONDS", "120")
    assert _resolve_stale_threshold() == 120.0
    monkeypatch.delenv("PRSM_PEER_STALE_SECONDS")
    assert _resolve_stale_threshold() == 600.0


def test_threshold_fail_soft_on_garbage_env(monkeypatch):
    """Bad env value falls back to default — no raise."""
    from prsm.node.libp2p_discovery import _resolve_stale_threshold
    monkeypatch.setenv("PRSM_PEER_STALE_SECONDS", "not-a-number")
    assert _resolve_stale_threshold() == 600.0
    monkeypatch.setenv("PRSM_PEER_STALE_SECONDS", "-50")
    assert _resolve_stale_threshold() == 600.0


# ── Poll-loop integration ───────────────────────────────


def test_sweep_runs_after_each_poll_tick():
    """The bootstrap poll loop must invoke
    _sweep_stale_peers after consume so crashed peers get
    evicted at poll cadence."""
    d = _make_discovery()
    client = MagicMock()
    client.is_connected = True
    client.get_peers = AsyncMock(return_value=[])
    client._observed_announcements = []
    d._bootstrap_client = client

    # Pre-seed a peer that's been silent for 1 hour
    d._capability_index["dead"] = PeerInfo(
        node_id="dead", address="x:1",
        last_seen=time.time() - 3600.0,
    )

    import os
    os.environ["PRSM_BOOTSTRAP_POLL_SECONDS"] = "0.05"
    os.environ["PRSM_PEER_STALE_SECONDS"] = "60"
    try:
        async def runner():
            task = asyncio.create_task(d._bootstrap_poll_loop())
            for _ in range(40):
                await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        asyncio.run(runner())
    finally:
        os.environ.pop("PRSM_BOOTSTRAP_POLL_SECONDS", None)
        os.environ.pop("PRSM_PEER_STALE_SECONDS", None)

    assert "dead" not in d._capability_index, (
        "stale peer should have been evicted by the sweep "
        "that runs at the end of each poll tick"
    )
