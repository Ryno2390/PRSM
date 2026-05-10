"""Sprint 165 — periodic peer-list polling against the bootstrap
server keeps known_peers fresh after initial registration.

Sprint 164 closed the connectivity gap: nodes register through
the BootstrapClient WebSocket fallback. But the BootstrapClient
only returns peers in the synchronous `register_ack` response.
After that, no further peer updates flow — a node that registers
when alone stays alone forever, even when other operators come
online later.

Sprint 165 adds a periodic poll: every PRSM_BOOTSTRAP_POLL_SECONDS
(default 60), Libp2pDiscovery calls client.get_peers() and hydrates
the local capability index with any newly-registered nodes.

Note: tests/conftest.py has an autouse `mock_asyncio_sleep` fixture
that replaces asyncio.sleep with a no-op. That breaks any test
that depends on real-time poll-loop ticks. We test the hydration
+ task-creation invariants directly via the helper method
`_hydrate_peers_from_bootstrap` and via attribute inspection,
which together cover the same behavior. Live verification of
end-to-end polling is performed via the dogfood script.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.bootstrap.client import BootstrapPeer
from prsm.node.libp2p_discovery import Libp2pDiscovery


def _bp(peer_id, port=9001, address="example.com"):
    return BootstrapPeer(
        peer_id=peer_id, address=address,
        port=port, capabilities=[], region=None, version=None,
    )


def _make_discovery(self_node_id="self-node"):
    transport = MagicMock()
    transport.identity.node_id = self_node_id
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)
    return Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://bootstrap.example.com:8765"],
    )


# ──────────────────────────────────────────────────────────────────────
# Hydration helper (the heart of the poll-loop tick)
# ──────────────────────────────────────────────────────────────────────


def test_hydrate_skips_self_node():
    """Self peer_id must NOT enter capability_index — would create
    self-edges in the capability graph."""
    d = _make_discovery(self_node_id="self-node")
    d._hydrate_peers_from_bootstrap([
        _bp("self-node"),
        _bp("other-node"),
    ])
    peer_ids = {p.node_id for p in d.get_known_peers()}
    assert "self-node" not in peer_ids
    assert "other-node" in peer_ids


def test_hydrate_adds_multiple_peers():
    """Sprint 165 — multiple peers from one tick all enter the index."""
    d = _make_discovery()
    d._hydrate_peers_from_bootstrap([
        _bp("peer-A", port=9001, address="a.example.com"),
        _bp("peer-B", port=9002, address="b.example.com"),
    ])
    known = {p.node_id: p for p in d.get_known_peers()}
    assert set(known.keys()) == {"peer-A", "peer-B"}
    assert known["peer-A"].address == "a.example.com:9001"
    assert known["peer-B"].address == "b.example.com:9002"


def test_hydrate_skips_empty_peer_id():
    """Defensive — bootstrap peer payloads with no peer_id are
    skipped, not added with empty key."""
    d = _make_discovery()
    d._hydrate_peers_from_bootstrap([
        _bp(""),
        _bp("real-peer"),
    ])
    peer_ids = {p.node_id for p in d.get_known_peers()}
    assert "" not in peer_ids
    assert "real-peer" in peer_ids


def test_hydrate_tracks_discovered_peer_count():
    """Sprint 167 — _bootstrap_status.discovered_peer_count
    reflects the count from the most recent poll, excluding self."""
    d = _make_discovery(self_node_id="self-node")
    d._hydrate_peers_from_bootstrap([
        _bp("peer-A"),
        _bp("peer-B"),
        _bp("self-node"),  # excluded
    ])
    assert d._bootstrap_status["discovered_peer_count"] == 2
    status = d.get_bootstrap_status()
    assert status["discovered_peer_count"] == 2

    # Second poll with fewer peers — count drops.
    d._hydrate_peers_from_bootstrap([_bp("peer-A")])
    assert d.get_bootstrap_status()["discovered_peer_count"] == 1


def test_initial_discovered_peer_count_zero():
    """Before any poll, the count is 0."""
    d = _make_discovery()
    assert d.get_bootstrap_status()["discovered_peer_count"] == 0


def test_hydrate_repeated_call_updates_existing_entry():
    """Sprint 165 — re-hydrating the same peer refreshes its
    last_seen timestamp (so liveness tracking works)."""
    d = _make_discovery()
    d._hydrate_peers_from_bootstrap([_bp("peer-X", port=9001)])
    first_seen = d._capability_index["peer-X"].last_seen
    # Force a noticeable timestamp difference by sleeping the
    # equivalent of a poll interval. Use real-time time.sleep
    # which is mocked to a no-op by conftest, so we directly
    # mutate the timestamp instead.
    d._capability_index["peer-X"].last_seen = first_seen - 100.0
    d._hydrate_peers_from_bootstrap([_bp("peer-X", port=9001)])
    second_seen = d._capability_index["peer-X"].last_seen
    assert second_seen > (first_seen - 100.0)


# ──────────────────────────────────────────────────────────────────────
# Poll-task wiring (sprint 165 invariant)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_fallback_creates_poll_task():
    """Sprint 165 invariant — successful BootstrapClient registration
    creates _bootstrap_poll_task on the discovery instance."""

    class _Stub:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            return []

        async def start_heartbeat(self):
            pass

        async def get_peers(self):
            return []

        async def disconnect(self):
            pass

    d = _make_discovery()
    with patch("prsm.bootstrap.client.BootstrapClient", _Stub):
        connected = await d.bootstrap()
    assert connected == 1
    assert getattr(d, "_bootstrap_poll_task", None) is not None
    assert not d._bootstrap_poll_task.done()
    # Cleanup so other tests don't see a dangling task.
    d._bootstrap_poll_task.cancel()
    try:
        await d._bootstrap_poll_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_stop_cancels_poll_task():
    """Sprint 165 — stop() must cancel the poll task (otherwise
    test/lifecycle teardown leaks tasks)."""

    class _Stub:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            return []

        async def start_heartbeat(self):
            pass

        async def get_peers(self):
            return []

        async def disconnect(self):
            pass

    d = _make_discovery()
    with patch("prsm.bootstrap.client.BootstrapClient", _Stub):
        await d.bootstrap()
    task = d._bootstrap_poll_task
    await d.stop()
    assert task.done()


@pytest.mark.asyncio
async def test_stop_disconnects_bootstrap_client():
    """Sprint 166 — stop() must call client.disconnect() so the
    bootstrap-server WebSocket closes cleanly. Pre-fix the
    poll task was cancelled but the WebSocket leaked."""
    disconnect_calls = {"n": 0}

    class _TrackingClient:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            return []

        async def start_heartbeat(self):
            pass

        async def get_peers(self):
            return []

        async def disconnect(self):
            disconnect_calls["n"] += 1

    d = _make_discovery()
    with patch("prsm.bootstrap.client.BootstrapClient", _TrackingClient):
        await d.bootstrap()
    await d.stop()
    assert disconnect_calls["n"] == 1
    assert d._bootstrap_client is None


@pytest.mark.asyncio
async def test_stop_tolerates_disconnect_raise():
    """Sprint 166 — disconnect() raising shouldn't break shutdown
    or leak the poll task."""

    class _BrokenDisconnectClient:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            return []

        async def start_heartbeat(self):
            pass

        async def get_peers(self):
            return []

        async def disconnect(self):
            raise RuntimeError("simulated socket already closed")

    d = _make_discovery()
    with patch("prsm.bootstrap.client.BootstrapClient",
               _BrokenDisconnectClient):
        await d.bootstrap()
    task = d._bootstrap_poll_task
    # Should NOT propagate the RuntimeError.
    await d.stop()
    assert task.done()
    assert d._bootstrap_client is None


@pytest.mark.asyncio
async def test_no_poll_task_when_fallback_skipped():
    """Sprint 165 — when libp2p path succeeds, no fallback fires
    and no poll task is created."""
    d = _make_discovery()
    d.transport.connect_to_peer = AsyncMock(return_value=MagicMock())
    connected = await d.bootstrap()
    assert connected == 1
    assert getattr(d, "_bootstrap_poll_task", None) is None


# ──────────────────────────────────────────────────────────────────────
# Resilience: get_peers failure must not propagate or kill the loop
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_loop_resilient_to_get_peers_raise():
    """Sprint 165 — get_peers() raising during a tick is logged at
    debug and the loop continues. We exercise this by invoking
    `_bootstrap_poll_loop` directly with a mocked client whose
    get_peers raises, and asserting the loop is cancellable cleanly
    without propagating the exception.
    """
    d = _make_discovery()

    class _BrokenClient:
        async def get_peers(self):
            raise RuntimeError("transient")

    d._bootstrap_client = _BrokenClient()
    task = asyncio.create_task(d._bootstrap_poll_loop())
    # Yield once to let the loop reach its first asyncio.sleep
    # (mocked to no-op by conftest), then cancel.
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # No exception propagated past the cancel — the loop swallowed
    # the RuntimeError as designed.
