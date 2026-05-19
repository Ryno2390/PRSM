"""Sprint 573 — auto-dial sweep after bootstrap hydration.

Sprint 569 shipped POST /peers/connect as the manual operator
trigger to turn a bootstrap-known peer into a connected peer.
Sprint 567 explicitly named the follow-on: "auto-dial sweep" so
operators don't have to call /peers/connect for every known peer
after their daemon boots.

Pre-sprint-573 flow:
  1. Daemon boots, BootstrapClient.connect() returns peer_list
  2. _try_bootstrap_client populates self.known_peers from the list
  3. ...nothing happens. transport.peers stays empty.
  4. Operator must manually POST /peers/connect for each known peer.

Sprint 573 wires a background auto-dial sweep that fires after
bootstrap hydration completes. Each known peer that isn't already
in transport.peers gets a connect_to_peer call. Failures are
logged but don't break the loop (peer might be NAT'd, down, etc.).

Invariants this sweep MUST satisfy:
- Skip self (transport.identity.node_id)
- Skip already-connected peers (transport.peers.keys())
- Skip peers with 0.0.0.0 addresses (defensive, sprint-570 F28 era)
- Best-effort: each connect runs in its own try/except
- Bounded: don't fan out unbounded; cap at target_peers (default
  is small)
- Async: each dial is non-blocking; sweep is fire-and-forget from
  the caller's perspective, though tests want awaitable
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_peer_discovery_with(transport, known_peers):
    from prsm.node.discovery import PeerDiscovery, PeerInfo

    d = PeerDiscovery(transport=transport, bootstrap_nodes=[])
    d._local_capabilities = []
    d._local_backends = []
    d._local_gpu_available = False
    for pid, addr in known_peers.items():
        d.known_peers[pid] = PeerInfo(
            node_id=pid,
            address=addr,
            capabilities=[],
        )
    return d


@pytest.mark.asyncio
async def test_auto_dial_sweep_dials_each_unconnected_known_peer():
    """Every known peer that isn't already in transport.peers
    must get a transport.connect_to_peer call.
    """
    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.peers = {}
    transport.connect_to_peer = AsyncMock(return_value=MagicMock())

    d = _make_peer_discovery_with(
        transport=transport,
        known_peers={
            "peer-a": "1.1.1.1:9001",
            "peer-b": "2.2.2.2:9001",
            "peer-c": "3.3.3.3:9001",
        },
    )

    await d._auto_dial_sweep()

    addrs = {c.args[0] for c in transport.connect_to_peer.await_args_list}
    assert addrs == {"1.1.1.1:9001", "2.2.2.2:9001", "3.3.3.3:9001"}


@pytest.mark.asyncio
async def test_auto_dial_sweep_skips_self():
    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.peers = {}
    transport.connect_to_peer = AsyncMock(return_value=MagicMock())

    d = _make_peer_discovery_with(
        transport=transport,
        known_peers={
            "self-node": "5.5.5.5:9001",
            "peer-a": "1.1.1.1:9001",
        },
    )

    await d._auto_dial_sweep()

    dialed = {c.args[0] for c in transport.connect_to_peer.await_args_list}
    assert dialed == {"1.1.1.1:9001"}, (
        "Auto-dial must never dial self — would create self-loop"
    )


@pytest.mark.asyncio
async def test_auto_dial_sweep_skips_already_connected():
    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.peers = {"peer-a": MagicMock()}  # already connected
    transport.connect_to_peer = AsyncMock(return_value=MagicMock())

    d = _make_peer_discovery_with(
        transport=transport,
        known_peers={
            "peer-a": "1.1.1.1:9001",
            "peer-b": "2.2.2.2:9001",
        },
    )

    await d._auto_dial_sweep()

    dialed = {c.args[0] for c in transport.connect_to_peer.await_args_list}
    assert dialed == {"2.2.2.2:9001"}, (
        "Auto-dial must skip peers already in transport.peers — "
        "redundant dials waste resources + can disrupt live conn"
    )


@pytest.mark.asyncio
async def test_auto_dial_sweep_skips_0_0_0_0_addresses():
    """Defense-in-depth for legacy pre-sprint-570 peers that may
    still have polluted known_peers entries.
    """
    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.peers = {}
    transport.connect_to_peer = AsyncMock(return_value=MagicMock())

    d = _make_peer_discovery_with(
        transport=transport,
        known_peers={
            "peer-bogus": "0.0.0.0:9001",
            "peer-empty": ":9001",
            "peer-real": "1.1.1.1:9001",
        },
    )

    await d._auto_dial_sweep()

    dialed = {c.args[0] for c in transport.connect_to_peer.await_args_list}
    assert dialed == {"1.1.1.1:9001"}


@pytest.mark.asyncio
async def test_auto_dial_sweep_continues_after_single_failure():
    """One dial raising must not abort the rest of the sweep —
    each connect runs in its own try/except.
    """
    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.peers = {}

    # connect_to_peer raises for peer-a, succeeds for peer-b
    async def connect_side_effect(addr):
        if addr == "1.1.1.1:9001":
            raise OSError("no route to host")
        return MagicMock()

    transport.connect_to_peer = AsyncMock(side_effect=connect_side_effect)

    d = _make_peer_discovery_with(
        transport=transport,
        known_peers={
            "peer-a": "1.1.1.1:9001",
            "peer-b": "2.2.2.2:9001",
        },
    )

    await d._auto_dial_sweep()  # must not raise

    dialed = {c.args[0] for c in transport.connect_to_peer.await_args_list}
    assert dialed == {"1.1.1.1:9001", "2.2.2.2:9001"}, (
        "Both dials must be attempted — one failure cannot abort the loop"
    )


@pytest.mark.asyncio
async def test_auto_dial_sweep_with_no_known_peers_is_noop():
    """Empty known_peers → zero dials. No-op safety check."""
    transport = MagicMock()
    transport.identity = MagicMock(node_id="self-node")
    transport.peers = {}
    transport.connect_to_peer = AsyncMock(return_value=MagicMock())

    d = _make_peer_discovery_with(transport=transport, known_peers={})

    await d._auto_dial_sweep()

    transport.connect_to_peer.assert_not_called()
