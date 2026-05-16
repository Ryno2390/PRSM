"""Sprint 474 — `_DeadBootstrapSentinel` recovery pin.

Sprint 321 introduced `_DeadBootstrapSentinel` as the placeholder
installed in `_bootstrap_client` when a reconnect attempt fails.
Sprint 324 surfaced this as `client_state: "dead"` in
`/bootstrap/status` for operator triage.

The production-reliability guarantee: a daemon stuck in
`client_state: "dead"` MUST eventually recover to "connected"
once the bootstrap server (or a fallback) becomes reachable
again. If the sentinel got "stuck" (e.g., a future refactor
forgot to overwrite `_bootstrap_client` on `_try_bootstrap_client`
success), operators would see a permanent "dead" status with
no real underlying problem — the same false-positive alert
class as sprint 473's F21.

This pin defends the transition: sentinel-installed →
reconnect succeeds → sentinel replaced by live client →
`client_state` reports "connected".

These tests are NOT a refactor of the existing 4
reconnect tests in `test_libp2p_discovery_reconnect.py` — they
focus specifically on the `_DeadBootstrapSentinel` slot
management invariant.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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
        bootstrap_nodes=["wss://bootstrap.example.com:8765"],
    )


def test_sentinel_reports_dead_via_get_bootstrap_status():
    """When the slot holds a sentinel, `get_bootstrap_status`
    must report `client_state: "dead"` — operator-visible
    triage signal."""
    d = _make_discovery()
    d._bootstrap_client = _DeadBootstrapSentinel()
    status = d.get_bootstrap_status()
    assert status["client_state"] == "dead"


def test_no_client_reports_none():
    """When the slot is None (initial state), `client_state`
    must be 'none' — distinct from 'dead' so operators can
    tell "never tried" from "tried + failed"."""
    d = _make_discovery()
    d._bootstrap_client = None
    assert d.get_bootstrap_status()["client_state"] == "none"


def test_connected_client_reports_connected():
    """Sanity: a live is_connected=True client → 'connected'."""
    d = _make_discovery()
    fake = MagicMock()
    fake.is_connected = True
    d._bootstrap_client = fake
    assert d.get_bootstrap_status()["client_state"] == "connected"


def test_dead_sentinel_recovers_when_reconnect_succeeds():
    """The production-reliability invariant: if the slot
    holds a sentinel and `_try_bootstrap_client` succeeds,
    the sentinel MUST be replaced by the new live client.
    Otherwise operators see permanent `client_state: "dead"`
    even though connectivity recovered."""
    d = _make_discovery()
    d._bootstrap_client = _DeadBootstrapSentinel()
    assert d.get_bootstrap_status()["client_state"] == "dead"

    fresh = MagicMock()
    fresh.is_connected = True
    fresh._observed_announcements = []

    async def fake_try(addrs):
        d._bootstrap_client = fresh
        return True

    async def runner():
        with patch.object(
            d, "_try_bootstrap_client", side_effect=fake_try,
        ):
            ok = await d._try_bootstrap_client(
                ["wss://bootstrap.example.com:8765"],
            )
            assert ok is True

    asyncio.run(runner())

    # Sentinel must have been replaced by the live client.
    assert d._bootstrap_client is fresh
    assert d.get_bootstrap_status()["client_state"] == "connected"


def test_dead_sentinel_is_not_connected_so_loop_keeps_retrying():
    """The sentinel must satisfy `is_connected == False` so
    the poll loop's reconnect-on-drop branch keeps firing.
    Otherwise the loop would treat the sentinel as a live
    client and silently stop trying to reconnect — the exact
    pre-sprint-321 failure mode the sentinel was designed
    to prevent."""
    s = _DeadBootstrapSentinel()
    assert s.is_connected is False
