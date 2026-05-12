"""Sprint 321 — Libp2pDiscovery reconnects to bootstrap on drop.

Pre-sprint-321 if the BootstrapClient WebSocket drops (server
restart / transient network blip / heartbeat timeout), the
heartbeat loop set `_connected = False` and exited; the poll
loop kept calling `client.get_peers()` against the dead socket,
caught `ConnectionError` at debug, slept, retried forever. The
node silently lost bootstrap awareness — no new peer discovery,
no peer_leave eviction, indefinitely until process restart.

Sprint 321 makes the poll loop reconnect with a fresh
BootstrapClient on `get_peers()` failure, bounded by exponential
backoff so we don't hammer a degraded server.

The poll loop's existing per-tick try/except keeps semantics
unchanged for non-network exceptions (json parse errors etc).
Only `client.is_connected == False` after a recv exception
triggers the reconnect path.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.node.libp2p_discovery import Libp2pDiscovery


def _make_discovery():
    transport = MagicMock()
    transport.identity.node_id = "self-node"
    transport.port = 9001
    transport.connect_to_peer = AsyncMock(return_value=None)
    return Libp2pDiscovery(
        transport=transport,
        bootstrap_nodes=["wss://bootstrap.example.com:8765"],
    )


# ── Reconnect happens after get_peers raises ConnectionError


def test_poll_loop_reconnects_after_get_peers_failure():
    """After get_peers raises ConnectionError, the loop must
    attempt to rebuild the BootstrapClient and resume polling."""
    d = _make_discovery()

    dead_client = MagicMock()
    dead_client.is_connected = False
    dead_client.get_peers = AsyncMock(
        side_effect=ConnectionError("socket dropped"),
    )
    dead_client._observed_announcements = []
    d._bootstrap_client = dead_client

    fresh_client = MagicMock()
    fresh_client.is_connected = True
    fresh_client.get_peers = AsyncMock(return_value=[])
    fresh_client._observed_announcements = []

    # Patch _try_bootstrap_client so the loop's reconnect call
    # picks up our pre-built fresh client deterministically
    async def fake_reconnect(addrs):
        d._bootstrap_client = fresh_client
        return True

    with patch.object(d, "_try_bootstrap_client", side_effect=fake_reconnect):
        import os
        os.environ["PRSM_BOOTSTRAP_POLL_SECONDS"] = "0.05"
        try:
            async def runner():
                task = asyncio.create_task(d._bootstrap_poll_loop())
                # Hand-yield to step the task through dead-call
                # → reconnect → fresh-call
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

    # Fresh client replaced the dead one
    assert d._bootstrap_client is fresh_client
    # And the fresh client received at least one get_peers call
    # (the reconnect immediately retries instead of waiting one
    # full poll interval)
    assert fresh_client.get_peers.await_count >= 1


def test_poll_loop_does_not_reconnect_when_client_still_connected():
    """If get_peers raises for non-network reasons (parse error
    etc) but the client is still connected, do NOT reconnect —
    just log + retry on next tick. Avoids reconnect thrash from
    transient logic errors."""
    d = _make_discovery()

    client = MagicMock()
    client.is_connected = True
    # First call raises a non-network error, second call succeeds
    client.get_peers = AsyncMock(
        side_effect=[ValueError("bad json"), []],
    )
    client._observed_announcements = []
    d._bootstrap_client = client

    reconnect_calls = []
    async def track_reconnect(addrs):
        reconnect_calls.append(addrs)
        return True

    with patch.object(d, "_try_bootstrap_client", side_effect=track_reconnect):
        import os
        os.environ["PRSM_BOOTSTRAP_POLL_SECONDS"] = "0.05"
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

    # No reconnect was triggered — client was still connected
    assert reconnect_calls == []


def test_reconnect_failure_keeps_loop_alive_for_next_tick():
    """If reconnect itself fails, the loop must NOT terminate —
    on the next tick it must try again."""
    d = _make_discovery()

    dead_client = MagicMock()
    dead_client.is_connected = False
    dead_client.get_peers = AsyncMock(
        side_effect=ConnectionError("socket dropped"),
    )
    dead_client._observed_announcements = []
    d._bootstrap_client = dead_client

    reconnect_attempts = []
    async def always_fail_reconnect(addrs):
        reconnect_attempts.append(addrs)
        return False

    with patch.object(d, "_try_bootstrap_client", side_effect=always_fail_reconnect):
        import os
        os.environ["PRSM_BOOTSTRAP_POLL_SECONDS"] = "0.05"
        try:
            async def runner():
                task = asyncio.create_task(d._bootstrap_poll_loop())
                for _ in range(60):
                    await asyncio.sleep(0)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            asyncio.run(runner())
        finally:
            os.environ.pop("PRSM_BOOTSTRAP_POLL_SECONDS", None)

    # Multiple reconnect attempts were made — loop stayed alive
    assert len(reconnect_attempts) >= 2


def test_reconnect_only_targets_websocket_bootstrap_addresses():
    """Reconnect logic must only attempt ws:// / wss:// addresses
    (same constraint as initial _try_bootstrap_client fallback).
    Non-ws addresses bypass the fallback path entirely."""
    d = _make_discovery()
    # Override bootstrap_nodes to mixed types — only wss should
    # be passed to reconnect
    d.bootstrap_nodes = [
        "/dns4/peer.example.com/tcp/4001/p2p/QmAbc",  # libp2p multiaddr
        "wss://bootstrap.example.com:8765",
        "https://not-supported.example.com",
    ]

    dead_client = MagicMock()
    dead_client.is_connected = False
    dead_client.get_peers = AsyncMock(
        side_effect=ConnectionError("dropped"),
    )
    dead_client._observed_announcements = []
    d._bootstrap_client = dead_client

    seen = []
    async def capture_reconnect(addrs):
        seen.append(list(addrs))
        return False

    with patch.object(d, "_try_bootstrap_client", side_effect=capture_reconnect):
        import os
        os.environ["PRSM_BOOTSTRAP_POLL_SECONDS"] = "0.05"
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

    # At least one reconnect attempt was made
    assert seen, "expected at least one reconnect attempt"
    for addrs in seen:
        # Every reconnect attempt receives ONLY wss:// or ws:// URLs
        for a in addrs:
            assert a.startswith(("ws://", "wss://")), (
                f"reconnect attempted on non-WS address: {a}"
            )
