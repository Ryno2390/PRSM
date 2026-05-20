"""Sprint 632 — BootstrapClient periodic peer-list refresh.

Sprint 630 fleet recovery during live work needed 3 daemon restarts
in specific order because BootstrapClient only fetches peers ONCE
(at registration). If the bootstrap server has empty state at
register time (server just restarted, or peer hasn't joined yet),
the client's `_peers` stays empty even when peers later appear on
the server — no auto-refresh kicks in.

Sprint 632 fix: a `start_peer_refresh(interval=...)` task that
periodically calls `get_peers()` + invokes `on_peers_discovered`
for newly-discovered peers. Heartbeat loop is unchanged.

Live evidence the bug is real:
- Sprint 630: Mac killed, restarted. bootstrap-us:8765 returned
  peers=[] at register (server held stale state from prior Mac
  registration). Mac never re-fetched. Required full
  bootstrap-server + droplet + Mac restart cascade to recover
  symmetric discovery.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional

import pytest
import websockets

from prsm.bootstrap.client import BootstrapClient


class _MutablePeerStubServer:
    """Stub bootstrap server whose peer list can change AFTER
    a client has already registered. Models the sprint-630 race
    where the server's peer registry grows after a client has
    fetched its initial peer list.
    """

    def __init__(self) -> None:
        self.peers_to_return: List[dict] = []  # Empty at register
        self._server = None
        self.port: Optional[int] = None
        self.get_peers_calls = 0  # Count refresh invocations

    async def __aenter__(self):
        self._server = await websockets.serve(
            self._handle, "127.0.0.1", 0,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def _handle(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")
            if mtype == "register":
                # Use whatever's in peers_to_return at register time
                await ws.send(json.dumps({
                    "type": "register_ack",
                    "peer_id": msg.get("peer_id"),
                    "peers": list(self.peers_to_return),
                    "heartbeat_interval": 30,
                    "server_time": "2026-05-20T00:00:00+00:00",
                }))
            elif mtype == "get_peers":
                self.get_peers_calls += 1
                await ws.send(json.dumps({
                    "type": "peer_list",
                    "peers": list(self.peers_to_return),
                    "total": len(self.peers_to_return),
                    "server_time": "2026-05-20T00:00:01+00:00",
                }))
            elif mtype == "heartbeat":
                # No response required
                pass


@pytest.mark.asyncio
async def test_periodic_refresh_picks_up_peers_joined_after_register():
    """Server returns [] at register; adds a peer later; client
    refresh must surface it via on_peers_discovered."""
    async with _MutablePeerStubServer() as stub:
        discovered_calls: List[List] = []

        def _on_peers(peers):
            discovered_calls.append(list(peers))

        client = BootstrapClient(
            bootstrap_url=stub.url,
            node_id="test-node",
            port=9001,
            capabilities=["compute"],
            on_peers_discovered=_on_peers,
            connect_timeout=5.0,
        )
        peers_at_register = await client.connect()
        assert peers_at_register == [], "stub returns [] at register"
        # on_peers_discovered should NOT have fired (peers empty)
        assert discovered_calls == []

        # Now: peer joins on the server side
        stub.peers_to_return = [{
            "peer_id": "late-joiner",
            "address": "10.0.0.5",
            "port": 9005,
            "capabilities": ["compute"],
            "region": None,
            "version": "test",
        }]

        # Sprint 632 contract: client refreshes peer list periodically.
        # Use 0.1s interval for fast test.
        await client.start_peer_refresh(interval=0.1)

        # Wait for at least one refresh to happen
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if discovered_calls:
                break
            await asyncio.sleep(0.05)

        await client.disconnect()

        assert stub.get_peers_calls >= 1, (
            "client must call get_peers at least once during refresh "
            f"(saw {stub.get_peers_calls})"
        )
        assert discovered_calls, (
            "on_peers_discovered must fire when refresh surfaces a "
            "new peer"
        )
        # The newly-discovered peer is the one we added
        latest = discovered_calls[-1]
        assert len(latest) == 1
        assert latest[0].peer_id == "late-joiner"


@pytest.mark.asyncio
async def test_periodic_refresh_skips_callback_when_no_new_peers():
    """Refresh that returns the same peer list should NOT re-fire
    the on_peers_discovered callback — otherwise downstream
    auto-dial sweeps would re-attempt connections every interval.

    Sprint 632 contract: callback fires only for newly-discovered
    peers (delta), not on every refresh tick.
    """
    async with _MutablePeerStubServer() as stub:
        # Pre-populate so register returns a peer
        stub.peers_to_return = [{
            "peer_id": "existing-peer",
            "address": "10.0.0.6",
            "port": 9006,
            "capabilities": ["compute"],
            "region": None,
            "version": "test",
        }]

        discovered_calls: List[List] = []

        def _on_peers(peers):
            discovered_calls.append(list(peers))

        client = BootstrapClient(
            bootstrap_url=stub.url,
            node_id="test-node-2",
            port=9001,
            on_peers_discovered=_on_peers,
            connect_timeout=5.0,
        )
        await client.connect()
        # Callback fired once at register
        assert len(discovered_calls) == 1
        assert discovered_calls[0][0].peer_id == "existing-peer"

        # Refresh with same peer — should NOT re-fire
        await client.start_peer_refresh(interval=0.05)
        # Poll until refresh ticks fire (multi-await yields the loop
        # so the refresh task can run; a single long sleep with
        # pytest-asyncio can starve background tasks).
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if stub.get_peers_calls >= 2:
                break
            await asyncio.sleep(0.05)
        await client.disconnect()

        # Refresh ran but callback didn't re-fire (no delta)
        assert stub.get_peers_calls >= 1, (
            f"refresh must actually call get_peers "
            f"(got {stub.get_peers_calls})"
        )
        assert len(discovered_calls) == 1, (
            "on_peers_discovered must NOT re-fire on identical "
            "peer list (delta-only contract)"
        )


@pytest.mark.asyncio
async def test_start_peer_refresh_is_idempotent():
    """Calling start_peer_refresh twice must not spawn a second
    task — protects against accidental double-scheduling.
    """
    async with _MutablePeerStubServer() as stub:
        client = BootstrapClient(
            bootstrap_url=stub.url,
            node_id="test-node-3",
            port=9001,
            connect_timeout=5.0,
        )
        await client.connect()
        first_task = None
        await client.start_peer_refresh(interval=0.05)
        first_task = client._refresh_task
        await client.start_peer_refresh(interval=0.05)
        # Idempotent: second call must NOT swap the task object.
        assert client._refresh_task is first_task, (
            "second start_peer_refresh must be a no-op; got new task"
        )
        # Poll for ticks
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if stub.get_peers_calls >= 2:
                break
            await asyncio.sleep(0.05)
        await client.disconnect()
        # The is-still-the-same-task assertion above is the
        # primary idempotency check. The call count just sanity-
        # checks the refresh actually ran.
        assert stub.get_peers_calls >= 1, (
            f"refresh must have run at least once; got "
            f"{stub.get_peers_calls}"
        )
