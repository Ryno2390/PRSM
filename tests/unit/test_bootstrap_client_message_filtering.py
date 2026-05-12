"""Sprint 319 — BootstrapClient must filter server messages by `type`.

Live dogfood 2026-05-12 against the canonical PRSM bootstrap server
(`wss://bootstrap1.prsm-network.com:8765`) reproduced the bug end
to end:

  1. Node A connects + registers (peers=[] returned — A is alone).
  2. Node B connects + registers (peers=[A] returned).
  3. Server pushes a `peer_join` announcement to A's socket so
     A learns about B in near-real-time.
  4. A calls `client.get_peers()` — pre-sprint-319, the client
     `recv()`'d ONE message and called `.get("peers", [])` on it
     without filtering by `type`. The push announcement (no
     `peers` key) was read instead of the actual `peer_list`
     response, and `get_peers()` returned `[]`.
  5. The real `peer_list` response stayed queued in the socket
     buffer, misaligning every subsequent request/response on
     this connection — A's NEXT `get_peers()` would return the
     stale response from the previous request.

Pre-sprint-319 behavior:
- Push-announcement-poisoned `get_peers()` returns []
- Every following request reads the stale response

Post-sprint-319 contract:
- `get_peers()` loops `recv()` until it sees `type == peer_list`
  (with bounded retries to fail loud on broken servers)
- Server-pushed announcements (peer_join/peer_leave/etc) are
  drained off the socket but do not corrupt the request/response
  alignment

The same hazard applies to `connect()` — register-ack reads
should filter by `type == register_ack`. A pathological server
that sends an announcement before the register_ack would today
cause `connect()` to raise on the announcement payload.

These tests use an in-process WebSocket server to emulate the
canonical bootstrap-server message ordering deterministically
so we can validate the fix without depending on the live
network.
"""
from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any, List, Optional

import pytest
import websockets

from prsm.bootstrap.client import BootstrapClient


# ── Test harness ──────────────────────────────────────────


class _StubBootstrapServer:
    """In-process server that emulates the canonical PRSM
    bootstrap-server message protocol enough to test the
    client's message-type filtering.

    The scripted sequence is:
      - on `register`: reply with `register_ack`, then
        immediately push a `peer_join` announcement (no `peers`
        key — this is the poison message)
      - on `get_peers`: reply with `type=peer_list` and a
        non-empty peers array

    The poison ordering forces the bug surface: a fast client
    that does a single recv() after sending get_peers will read
    the queued peer_join announcement instead of the peer_list
    response.
    """

    def __init__(self) -> None:
        self.peers_to_return: List[dict] = [
            {
                "peer_id": "other-node",
                "address": "10.0.0.2",
                "port": 9002,
                "capabilities": ["compute"],
                "region": None,
                "version": "test",
            },
        ]
        self._server = None
        self.port: Optional[int] = None

    async def __aenter__(self) -> "_StubBootstrapServer":
        self._server = await websockets.serve(
            self._handle, "127.0.0.1", 0,
        )
        # websockets >= 12 exposes sockets via .sockets; getsockname()
        # returns the bound port
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    async def _handle(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")
            if mtype == "register":
                await ws.send(json.dumps({
                    "type": "register_ack",
                    "peer_id": msg.get("peer_id"),
                    "peers": [],
                    "heartbeat_interval": 30,
                    "server_time": "2026-05-12T00:00:00+00:00",
                }))
                # Immediately push a peer_join announcement that
                # has NO `peers` key — this is the message that
                # poisons get_peers in the pre-sprint-319 client
                await ws.send(json.dumps({
                    "announcement_id": "ann-1",
                    "announcement_type": "peer_join",
                    "peer_id": "other-node",
                    "peer_endpoint": "10.0.0.2:9002",
                    "timestamp": "2026-05-12T00:00:01+00:00",
                }))
            elif mtype == "get_peers":
                await ws.send(json.dumps({
                    "type": "peer_list",
                    "peers": self.peers_to_return,
                    "total": len(self.peers_to_return),
                    "server_time": "2026-05-12T00:00:02+00:00",
                }))
            elif mtype == "disconnect":
                return
            elif mtype == "heartbeat":
                # No response required
                pass


# ── Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_peers_filters_by_type_skipping_announcements():
    """The canonical bug repro: after register, a peer_join
    announcement sits queued in the socket. `get_peers()` must
    skip the announcement and return the peer_list payload,
    not silently return [].
    """
    async with _StubBootstrapServer() as srv:
        c = BootstrapClient(
            bootstrap_url=srv.url,
            node_id="probe",
            port=9001,
            capabilities=["x"],
            version="t",
        )
        await c.connect()
        # Give the server's push announcement time to land in
        # the local socket buffer before we issue get_peers
        await asyncio.sleep(0.05)

        peers = await c.get_peers()

        # Pre-sprint-319 this returned [] because the
        # peer_join announcement was read in place of the
        # peer_list response
        assert len(peers) == 1
        assert peers[0].peer_id == "other-node"
        await c.disconnect()


@pytest.mark.asyncio
async def test_request_response_alignment_after_announcement():
    """Two `get_peers()` calls back-to-back should each return
    the same fresh peer_list — pre-fix the second call would
    return the STALE peer_list response queued from the first
    call's misalignment.
    """
    async with _StubBootstrapServer() as srv:
        c = BootstrapClient(
            bootstrap_url=srv.url, node_id="probe", port=9001,
        )
        await c.connect()
        await asyncio.sleep(0.05)
        a = await c.get_peers()
        b = await c.get_peers()
        assert [p.peer_id for p in a] == ["other-node"]
        assert [p.peer_id for p in b] == ["other-node"]
        await c.disconnect()


@pytest.mark.asyncio
async def test_get_peers_raises_loudly_on_unbounded_garbage():
    """If the server only ever sends non-peer_list messages,
    get_peers must NOT hang forever — it must time out / raise
    so operators see a real error instead of silent empty
    results.
    """
    class _AnnouncementOnlyServer:
        def __init__(self): self._server = None; self.port = None
        async def __aenter__(self):
            self._server = await websockets.serve(
                self._handle, "127.0.0.1", 0,
            )
            self.port = self._server.sockets[0].getsockname()[1]
            return self
        async def __aexit__(self, *exc):
            self._server.close()
            await self._server.wait_closed()
        @property
        def url(self): return f"ws://127.0.0.1:{self.port}"
        async def _handle(self, ws):
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "register":
                    await ws.send(json.dumps({
                        "type": "register_ack",
                        "peer_id": msg.get("peer_id"),
                        "peers": [], "heartbeat_interval": 30,
                        "server_time": "x",
                    }))
                elif msg.get("type") == "get_peers":
                    # Pathological: server replies with garbage
                    # type forever, never sends peer_list
                    for _ in range(50):
                        await ws.send(json.dumps({
                            "announcement_id": "x",
                            "announcement_type": "peer_join",
                            "peer_id": "z",
                        }))

    async with _AnnouncementOnlyServer() as srv:
        c = BootstrapClient(
            bootstrap_url=srv.url, node_id="probe", port=9001,
            connect_timeout=1.0,
        )
        await c.connect()
        with pytest.raises(
            (ConnectionError, asyncio.TimeoutError),
        ):
            await c.get_peers()
        await c.disconnect()


@pytest.mark.asyncio
async def test_connect_filters_register_ack_from_announcements():
    """If a server emits an announcement BEFORE the register_ack
    (which the canonical server doesn't today but could under
    a race), connect() must still find the register_ack rather
    than parsing the announcement as the ack.
    """
    class _PreAckAnnouncementServer:
        def __init__(self): self._server = None; self.port = None
        async def __aenter__(self):
            self._server = await websockets.serve(
                self._handle, "127.0.0.1", 0,
            )
            self.port = self._server.sockets[0].getsockname()[1]
            return self
        async def __aexit__(self, *exc):
            self._server.close()
            await self._server.wait_closed()
        @property
        def url(self): return f"ws://127.0.0.1:{self.port}"
        async def _handle(self, ws):
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "register":
                    # Pre-ack announcement
                    await ws.send(json.dumps({
                        "announcement_id": "pre-ack",
                        "announcement_type": "system_notice",
                        "msg": "welcome",
                    }))
                    await ws.send(json.dumps({
                        "type": "register_ack",
                        "peer_id": msg.get("peer_id"),
                        "peers": [], "heartbeat_interval": 30,
                        "server_time": "x",
                    }))

    async with _PreAckAnnouncementServer() as srv:
        c = BootstrapClient(
            bootstrap_url=srv.url, node_id="probe", port=9001,
            connect_timeout=2.0,
        )
        peers = await c.connect()
        assert c.is_registered
        assert peers == []
        await c.disconnect()
