"""/peers connected_count uses kernel-truth (sprint 135).

Pre-fix: `connected_count` came from `len(transport._peers)`,
which only tracked OUTBOUND peers (those we initiated). Inbound
peers were invisible to /peers, even though /status reported
them via transport.peer_count. Multinode dogfood (sprint 133)
caught the discrepancy: same node, /peers said 0, /status said 2.

Post-fix: connected_count = transport.peer_count (kernel truth),
matching /status. The connected[] list is best-effort union of
outbound (rich metadata) + inbound (minimal stubs).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.libp2p_transport import PeerConnection


def _node(*, outbound=0, kernel_count=0, kernel_addrs=None):
    node = MagicMock()
    node.identity.node_id = "test"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None

    transport = MagicMock()
    transport.peer_count = kernel_count
    transport.peer_addresses = kernel_addrs or []
    peers_dict = {}
    for i in range(outbound):
        pid = f"out-peer-{i}"
        peers_dict[pid] = PeerConnection(
            peer_id=pid,
            address=f"/ip4/127.0.0.1/udp/9001/quic-v1/p2p/{pid}",
            websocket=None,
            outbound=True,
        )
    transport.peers = peers_dict
    node.transport = transport
    node.discovery = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestKernelTruthCount:
    def test_inbound_only_count_matches_status(self):
        # 2 inbound (kernel knows), 0 outbound (we didn't track)
        node = _node(
            outbound=0,
            kernel_count=2,
            kernel_addrs=[
                "/ip4/127.0.0.1/udp/9011/quic-v1/p2p/peer-b",
                "/ip4/127.0.0.1/udp/9021/quic-v1/p2p/peer-c",
            ],
        )
        body = _client(node).get("/peers").json()
        # Pre-fix this was 0; post-fix should be 2 (kernel truth)
        assert body["connected_count"] == 2
        # Connected list shows minimal stubs for inbound
        assert len(body["connected"]) == 2
        for entry in body["connected"]:
            assert entry["peer_id"] is None
            assert entry["outbound"] is False

    def test_outbound_keeps_rich_metadata(self):
        node = _node(
            outbound=1,
            kernel_count=1,
            kernel_addrs=[
                "/ip4/127.0.0.1/udp/9001/quic-v1/p2p/out-peer-0",
            ],
        )
        body = _client(node).get("/peers").json()
        assert body["connected_count"] == 1
        # Outbound entry has rich metadata
        assert len(body["connected"]) == 1
        entry = body["connected"][0]
        assert entry["peer_id"] == "out-peer-0"
        assert entry["outbound"] is True

    def test_mixed_inbound_outbound_no_dupes(self):
        # 1 outbound + 2 distinct inbound = 3 connections
        node = _node(
            outbound=1,
            kernel_count=3,
            kernel_addrs=[
                "/ip4/127.0.0.1/udp/9001/quic-v1/p2p/out-peer-0",  # already in _peers
                "/ip4/127.0.0.1/udp/9011/quic-v1/p2p/peer-b",
                "/ip4/127.0.0.1/udp/9021/quic-v1/p2p/peer-c",
            ],
        )
        body = _client(node).get("/peers").json()
        assert body["connected_count"] == 3
        # connected[] = 1 outbound (rich) + 2 inbound (stubs) = 3
        # The outbound peer's address is in seen_addresses so
        # the kernel-list duplicate gets skipped.
        assert len(body["connected"]) == 3
        outbound_count = sum(
            1 for e in body["connected"] if e["outbound"]
        )
        assert outbound_count == 1


class TestNoTransport:
    def test_no_transport_returns_zero(self):
        node = _node()
        node.transport = None
        body = _client(node).get("/peers").json()
        assert body["connected_count"] == 0
        assert body["connected"] == []


class TestPeerAddressesProbeFailSoft:
    def test_peer_addresses_failure_doesnt_break_endpoint(self):
        node = _node(outbound=1, kernel_count=2)
        # Make peer_addresses raise
        type(node.transport).peer_addresses = property(
            lambda self: (_ for _ in ()).throw(
                RuntimeError("simulated"),
            ),
        )
        body = _client(node).get("/peers").json()
        # Endpoint still returns 200 with kernel-truth count
        assert body["connected_count"] == 2
        # Connected list has at least the outbound entry
        assert len(body["connected"]) >= 1
