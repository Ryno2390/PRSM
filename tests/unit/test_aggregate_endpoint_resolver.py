"""Tests for the AggregateEndpointResolver placeholder follow-on.

Closes the node.py:1729 placeholder
``lambda node_id: f"https://{node_id}/compute/aggregate"``,
which assumed node_id was a hostname.

The new resolver consults two ordered backends:
  1. StaticMapEndpointResolver — operator-supplied node_id → URL dict
  2. TransportPeerEndpointResolver — derives URL from
     ``WebSocketTransport.get_peer(node_id).address`` (host:port)

If neither resolves the node_id, the resolver raises a typed error
that ``HttpAggregateTransport`` already maps to a clean failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pytest

from prsm.compute.query_orchestrator.aggregate_endpoint_resolver import (
    AggregateEndpointUnresolvedError,
    ChainedEndpointResolver,
    StaticMapEndpointResolver,
    TransportPeerEndpointResolver,
)


# ──────────────────────────────────────────────────────────────────────
# Test doubles for the WebSocketTransport peer-registry interface.
# Real transport is in prsm/node/transport.py — we stub the minimal
# get_peer(node_id) -> PeerConnection|None contract.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _StubPeer:
    address: str  # "host:port" — matches WebSocketTransport convention.


@dataclass
class _StubTransport:
    peers: Dict[str, _StubPeer] = field(default_factory=dict)

    def get_peer(self, node_id: str) -> Optional[_StubPeer]:
        return self.peers.get(node_id)


# ──────────────────────────────────────────────────────────────────────
# StaticMapEndpointResolver
# ──────────────────────────────────────────────────────────────────────


class TestStaticMapEndpointResolver:
    def test_returns_mapped_url(self):
        resolver = StaticMapEndpointResolver(
            {"node-a": "https://aggregator-a.example:9443"},
        )
        assert resolver("node-a") == "https://aggregator-a.example:9443"

    def test_unmapped_node_id_raises(self):
        resolver = StaticMapEndpointResolver({"node-a": "https://a.example"})
        with pytest.raises(AggregateEndpointUnresolvedError):
            resolver("node-b")

    def test_empty_map_raises_for_any_node(self):
        resolver = StaticMapEndpointResolver({})
        with pytest.raises(AggregateEndpointUnresolvedError):
            resolver("anything")

    def test_url_with_no_scheme_rejected_at_construction(self):
        # Force https/http to avoid accidental plaintext aggregator
        # endpoint leaking into prod via operator typo.
        with pytest.raises(ValueError, match="scheme"):
            StaticMapEndpointResolver({"node-a": "aggregator-a.example"})

    def test_constructor_copies_map(self):
        # Mutating the source dict after construction must not change
        # resolver behavior (defense against operator/test-fixture
        # accidentally mutating the table out from under us).
        src = {"node-a": "https://a.example"}
        resolver = StaticMapEndpointResolver(src)
        src["node-a"] = "https://evil.example"
        assert resolver("node-a") == "https://a.example"


# ──────────────────────────────────────────────────────────────────────
# TransportPeerEndpointResolver
# ──────────────────────────────────────────────────────────────────────


class TestTransportPeerEndpointResolver:
    def test_resolves_via_peer_address(self):
        transport = _StubTransport(
            peers={"node-a": _StubPeer(address="10.0.0.5:9001")},
        )
        resolver = TransportPeerEndpointResolver(transport)
        # Default scheme=https, default aggregate port falls back to
        # the peer's WS port.
        assert resolver("node-a") == "https://10.0.0.5:9001"

    def test_explicit_aggregate_port_overrides_peer_port(self):
        transport = _StubTransport(
            peers={"node-a": _StubPeer(address="10.0.0.5:9001")},
        )
        resolver = TransportPeerEndpointResolver(
            transport, aggregate_port=9443,
        )
        assert resolver("node-a") == "https://10.0.0.5:9443"

    def test_scheme_override(self):
        transport = _StubTransport(
            peers={"node-a": _StubPeer(address="10.0.0.5:9001")},
        )
        resolver = TransportPeerEndpointResolver(transport, scheme="http")
        assert resolver("node-a") == "http://10.0.0.5:9001"

    def test_unknown_node_raises(self):
        transport = _StubTransport(peers={})
        resolver = TransportPeerEndpointResolver(transport)
        with pytest.raises(AggregateEndpointUnresolvedError):
            resolver("node-z")

    def test_malformed_peer_address_raises(self):
        # "no-port-here" lacks the host:port shape — refuse rather than
        # synthesizing a wrong URL.
        transport = _StubTransport(
            peers={"node-a": _StubPeer(address="no-port-here")},
        )
        resolver = TransportPeerEndpointResolver(transport)
        with pytest.raises(AggregateEndpointUnresolvedError, match="address"):
            resolver("node-a")

    def test_invalid_scheme_rejected_at_construction(self):
        transport = _StubTransport()
        with pytest.raises(ValueError, match="scheme"):
            TransportPeerEndpointResolver(transport, scheme="ftp")

    def test_invalid_port_rejected_at_construction(self):
        transport = _StubTransport()
        with pytest.raises(ValueError, match="port"):
            TransportPeerEndpointResolver(transport, aggregate_port=0)
        with pytest.raises(ValueError, match="port"):
            TransportPeerEndpointResolver(transport, aggregate_port=70000)


# ──────────────────────────────────────────────────────────────────────
# ChainedEndpointResolver
# ──────────────────────────────────────────────────────────────────────


class TestChainedEndpointResolver:
    def test_first_resolver_wins(self):
        primary = StaticMapEndpointResolver(
            {"node-a": "https://primary.example"},
        )
        transport = _StubTransport(
            peers={"node-a": _StubPeer(address="10.0.0.5:9001")},
        )
        secondary = TransportPeerEndpointResolver(transport)
        chained = ChainedEndpointResolver([primary, secondary])
        assert chained("node-a") == "https://primary.example"

    def test_falls_back_to_secondary(self):
        primary = StaticMapEndpointResolver({})  # always misses
        transport = _StubTransport(
            peers={"node-a": _StubPeer(address="10.0.0.5:9001")},
        )
        secondary = TransportPeerEndpointResolver(transport)
        chained = ChainedEndpointResolver([primary, secondary])
        assert chained("node-a") == "https://10.0.0.5:9001"

    def test_all_miss_raises(self):
        primary = StaticMapEndpointResolver({})
        secondary = TransportPeerEndpointResolver(_StubTransport())
        chained = ChainedEndpointResolver([primary, secondary])
        with pytest.raises(AggregateEndpointUnresolvedError):
            chained("node-z")

    def test_empty_chain_rejected_at_construction(self):
        with pytest.raises(ValueError, match="empty"):
            ChainedEndpointResolver([])
