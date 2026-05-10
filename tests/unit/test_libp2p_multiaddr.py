"""Libp2pTransport._to_multiaddr address parsing (sprint 120).

Pre-fix: hostnames like bootstrap1.prsm-network.com were
formatted as `/ip4/{host}/...` which libp2p rejects (requires
dotted-quad IPv4 literal). The bootstrap connection failed:

  PrsmConnect: invalid multiaddr "/ip4/bootstrap1.prsm-network.com
  /tcp/8765/ws": failed to parse multiaddr: invalid value
  "bootstrap1.prsm-network.com" for protocol ip4

Fix: detect IPv4 literal vs DNS hostname, emit /ip4/ or /dns4/
accordingly per libp2p multiaddr spec.
"""
from __future__ import annotations

import pytest

from prsm.node.libp2p_transport import Libp2pTransport


class TestAddrProto:
    def test_ipv4_literal(self):
        assert Libp2pTransport._addr_proto("127.0.0.1") == "ip4"
        assert Libp2pTransport._addr_proto("1.2.3.4") == "ip4"
        assert Libp2pTransport._addr_proto("100.83.80.91") == "ip4"

    def test_hostname(self):
        assert Libp2pTransport._addr_proto("example.com") == "dns4"
        assert (
            Libp2pTransport._addr_proto("bootstrap1.prsm-network.com")
            == "dns4"
        )
        assert Libp2pTransport._addr_proto("localhost") == "dns4"


class TestToMultiaddr:
    def test_wss_hostname_uses_dns4(self):
        """The bootstrap regression case."""
        result = Libp2pTransport._to_multiaddr(
            "wss://bootstrap1.prsm-network.com:8765",
        )
        assert result == "/dns4/bootstrap1.prsm-network.com/tcp/8765/ws"

    def test_wss_ipv4_uses_ip4(self):
        result = Libp2pTransport._to_multiaddr("wss://1.2.3.4:8765")
        assert result == "/ip4/1.2.3.4/tcp/8765/ws"

    def test_ws_hostname_uses_dns4(self):
        result = Libp2pTransport._to_multiaddr(
            "ws://example.com:80",
        )
        assert result == "/dns4/example.com/tcp/80/ws"

    def test_plain_host_port_uses_quic(self):
        # Hostname → dns4, IPv4 → ip4
        assert Libp2pTransport._to_multiaddr("example.com:9001") == (
            "/dns4/example.com/udp/9001/quic-v1"
        )
        assert Libp2pTransport._to_multiaddr("127.0.0.1:9001") == (
            "/ip4/127.0.0.1/udp/9001/quic-v1"
        )

    def test_passthrough_existing_multiaddr(self):
        for src in (
            "/ip4/1.2.3.4/tcp/8765/ws",
            "/ip6/::1/tcp/8765/ws",
            "/dns4/example.com/tcp/8765/ws",
            "/dns/example.com/tcp/8765/ws",
            "/dns6/example.com/tcp/8765/ws",
        ):
            assert Libp2pTransport._to_multiaddr(src) == src

    def test_hostname_only_fallback_uses_dns4(self):
        result = Libp2pTransport._to_multiaddr("example.com")
        assert result == "/dns4/example.com/udp/9001/quic-v1"
