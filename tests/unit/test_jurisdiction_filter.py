"""Unit tests for prsm.node.jurisdiction_filter — R9 Phase 6.3."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.node.jurisdiction_filter import (
    ChainedGeoIPResolver,
    FilterDecision,
    GeoIPConfigError,
    GeoIPResolver,
    PeerJurisdictionFilter,
    StaticGeoIPResolver,
    _host_to_ip_maybe,
    _normalize_country_code,
    _normalize_country_set,
)


class TestNormalization:
    def test_valid_lowercase(self):
        assert _normalize_country_code("us") == "us"
        assert _normalize_country_code("cn") == "cn"

    def test_uppercase_normalized(self):
        assert _normalize_country_code("US") == "us"
        assert _normalize_country_code("CN") == "cn"

    def test_whitespace_stripped(self):
        assert _normalize_country_code("  de  ") == "de"

    def test_invalid_length(self):
        with pytest.raises(GeoIPConfigError):
            _normalize_country_code("usa")
        with pytest.raises(GeoIPConfigError):
            _normalize_country_code("u")
        with pytest.raises(GeoIPConfigError):
            _normalize_country_code("")

    def test_invalid_chars(self):
        with pytest.raises(GeoIPConfigError):
            _normalize_country_code("u1")
        with pytest.raises(GeoIPConfigError):
            _normalize_country_code("12")
        with pytest.raises(GeoIPConfigError):
            _normalize_country_code("我国")  # non-ASCII

    def test_normalize_set(self):
        result = _normalize_country_set(["US", "cn", " DE "])
        assert result == frozenset({"us", "cn", "de"})

    def test_normalize_set_raises_on_invalid(self):
        with pytest.raises(GeoIPConfigError):
            _normalize_country_set(["us", "invalid"])


class TestStaticResolver:
    def test_basic_lookup(self):
        resolver = StaticGeoIPResolver({
            "peer.cn.example": "cn",
            "peer.us.example": "US",
        })
        assert resolver.resolve("peer.cn.example") == "cn"
        assert resolver.resolve("peer.us.example") == "us"

    def test_case_insensitive_host(self):
        resolver = StaticGeoIPResolver({"PEER.example": "cn"})
        assert resolver.resolve("peer.example") == "cn"
        assert resolver.resolve("PEER.EXAMPLE") == "cn"

    def test_miss_returns_none(self):
        resolver = StaticGeoIPResolver({"peer.example": "us"})
        assert resolver.resolve("unknown.example") is None

    def test_empty_host_returns_none(self):
        resolver = StaticGeoIPResolver({})
        assert resolver.resolve("") is None

    def test_construction_rejects_invalid_country(self):
        with pytest.raises(GeoIPConfigError):
            StaticGeoIPResolver({"peer.example": "INVALID"})

    def test_construction_rejects_empty_host_key(self):
        with pytest.raises(GeoIPConfigError):
            StaticGeoIPResolver({"": "us"})

    def test_satisfies_protocol(self):
        resolver = StaticGeoIPResolver({})
        assert isinstance(resolver, GeoIPResolver)


class TestHostToIpMaybe:
    def test_ipv4_literal(self):
        assert _host_to_ip_maybe("192.0.2.1") == "192.0.2.1"

    def test_ipv6_literal(self):
        # IPv6 literals get canonicalized
        assert _host_to_ip_maybe("::1") == "::1"
        assert _host_to_ip_maybe("2001:db8::1") == "2001:db8::1"

    def test_dns_resolution_success(self):
        """Mock getaddrinfo to return a known IPv4 address."""
        import socket

        fake_infos = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "",
             ("93.184.216.34", 0)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_infos):
            assert _host_to_ip_maybe("example.com") == "93.184.216.34"

    def test_dns_resolution_prefers_ipv4(self):
        import socket

        fake_infos = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "",
             ("2606:2800:220:1:248:1893:25c8:1946", 0, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "",
             ("93.184.216.34", 0)),
        ]
        with patch("socket.getaddrinfo", return_value=fake_infos):
            assert _host_to_ip_maybe("example.com") == "93.184.216.34"

    def test_dns_failure_returns_none(self):
        import socket

        with patch("socket.getaddrinfo", side_effect=socket.gaierror):
            assert _host_to_ip_maybe("unresolvable.invalid") is None


class TestChainedResolver:
    def test_requires_nonempty(self):
        with pytest.raises(GeoIPConfigError):
            ChainedGeoIPResolver([])

    def test_first_hit_wins(self):
        first = StaticGeoIPResolver({"peer.example": "cn"})
        second = StaticGeoIPResolver({"peer.example": "us"})
        chained = ChainedGeoIPResolver([first, second])
        assert chained.resolve("peer.example") == "cn"

    def test_falls_through_on_miss(self):
        first = StaticGeoIPResolver({"other.example": "xx"})
        second = StaticGeoIPResolver({"peer.example": "us"})
        chained = ChainedGeoIPResolver([first, second])
        assert chained.resolve("peer.example") == "us"

    def test_all_miss_returns_none(self):
        first = StaticGeoIPResolver({})
        second = StaticGeoIPResolver({})
        chained = ChainedGeoIPResolver([first, second])
        assert chained.resolve("peer.example") is None

    def test_exception_in_resolver_does_not_break_chain(self):
        """A buggy resolver shouldn't poison the chain."""

        class BrokenResolver:
            def resolve(self, host):
                raise RuntimeError("oops")

        fallback = StaticGeoIPResolver({"peer.example": "de"})
        chained = ChainedGeoIPResolver([BrokenResolver(), fallback])
        assert chained.resolve("peer.example") == "de"


class TestPeerJurisdictionFilter:
    def _resolver(self, mapping: dict) -> StaticGeoIPResolver:
        return StaticGeoIPResolver(mapping)

    def test_empty_host_blocked(self):
        f = PeerJurisdictionFilter(
            resolver=self._resolver({}), excluded={"cn"}
        )
        decision = f.evaluate("")
        assert decision.allow is False
        assert decision.reason == "empty_host"

    def test_excluded_peer_blocked(self):
        f = PeerJurisdictionFilter(
            resolver=self._resolver({"peer.cn": "cn"}),
            excluded={"cn", "ru"},
        )
        decision = f.evaluate("peer.cn")
        assert decision.allow is False
        assert decision.detected_jurisdiction == "cn"
        assert "cn" in decision.reason

    def test_allowed_peer_outside_excluded(self):
        f = PeerJurisdictionFilter(
            resolver=self._resolver({"peer.us": "us"}),
            excluded={"cn", "ru"},
        )
        decision = f.evaluate("peer.us")
        assert decision.allow is True
        assert decision.detected_jurisdiction == "us"
        assert decision.reason == "allowed"

    def test_required_allowlist(self):
        f = PeerJurisdictionFilter(
            resolver=self._resolver({
                "peer.us": "us",
                "peer.de": "de",
                "peer.sg": "sg",
            }),
            required={"us", "de"},
        )
        assert f.evaluate("peer.us").allow is True
        assert f.evaluate("peer.de").allow is True
        # sg resolved but not in required → blocked
        d = f.evaluate("peer.sg")
        assert d.allow is False
        assert "not_in_required" in d.reason
        assert d.detected_jurisdiction == "sg"

    def test_excluded_wins_over_required(self):
        """If a code is in required AND resolved peer matches excluded,
        excluded wins. Configured via separate sets so this is edge
        case around future config mistakes, but evaluate() handles it
        defensively."""
        # Note: post_init check forbids a country in both sets — the
        # dataclass can't even be constructed. This tests that safety.
        with pytest.raises(GeoIPConfigError):
            PeerJurisdictionFilter(
                resolver=self._resolver({}),
                excluded={"us"},
                required={"us", "de"},
            )

    def test_resolution_failure_strict_blocks(self):
        f = PeerJurisdictionFilter(
            resolver=self._resolver({}),
            excluded={"cn"},
            policy="strict",
        )
        d = f.evaluate("unknown.example")
        assert d.allow is False
        assert d.reason == "resolution_failed_strict"
        assert d.detected_jurisdiction is None

    def test_resolution_failure_soft_allows(self):
        f = PeerJurisdictionFilter(
            resolver=self._resolver({}),
            excluded={"cn"},
            policy="soft",
        )
        d = f.evaluate("unknown.example")
        assert d.allow is True
        assert d.reason == "resolution_failed_soft"
        assert d.detected_jurisdiction is None

    def test_invalid_policy_rejected(self):
        with pytest.raises(GeoIPConfigError):
            PeerJurisdictionFilter(
                resolver=self._resolver({}),
                policy="permissive",  # type: ignore[arg-type]
            )

    def test_invalid_country_in_excluded_rejected(self):
        with pytest.raises(GeoIPConfigError):
            PeerJurisdictionFilter(
                resolver=self._resolver({}),
                excluded={"usa"},  # wrong length
            )

    def test_default_policy_is_strict(self):
        """Per R9 §8 no default jurisdiction lists, but a default
        resolution-failure policy is necessary somewhere. Strict is
        the safer default for an operator who opted into filtering."""
        f = PeerJurisdictionFilter(
            resolver=self._resolver({}), excluded={"cn"}
        )
        assert f.policy == "strict"

    def test_empty_excluded_all_allowed_under_soft(self):
        """With no excluded set + soft policy, every resolvable peer
        is allowed. Essentially a logging-only filter."""
        f = PeerJurisdictionFilter(
            resolver=self._resolver({"peer.cn": "cn"}),
            policy="soft",
        )
        d = f.evaluate("peer.cn")
        assert d.allow is True
        assert d.detected_jurisdiction == "cn"


class TestWebSocketTransportIntegration:
    def test_construction_accepts_filter(self):
        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport

        identity = generate_node_identity()
        resolver = StaticGeoIPResolver({"peer.example": "cn"})
        jfilter = PeerJurisdictionFilter(resolver=resolver, excluded={"cn"})

        transport = WebSocketTransport(
            identity=identity, port=0, jurisdiction_filter=jfilter
        )
        assert transport._jurisdiction_filter is jfilter

    def test_default_no_filter(self):
        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport

        identity = generate_node_identity()
        transport = WebSocketTransport(identity=identity, port=0)
        assert transport._jurisdiction_filter is None

    @pytest.mark.asyncio
    async def test_blocked_peer_skipped_before_transport(self):
        """Filter runs BEFORE the transport adapter. A blocked peer
        never triggers TCP I/O — we verify by injecting an adapter
        that would raise if called."""
        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport
        from prsm.node.transport_adapter import SocksAdapter
        from prsm.node.transport_adapter import TransportConnectError

        identity = generate_node_identity()
        resolver = StaticGeoIPResolver({"peer.cn.example": "cn"})
        jfilter = PeerJurisdictionFilter(resolver=resolver, excluded={"cn"})

        # Adapter that would raise loudly if called — we're asserting
        # the filter short-circuits before the adapter gets touched.
        adapter = SocksAdapter("127.0.0.1", 9050)
        adapter.open_connection = AsyncMock(
            side_effect=AssertionError(
                "adapter.open_connection called for a filtered peer!"
            )
        )

        transport = WebSocketTransport(
            identity=identity, port=0,
            transport_adapter=adapter,
            jurisdiction_filter=jfilter,
        )

        result = await transport.connect_to_peer(
            "ws://peer.cn.example:9001"
        )
        assert result is None
        adapter.open_connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_peer_reaches_transport(self):
        """Filter allows → adapter is called. Verify by mocking adapter
        to raise a benign error after being called."""
        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport
        from prsm.node.transport_adapter import SocksAdapter
        from prsm.node.transport_adapter import TransportConnectError

        identity = generate_node_identity()
        resolver = StaticGeoIPResolver({"peer.us.example": "us"})
        jfilter = PeerJurisdictionFilter(resolver=resolver, excluded={"cn"})

        adapter = SocksAdapter("127.0.0.1", 9050)
        adapter.open_connection = AsyncMock(
            side_effect=TransportConnectError("mock: not actually connecting")
        )

        transport = WebSocketTransport(
            identity=identity, port=0,
            transport_adapter=adapter,
            jurisdiction_filter=jfilter,
        )

        result = await transport.connect_to_peer(
            "ws://peer.us.example:9001"
        )
        # Connection ultimately failed (adapter raised) but filter
        # passed — confirmed by adapter being called.
        assert result is None
        adapter.open_connection.assert_called_once()
