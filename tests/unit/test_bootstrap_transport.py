"""Unit tests for prsm.node.bootstrap_transport — R9 Phase 6.2 Task 3."""
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.bootstrap_transport import (
    _build_txt_query,
    _encode_domain_name,
    _parse_txt_response,
    _parse_txt_rdata,
    _skip_name,
    _socks_proxy_url,
    make_doh_resolve_txt,
    make_https_get,
)
from prsm.node.transport_adapter import DirectAdapter, SocksAdapter


class TestSocksProxyUrl:
    def test_no_auth_socks5h(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        # rdns default True → socks5h
        assert _socks_proxy_url(adapter) == "socks5h://127.0.0.1:9050"

    def test_no_auth_socks5_explicit_rdns_false(self):
        adapter = SocksAdapter("127.0.0.1", 9050, rdns=False)
        assert _socks_proxy_url(adapter) == "socks5://127.0.0.1:9050"

    def test_with_username_only(self):
        adapter = SocksAdapter("proxy.example", 1080, username="alice")
        assert _socks_proxy_url(adapter) == "socks5h://alice@proxy.example:1080"

    def test_with_username_and_password(self):
        adapter = SocksAdapter(
            "proxy.example", 1080, username="alice", password="s3cret"
        )
        assert _socks_proxy_url(adapter) == "socks5h://alice:s3cret@proxy.example:1080"

    def test_password_with_special_chars_is_url_encoded(self):
        adapter = SocksAdapter(
            "proxy.example", 1080, username="user", password="p@ss w/slash"
        )
        # @ → %40, space → %20, / → %2F
        url = _socks_proxy_url(adapter)
        assert "user:p%40ss%20w%2Fslash@proxy.example:1080" in url

    def test_socks4_version(self):
        adapter = SocksAdapter("127.0.0.1", 1080, version=4)
        # SOCKS4 doesn't support rdns in httpx — scheme is "socks4"
        assert _socks_proxy_url(adapter) == "socks4://127.0.0.1:1080"


class TestMakeHttpsGet:
    def test_direct_adapter_produces_no_proxy_client(self):
        """With DirectAdapter, httpx is constructed without a proxy."""
        adapter = DirectAdapter()
        get = make_https_get(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = '{"hello": "world"}'
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = get("https://bootstrap.example/v1/peers.json")

        assert result == '{"hello": "world"}'
        mock_client_cls.assert_called_once()
        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs.get("proxy") is None

    def test_socks_adapter_injects_proxy_url(self):
        """With SocksAdapter, httpx is constructed with socks5h:// URL."""
        adapter = SocksAdapter("127.0.0.1", 9050)
        get = make_https_get(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = '{"ok": true}'
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = get("https://bootstrap.example/v1/peers.json")

        assert result == '{"ok": true}'
        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs.get("proxy") == "socks5h://127.0.0.1:9050"

    def test_respects_timeout_and_redirects_params(self):
        adapter = DirectAdapter()
        get = make_https_get(adapter, timeout=5.0, follow_redirects=False)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock(text="ok", raise_for_status=MagicMock())
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            get("https://example.com")

        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs["timeout"] == 5.0
        assert call_kwargs["follow_redirects"] is False

    def test_propagates_http_error(self):
        """4xx/5xx from the server propagate — HttpsBootstrapFetcher
        catches at a higher level, so we must raise here."""
        import httpx

        adapter = DirectAdapter()
        get = make_https_get(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404", request=MagicMock(), response=mock_response
            )
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                get("https://example.com/missing")


class TestDnsWireFormat:
    def test_encode_domain_basic(self):
        encoded = _encode_domain_name("example.com")
        # "example" (7 bytes) + "com" (3 bytes) + null terminator
        assert encoded == b"\x07example\x03com\x00"

    def test_encode_domain_strips_trailing_dot(self):
        assert _encode_domain_name("example.com.") == b"\x07example\x03com\x00"

    def test_encode_domain_strips_leading_dot(self):
        assert _encode_domain_name(".example.com") == b"\x07example\x03com\x00"

    def test_encode_domain_rejects_oversized_label(self):
        long_label = "a" * 64
        from prsm.node.transport_adapter import TransportConfigError
        with pytest.raises(TransportConfigError):
            _encode_domain_name(f"{long_label}.com")

    def test_build_txt_query_header_flags(self):
        query = _build_txt_query("example.com")
        # Header (12 bytes) + question (name + qtype + qclass)
        assert len(query) == 12 + len(b"\x07example\x03com\x00") + 4
        # flags field at offset 2-4: 0x0100 = standard query + RD
        flags = struct.unpack("!H", query[2:4])[0]
        assert flags == 0x0100
        # qdcount at offset 4-6: 1
        qdcount = struct.unpack("!H", query[4:6])[0]
        assert qdcount == 1

    def test_build_txt_query_rejects_empty_domain(self):
        from prsm.node.transport_adapter import TransportConfigError
        with pytest.raises(TransportConfigError):
            _build_txt_query("")

    def test_build_txt_query_rejects_oversized_domain(self):
        from prsm.node.transport_adapter import TransportConfigError
        with pytest.raises(TransportConfigError):
            _build_txt_query("a." * 127 + "com")  # > 253 bytes

    def test_parse_txt_rdata_single_string(self):
        # rdata: length-prefixed "hello"
        rdata = b"\x05hello"
        assert _parse_txt_rdata(rdata) == "hello"

    def test_parse_txt_rdata_multiple_concatenated(self):
        rdata = b"\x05hello\x05world"
        assert _parse_txt_rdata(rdata) == "helloworld"

    def test_parse_txt_rdata_empty(self):
        assert _parse_txt_rdata(b"") == ""

    def test_parse_txt_rdata_truncated_gracefully(self):
        """Length byte says 5 but only 2 bytes follow — don't crash."""
        assert _parse_txt_rdata(b"\x05ab") == ""


class TestParseTxtResponse:
    def _build_response(self, domain: str, txt_strings: list) -> bytes:
        """Synthesize a minimal DNS response with TXT records for testing."""
        # Header: id=0x1234, flags=0x8180 (response + RD + RA), qd=1, an=len(txt)
        header = struct.pack("!HHHHHH", 0x1234, 0x8180, 1, len(txt_strings), 0, 0)
        qname = _encode_domain_name(domain)
        question = qname + struct.pack("!HH", 16, 1)  # TXT, IN

        answers = bytearray()
        for text in txt_strings:
            # Use a compression pointer to the question name to keep msgs small
            answers.extend(b"\xc0\x0c")  # pointer to offset 12
            # type=16 (TXT), class=1 (IN), ttl=300, rdlength=?
            encoded = text.encode("utf-8")
            rdata = bytes([len(encoded)]) + encoded
            answers.extend(struct.pack("!HHIH", 16, 1, 300, len(rdata)))
            answers.extend(rdata)

        return header + question + bytes(answers)

    def test_single_txt_record(self):
        resp = self._build_response("_prsm.example.com", ["hello world"])
        assert _parse_txt_response(resp) == ["hello world"]

    def test_multiple_txt_records(self):
        resp = self._build_response(
            "_prsm.example.com", ["chunk1", "chunk2", "chunk3"]
        )
        assert _parse_txt_response(resp) == ["chunk1", "chunk2", "chunk3"]

    def test_too_short_returns_empty(self):
        assert _parse_txt_response(b"") == []
        assert _parse_txt_response(b"\x00\x00") == []

    def test_malformed_returns_empty(self):
        # Well-formed header but truncated answer section
        header = struct.pack("!HHHHHH", 0x1234, 0x8180, 0, 1, 0, 0)
        assert _parse_txt_response(header + b"\x00") == []


class TestSkipName:
    def test_simple_name(self):
        msg = b"\x07example\x03com\x00"
        assert _skip_name(msg, 0) == len(msg)

    def test_compression_pointer(self):
        # A pointer is 2 bytes total.
        msg = b"\xc0\x0c"
        assert _skip_name(msg, 0) == 2

    def test_name_with_padding(self):
        # Name followed by trailing bytes; _skip_name returns offset right
        # after the name's null terminator.
        msg = b"\x03www\x07example\x03com\x00XXXX"
        idx = _skip_name(msg, 0)
        assert idx == len(msg) - 4  # pointing at the first "X"


class TestMakeDohResolveTxt:
    def test_direct_adapter_no_proxy(self):
        adapter = DirectAdapter()
        resolve = make_doh_resolve_txt(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = b""  # empty DNS response → empty list
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            resolve("_prsm.example.com")

        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs.get("proxy") is None

    def test_socks_adapter_configures_proxy(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        resolve = make_doh_resolve_txt(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock(content=b"", raise_for_status=MagicMock())
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            resolve("_prsm.example.com")

        call_kwargs = mock_client_cls.call_args.kwargs
        assert call_kwargs.get("proxy") == "socks5h://127.0.0.1:9050"

    def test_returns_empty_on_http_failure(self):
        """DnsBootstrapFetcher expects [] on any failure; don't propagate."""
        adapter = DirectAdapter()
        resolve = make_doh_resolve_txt(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.side_effect = ConnectionError("DoH resolver unreachable")

            result = resolve("_prsm.example.com")

        assert result == []

    def test_sends_accept_dns_message_header(self):
        adapter = DirectAdapter()
        resolve = make_doh_resolve_txt(adapter)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock(content=b"", raise_for_status=MagicMock())
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            resolve("example.com")

        call_kwargs = mock_client_cls.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("Accept") == "application/dns-message"

    def test_query_round_trip_with_synthetic_dns_response(self):
        """End-to-end: parse a hand-built DNS TXT response through the
        DoH path. Verifies the resolver's wire-format handling."""
        adapter = DirectAdapter()
        resolve = make_doh_resolve_txt(adapter)

        # Build a valid DNS response containing "prsm=v1".
        def _build_dns_response():
            header = struct.pack("!HHHHHH", 0xABCD, 0x8180, 1, 1, 0, 0)
            qname = _encode_domain_name("_prsm.example.com")
            question = qname + struct.pack("!HH", 16, 1)
            answer = b"\xc0\x0c" + struct.pack("!HHIH", 16, 1, 300, 8)
            answer += b"\x07prsm=v1"
            return header + question + answer

        dns_response = _build_dns_response()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock(
                content=dns_response, raise_for_status=MagicMock()
            )
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = resolve("_prsm.example.com")

        assert result == ["prsm=v1"]

    def test_custom_resolver_url(self):
        adapter = DirectAdapter()
        resolve = make_doh_resolve_txt(
            adapter, resolver_url="https://dns.quad9.net/dns-query"
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_response = MagicMock(content=b"", raise_for_status=MagicMock())
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            resolve("_prsm.example.com")

        # client.get called with the configured resolver URL
        args, kwargs = mock_client.get.call_args
        assert args[0] == "https://dns.quad9.net/dns-query"


class TestIntegrationWithBootstrapFetcher:
    """Confirm the factories produce callables that plug into the
    existing HttpsBootstrapFetcher / DnsBootstrapFetcher API."""

    def test_https_get_compatible_with_fetcher(self):
        from prsm.node.bootstrap import HttpsBootstrapFetcher

        adapter = DirectAdapter()
        get = make_https_get(adapter)
        fetcher = HttpsBootstrapFetcher(url="https://example.com/peers.json", get=get)
        assert callable(fetcher.get)
        # Don't actually hit the network — just confirm the interface
        # accepts the callable.

    def test_doh_resolve_compatible_with_fetcher(self):
        from prsm.node.bootstrap import DnsBootstrapFetcher

        adapter = DirectAdapter()
        resolve = make_doh_resolve_txt(adapter)
        fetcher = DnsBootstrapFetcher(
            domain="_prsm.example.com", resolve_txt=resolve
        )
        assert callable(fetcher.resolve_txt)
