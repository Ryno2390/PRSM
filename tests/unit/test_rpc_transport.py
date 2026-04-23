"""Unit tests for prsm.node.rpc_transport — R9 Phase 6.2 Task 4."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prsm.node.rpc_transport import (
    make_requests_session_for_adapter,
    make_web3_http_provider,
)
from prsm.node.transport_adapter import (
    DirectAdapter,
    SocksAdapter,
    TransportConfigError,
)


class TestMakeRequestsSession:
    def test_direct_adapter_no_proxies(self):
        adapter = DirectAdapter()
        session = make_requests_session_for_adapter(adapter)
        # Vanilla session — no proxies configured.
        assert session.proxies == {} or not session.proxies.get("https")
        # Telemetry metadata attached.
        assert session._prsm_transport_adapter == "direct"
        assert session._prsm_proxy_url is None

    def test_socks_adapter_configures_socks5h_proxy(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        session = make_requests_session_for_adapter(adapter)
        # Both http + https route through the SOCKS proxy.
        assert session.proxies["http"] == "socks5h://127.0.0.1:9050"
        assert session.proxies["https"] == "socks5h://127.0.0.1:9050"
        assert session._prsm_transport_adapter == "socks5"
        assert session._prsm_proxy_url == "socks5h://127.0.0.1:9050"

    def test_socks_adapter_with_credentials(self):
        adapter = SocksAdapter(
            "proxy.example.com",
            1080,
            username="alice",
            password="s3cret",
        )
        session = make_requests_session_for_adapter(adapter)
        expected = "socks5h://alice:s3cret@proxy.example.com:1080"
        assert session.proxies["http"] == expected
        assert session.proxies["https"] == expected

    def test_socks4_adapter(self):
        adapter = SocksAdapter("127.0.0.1", 1080, version=4)
        session = make_requests_session_for_adapter(adapter)
        # SOCKS4 doesn't get the -h suffix (remote DNS is SOCKS5-only).
        assert session.proxies["https"] == "socks4://127.0.0.1:1080"

    def test_socks_adapter_rdns_false_uses_socks5_not_h(self):
        """rdns=False on the adapter should produce socks5:// not socks5h://.
        The local-DNS variant leaks the target hostname to the ISP's
        resolver — the operator opted in."""
        adapter = SocksAdapter("127.0.0.1", 9050, rdns=False)
        session = make_requests_session_for_adapter(adapter)
        assert session.proxies["https"] == "socks5://127.0.0.1:9050"

    def test_default_timeout_stored_on_session(self):
        adapter = DirectAdapter()
        session = make_requests_session_for_adapter(adapter, timeout=60.0)
        assert session._prsm_default_timeout == 60.0


class TestMakeWeb3HttpProvider:
    def test_direct_adapter_produces_direct_provider(self):
        adapter = DirectAdapter()
        provider = make_web3_http_provider(
            adapter, "https://mainnet.base.org"
        )
        # Provider exists and wraps an endpoint URI.
        assert provider.endpoint_uri == "https://mainnet.base.org"
        # Session is attached + has no proxies.
        session = provider._request_session_manager.cache_and_return_session(
            provider.endpoint_uri
        )
        assert not session.proxies.get("https")

    def test_socks_adapter_provider_session_has_proxy(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        provider = make_web3_http_provider(
            adapter, "https://mainnet.base.org"
        )
        session = provider._request_session_manager.cache_and_return_session(
            provider.endpoint_uri
        )
        assert session.proxies.get("https") == "socks5h://127.0.0.1:9050"

    def test_rejects_empty_rpc_url(self):
        adapter = DirectAdapter()
        with pytest.raises(TransportConfigError):
            make_web3_http_provider(adapter, "")

    def test_rejects_non_string_rpc_url(self):
        adapter = DirectAdapter()
        with pytest.raises(TransportConfigError):
            make_web3_http_provider(adapter, None)  # type: ignore[arg-type]

    def test_request_kwargs_timeout_forwarded(self):
        """Web3 provider's request_kwargs should carry the timeout default
        so eth_call / eth_sendRawTransaction use it."""
        adapter = DirectAdapter()
        provider = make_web3_http_provider(
            adapter, "https://mainnet.base.org", timeout=45.0
        )
        assert provider.get_request_kwargs()["timeout"] == 45.0

    def test_user_request_kwargs_override_timeout(self):
        """Explicit request_kwargs dict with timeout overrides the
        module default."""
        adapter = DirectAdapter()
        provider = make_web3_http_provider(
            adapter,
            "https://mainnet.base.org",
            timeout=120.0,
            request_kwargs={"timeout": 10.0, "verify": False},
        )
        kwargs = provider.get_request_kwargs()
        assert kwargs["timeout"] == 10.0
        assert kwargs["verify"] is False


class TestDropInReplacement:
    """Confirm the transport-routed provider is API-compatible with the
    pre-R9 direct Web3(Web3.HTTPProvider(rpc_url)) pattern."""

    def test_web3_instantiation_with_direct_provider(self):
        from web3 import Web3

        adapter = DirectAdapter()
        provider = make_web3_http_provider(
            adapter, "https://mainnet.base.org"
        )
        web3 = Web3(provider)
        # Drop-in — standard Web3 API surface available.
        assert hasattr(web3, "eth")
        assert hasattr(web3.eth, "get_block")

    def test_web3_instantiation_with_socks_provider(self):
        from web3 import Web3

        adapter = SocksAdapter("127.0.0.1", 9050)
        provider = make_web3_http_provider(
            adapter, "https://mainnet.base.org"
        )
        web3 = Web3(provider)
        assert hasattr(web3, "eth")
        # The session carries the proxy config — any subsequent
        # web3.eth.call / send_transaction / etc. goes through SOCKS.
        session = provider._request_session_manager.cache_and_return_session(
            provider.endpoint_uri
        )
        assert session.proxies.get("https") == "socks5h://127.0.0.1:9050"

    def test_proxy_url_not_leaked_to_direct_sessions(self):
        """Two providers constructed with different adapters must not
        share proxy config — isolation check."""
        from web3 import Web3

        direct_adapter = DirectAdapter()
        socks_adapter = SocksAdapter("127.0.0.1", 9050)

        direct_provider = make_web3_http_provider(
            direct_adapter, "https://example.com"
        )
        socks_provider = make_web3_http_provider(
            socks_adapter, "https://example.com"
        )

        direct_session = direct_provider._request_session_manager.cache_and_return_session(
            direct_provider.endpoint_uri
        )
        socks_session = socks_provider._request_session_manager.cache_and_return_session(
            socks_provider.endpoint_uri
        )

        assert not direct_session.proxies.get("https")
        assert socks_session.proxies.get("https") == "socks5h://127.0.0.1:9050"


class TestUnknownAdapter:
    def test_unknown_adapter_falls_back_to_direct(self):
        """Arbitrary adapter types (not DirectAdapter / SocksAdapter)
        should not crash — fall back to no-proxy behavior with a
        warning."""

        class CustomAdapter:
            name = "custom"

            async def open_connection(self, host, port, *, timeout=30.0):
                raise NotImplementedError

        adapter = CustomAdapter()
        session = make_requests_session_for_adapter(adapter)  # type: ignore[arg-type]
        # No proxies — unknown adapter fallback behavior.
        assert not session.proxies.get("https")
