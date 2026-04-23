"""Unit tests for prsm.node.transport_adapter — R9 Phase 6.2 Tasks 1-2."""
from __future__ import annotations

import asyncio
import socket
from unittest.mock import patch

import pytest

from prsm.node.transport_adapter import (
    DirectAdapter,
    SocksAdapter,
    SocksProxyConfig,
    TransportAdapter,
    TransportConfigError,
    TransportConnectError,
    adapter_from_config,
)


class TestProtocolConformance:
    def test_direct_adapter_satisfies_protocol(self):
        adapter = DirectAdapter()
        assert isinstance(adapter, TransportAdapter)
        assert adapter.name == "direct"

    def test_socks_adapter_satisfies_protocol(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        assert isinstance(adapter, TransportAdapter)
        assert adapter.name == "socks5"

    def test_socks4_adapter_name(self):
        adapter = SocksAdapter("127.0.0.1", 1080, version=4)
        assert adapter.name == "socks4"


class TestSocksProxyConfig:
    def test_valid_socks5(self):
        cfg = SocksProxyConfig(host="127.0.0.1", port=9050)
        assert cfg.rdns is True
        assert cfg.version == 5

    def test_valid_socks5_with_auth(self):
        cfg = SocksProxyConfig(
            host="proxy.example.com", port=1080, username="u", password="p"
        )
        assert cfg.username == "u"
        assert cfg.password == "p"

    def test_empty_host_rejected(self):
        with pytest.raises(TransportConfigError):
            SocksProxyConfig(host="", port=1080)

    def test_bad_port_rejected(self):
        with pytest.raises(TransportConfigError):
            SocksProxyConfig(host="127.0.0.1", port=0)
        with pytest.raises(TransportConfigError):
            SocksProxyConfig(host="127.0.0.1", port=65536)
        with pytest.raises(TransportConfigError):
            SocksProxyConfig(host="127.0.0.1", port=-1)

    def test_bad_version_rejected(self):
        with pytest.raises(TransportConfigError):
            SocksProxyConfig(host="127.0.0.1", port=1080, version=6)

    def test_socks4_with_auth_rejected(self):
        """SOCKS4 doesn't support auth; flag the config error early."""
        with pytest.raises(TransportConfigError):
            SocksProxyConfig(
                host="127.0.0.1", port=1080, username="u", version=4
            )


class TestDirectAdapter:
    @pytest.mark.asyncio
    async def test_rejects_empty_host(self):
        adapter = DirectAdapter()
        with pytest.raises(TransportConfigError):
            await adapter.open_connection("", 9001)

    @pytest.mark.asyncio
    async def test_rejects_bad_port(self):
        adapter = DirectAdapter()
        with pytest.raises(TransportConfigError):
            await adapter.open_connection("127.0.0.1", 0)
        with pytest.raises(TransportConfigError):
            await adapter.open_connection("127.0.0.1", 65536)

    @pytest.mark.asyncio
    async def test_opens_connection_to_listening_socket(self):
        """End-to-end: start an echo server, adapter connects, we send +
        receive the echo round-trip. Confirms the returned socket is in a
        usable connected state."""
        adapter = DirectAdapter()

        # Spin up an echo server on an ephemeral port.
        server_socket = socket.socket()
        server_socket.bind(("127.0.0.1", 0))
        port = server_socket.getsockname()[1]
        server_socket.listen(1)
        server_socket.setblocking(False)

        async def echo_once():
            loop = asyncio.get_running_loop()
            conn, _ = await loop.sock_accept(server_socket)
            data = await loop.sock_recv(conn, 1024)
            await loop.sock_sendall(conn, data)
            conn.close()

        server_task = asyncio.create_task(echo_once())

        try:
            sock = await adapter.open_connection("127.0.0.1", port)
            loop = asyncio.get_running_loop()
            await loop.sock_sendall(sock, b"hello")
            echoed = await loop.sock_recv(sock, 1024)
            assert echoed == b"hello"
            sock.close()
            await server_task
        finally:
            server_socket.close()

    @pytest.mark.asyncio
    async def test_timeout_on_unreachable(self):
        """Connecting to a non-routable IP should time out, not hang."""
        adapter = DirectAdapter()
        # TEST-NET-1 per RFC 5737 — guaranteed non-routable.
        with pytest.raises(TransportConnectError):
            await adapter.open_connection("192.0.2.1", 9001, timeout=0.5)

    @pytest.mark.asyncio
    async def test_connection_refused_raises_connect_error(self):
        """Connecting to a localhost port with nothing listening yields
        TransportConnectError, not a bare OSError bubbled up."""
        adapter = DirectAdapter()
        # Pick a port with nothing listening — reliably refused.
        with pytest.raises(TransportConnectError):
            # Use port 1 (reserved, nothing can listen); macOS + Linux both
            # give ECONNREFUSED or EACCES without delay.
            await adapter.open_connection("127.0.0.1", 1, timeout=2.0)


class TestSocksAdapter:
    def test_construction_validates_proxy(self):
        with pytest.raises(TransportConfigError):
            SocksAdapter("", 9050)
        with pytest.raises(TransportConfigError):
            SocksAdapter("127.0.0.1", 0)

    @pytest.mark.asyncio
    async def test_rejects_empty_target_host(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        with pytest.raises(TransportConfigError):
            await adapter.open_connection("", 9001)

    @pytest.mark.asyncio
    async def test_rejects_bad_target_port(self):
        adapter = SocksAdapter("127.0.0.1", 9050)
        with pytest.raises(TransportConfigError):
            await adapter.open_connection("example.com", 0)
        with pytest.raises(TransportConfigError):
            await adapter.open_connection("example.com", 99999)

    @pytest.mark.asyncio
    async def test_reports_connect_error_when_proxy_unreachable(self):
        """If the configured SOCKS proxy isn't running, we get a
        TransportConnectError with context rather than a raw python-socks
        exception leaking through."""
        # Use a port nothing is listening on.
        adapter = SocksAdapter("127.0.0.1", 1)
        with pytest.raises(TransportConnectError) as excinfo:
            await adapter.open_connection("example.com", 443, timeout=2.0)
        msg = str(excinfo.value)
        # Should name the proxy and the target for operator debugging.
        assert "127.0.0.1:1" in msg
        assert "example.com:443" in msg


class TestAdapterFromConfig:
    def test_direct(self):
        adapter = adapter_from_config("direct")
        assert isinstance(adapter, DirectAdapter)

    def test_direct_aliases(self):
        assert isinstance(adapter_from_config(""), DirectAdapter)
        assert isinstance(adapter_from_config("none"), DirectAdapter)
        assert isinstance(adapter_from_config("Direct"), DirectAdapter)
        assert isinstance(adapter_from_config(" direct "), DirectAdapter)

    def test_socks5_with_host_port(self):
        adapter = adapter_from_config(
            "socks5", proxy_host="127.0.0.1", proxy_port=9050
        )
        assert isinstance(adapter, SocksAdapter)
        assert adapter.name == "socks5"

    def test_socks5_with_auth(self):
        adapter = adapter_from_config(
            "socks5",
            proxy_host="proxy.example.com",
            proxy_port=1080,
            proxy_username="user",
            proxy_password="pass",
        )
        assert isinstance(adapter, SocksAdapter)
        assert adapter.config.username == "user"
        assert adapter.config.password == "pass"

    def test_socks5_missing_proxy_host(self):
        with pytest.raises(TransportConfigError):
            adapter_from_config("socks5", proxy_port=1080)

    def test_socks5_missing_proxy_port(self):
        with pytest.raises(TransportConfigError):
            adapter_from_config("socks5", proxy_host="127.0.0.1")

    def test_socks4(self):
        adapter = adapter_from_config(
            "socks4", proxy_host="127.0.0.1", proxy_port=1080
        )
        assert isinstance(adapter, SocksAdapter)
        assert adapter.name == "socks4"

    def test_unknown_transport_rejected(self):
        with pytest.raises(TransportConfigError) as excinfo:
            adapter_from_config("wireguard")
        assert "wireguard" in str(excinfo.value)
        # Message should list recognized values for operator guidance.
        assert "direct" in str(excinfo.value)

    def test_socks_alias_for_socks5(self):
        """Plain 'socks' resolves to SOCKS5 — the modern default."""
        adapter = adapter_from_config(
            "socks", proxy_host="127.0.0.1", proxy_port=1080
        )
        assert isinstance(adapter, SocksAdapter)
        assert adapter.name == "socks5"


class TestWebSocketTransportIntegration:
    """Verify WebSocketTransport accepts + uses the transport adapter."""

    def test_default_adapter_is_direct(self):
        """Constructing WebSocketTransport without an adapter yields
        DirectAdapter — preserves pre-R9 behavior."""
        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport

        identity = generate_node_identity()
        transport = WebSocketTransport(identity=identity, port=0)
        assert isinstance(transport._transport_adapter, DirectAdapter)

    def test_custom_adapter_accepted(self):
        """Passing a SocksAdapter at construction time is preserved."""
        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport

        identity = generate_node_identity()
        adapter = SocksAdapter("127.0.0.1", 9050)
        transport = WebSocketTransport(
            identity=identity, port=0, transport_adapter=adapter
        )
        assert transport._transport_adapter is adapter
        assert transport._transport_adapter.name == "socks5"

    @pytest.mark.asyncio
    async def test_socks_adapter_invoked_for_outbound_connect(self):
        """When a non-direct adapter is configured, connect_to_peer
        routes through the adapter's open_connection — NOT the direct
        websockets.connect fast path. Verified by mocking open_connection
        and asserting it gets called with the parsed host/port."""
        from unittest.mock import AsyncMock

        from prsm.node.identity import generate_node_identity
        from prsm.node.transport import WebSocketTransport

        identity = generate_node_identity()
        adapter = SocksAdapter("127.0.0.1", 9050)
        # Mock the adapter's open_connection to raise — we only want to
        # prove it was called with the right args. The actual websocket
        # handshake doesn't need to complete for this test.
        adapter.open_connection = AsyncMock(
            side_effect=TransportConnectError("mock: not actually connecting")
        )

        transport = WebSocketTransport(
            identity=identity, port=0, transport_adapter=adapter
        )

        # Adapter raises → connect_to_peer returns None gracefully.
        result = await transport.connect_to_peer("ws://peer.example.com:9001")
        assert result is None

        # Adapter was called with the parsed host+port, NOT the URI.
        adapter.open_connection.assert_called_once()
        call_args = adapter.open_connection.call_args
        assert call_args.args[0] == "peer.example.com"
        assert call_args.args[1] == 9001
