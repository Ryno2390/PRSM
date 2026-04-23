"""R9 Phase 6.2 Task 1-2 — Pluggable-transport adapter interface.

R9-SCOPING-1 §5.1-§5.3 specifies a pluggable-transport layer so a PRSM
node operator in a censoring jurisdiction can route the node's outbound
connections through Tor, V2Ray, Trojan, Shadowsocks, or similar
circumvention infrastructure — without making any of those a hard
protocol dependency.

This module defines the ``TransportAdapter`` protocol plus two reference
implementations:

- ``DirectAdapter`` — the default. No transformation of the outbound
  connection. Equivalent to pre-R9 behavior.
- ``SocksAdapter`` — routes through a local SOCKS5 proxy. This single
  adapter covers Tor (via ``torsocks`` or Tor's ControlPort SOCKS),
  V2Ray/VMess/VLESS, Trojan, Shadowsocks, and XRay — all of which
  expose a local SOCKS5 interface by convention.

What is NOT in this module (explicitly deferred to R9 Phase 6.2 Tasks
3-4): bootstrap-list discovery over alternate transports, FTNS RPC
routing through configured transport. Those integration points need
their own follow-on commits; this module ships the transport-layer
primitive they'll build on.

What is NOT ever in this module (per R9 §8 Foundation boundary
commitments): active Tor bridge distribution, curated VPN-provider
lists, jurisdiction-specific "safe mode" configuration presets. The
Foundation ships mechanism neutrally; users configure transport
selection based on their own jurisdictional threat model.

Usage
-----

.. code-block:: python

    # Direct (default):
    adapter = DirectAdapter()

    # Tor (assuming tor daemon running on localhost:9050):
    adapter = SocksAdapter(
        proxy_host="127.0.0.1",
        proxy_port=9050,
    )

    # V2Ray / Trojan / Shadowsocks (local proxy typical on 1080):
    adapter = SocksAdapter(
        proxy_host="127.0.0.1",
        proxy_port=1080,
        username="optional",
        password="optional",
    )

    # Open a socket through the adapter:
    sock = await adapter.open_connection("peer.example.com", 9001)
    # sock is a connected asyncio reader/writer pair that the caller
    # then hands to websockets.connect(sock=...) or similar.
"""
from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable


class TransportError(Exception):
    """Base for transport-adapter failures."""


class TransportConfigError(TransportError):
    """Adapter configured with invalid parameters."""


class TransportConnectError(TransportError):
    """Connection attempt failed at the transport layer.

    Covers: TCP connect timeout, proxy authentication failure, SOCKS
    protocol error, upstream unreachable reported by proxy.
    """


@runtime_checkable
class TransportAdapter(Protocol):
    """Protocol for outbound-connection transport adapters.

    Implementations open a TCP-like bytestream to ``(host, port)`` and
    return the underlying socket so the caller (WebSocketTransport,
    libp2p transport, HTTP client for FTNS RPC) can layer its own
    protocol on top.

    Implementations MUST:

    - Respect ``timeout`` (seconds) as an end-to-end wall-clock budget
      covering DNS resolution + proxy handshake + upstream CONNECT.
    - Raise ``TransportConnectError`` on failure, not propagate
      library-specific exceptions to callers.
    - Return a socket in the connected state, blocking semantics set
      by the caller.
    - Be safe to call concurrently from multiple asyncio tasks —
      implementations SHOULD NOT share mutable per-adapter state
      between concurrent ``open_connection`` calls.
    """

    async def open_connection(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 30.0,
    ) -> socket.socket:
        """Open a connected socket to ``(host, port)`` through this transport.

        :param host: Target hostname or IP.
        :param port: Target port.
        :param timeout: Wall-clock budget for the full connect sequence.
        :returns: Connected socket handed off to the caller.
        :raises TransportConnectError: Any transport-layer failure.
        """
        ...  # pragma: no cover

    @property
    def name(self) -> str:
        """Short identifier for logging/metrics (e.g., ``"direct"``, ``"socks"``)."""
        ...  # pragma: no cover


# ──────────────────────────────────────────────────────────────────────
# DirectAdapter — no transformation; default behavior
# ──────────────────────────────────────────────────────────────────────


class DirectAdapter:
    """The no-op adapter. Opens a direct TCP connection to the target.

    Equivalent to pre-R9 behavior. The default for users not operating
    in censoring jurisdictions.
    """

    name = "direct"

    async def open_connection(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 30.0,
    ) -> socket.socket:
        if not host:
            raise TransportConfigError("host must be non-empty")
        if port <= 0 or port > 65535:
            raise TransportConfigError(f"port must be 1..65535; got {port}")
        loop = asyncio.get_running_loop()
        try:
            sock = await asyncio.wait_for(
                _tcp_connect(loop, host, port),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise TransportConnectError(
                f"direct connect timed out after {timeout}s to {host}:{port}"
            ) from exc
        except OSError as exc:
            raise TransportConnectError(
                f"direct connect failed to {host}:{port}: {exc}"
            ) from exc
        return sock


async def _tcp_connect(
    loop: asyncio.AbstractEventLoop, host: str, port: int
) -> socket.socket:
    """Open a direct TCP connection. Loop-level primitive isolated for mocking."""
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not infos:
        raise OSError(f"no address info for {host}")
    family, socktype, proto, _, sockaddr = infos[0]
    sock = socket.socket(family, socktype, proto)
    sock.setblocking(False)
    try:
        await loop.sock_connect(sock, sockaddr)
    except Exception:
        sock.close()
        raise
    return sock


# ──────────────────────────────────────────────────────────────────────
# SocksAdapter — routes through a local SOCKS5 proxy
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SocksProxyConfig:
    """Config for a SOCKS5 (or SOCKS4) proxy.

    :param host: Proxy host (typically ``127.0.0.1`` for local proxy).
    :param port: Proxy port. Conventions: 9050 (Tor), 1080 (V2Ray/Trojan/
        Shadowsocks client default), 1086 / 1081 (operator choice).
    :param username: Optional SOCKS5 username.
    :param password: Optional SOCKS5 password.
    :param rdns: If True, resolve the target hostname at the proxy
        (remote DNS). Recommended for Tor + privacy-sensitive
        configurations — prevents local DNS leakage. Default True.
    :param version: SOCKS protocol version. Default 5. SOCKS4 is
        supported for legacy tools but lacks auth + IPv6; avoid unless
        required by a specific proxy.
    """

    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    rdns: bool = True
    version: int = 5

    def __post_init__(self) -> None:
        if not self.host:
            raise TransportConfigError("SOCKS proxy host must be non-empty")
        if self.port <= 0 or self.port > 65535:
            raise TransportConfigError(
                f"SOCKS proxy port must be 1..65535; got {self.port}"
            )
        if self.version not in (4, 5):
            raise TransportConfigError(
                f"SOCKS version must be 4 or 5; got {self.version}"
            )
        if self.version == 4 and (self.username or self.password):
            raise TransportConfigError(
                "SOCKS4 does not support username/password auth"
            )


class SocksAdapter:
    """Routes outbound connections through a local SOCKS5 (or SOCKS4) proxy.

    Covers Tor (``127.0.0.1:9050`` by default for ``torsocks``), V2Ray,
    VMess, VLESS, Trojan, Shadowsocks, XRay — all of which expose a
    local SOCKS5 interface by convention.

    The adapter does NOT manage the proxy daemon itself. The operator
    runs ``tor``, ``v2ray``, ``trojan-client``, etc. separately and
    configures the adapter with the proxy's listen address.

    See R9-SCOPING-1 §5 for the rationale behind SOCKS-as-abstraction.
    """

    def __init__(
        self,
        proxy_host: str,
        proxy_port: int,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        rdns: bool = True,
        version: int = 5,
    ) -> None:
        self.config = SocksProxyConfig(
            host=proxy_host,
            port=proxy_port,
            username=username,
            password=password,
            rdns=rdns,
            version=version,
        )

    @property
    def name(self) -> str:
        return f"socks{self.config.version}"

    async def open_connection(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 30.0,
    ) -> socket.socket:
        # Lazy import so DirectAdapter users don't pay the import cost or
        # require python-socks to be installed just to operate the node.
        try:
            from python_socks import ProxyType
            from python_socks.async_.asyncio import Proxy
        except ImportError as exc:  # pragma: no cover - dep-availability
            raise TransportConnectError(
                "SOCKS transport requires python-socks[asyncio]. "
                "Install with: pip install 'python-socks[asyncio]'"
            ) from exc

        if not host:
            raise TransportConfigError("target host must be non-empty")
        if port <= 0 or port > 65535:
            raise TransportConfigError(f"target port must be 1..65535; got {port}")

        proxy_type = ProxyType.SOCKS5 if self.config.version == 5 else ProxyType.SOCKS4
        proxy = Proxy.create(
            proxy_type=proxy_type,
            host=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            rdns=self.config.rdns,
        )
        try:
            sock = await proxy.connect(
                dest_host=host,
                dest_port=port,
                timeout=timeout,
            )
        except Exception as exc:
            # python-socks raises multiple exception types; normalize.
            raise TransportConnectError(
                f"SOCKS{self.config.version} connect via "
                f"{self.config.host}:{self.config.port} to {host}:{port} "
                f"failed: {exc}"
            ) from exc
        return sock


# ──────────────────────────────────────────────────────────────────────
# Convenience: resolve an adapter from environment/config
# ──────────────────────────────────────────────────────────────────────


def adapter_from_config(
    transport_type: str,
    *,
    proxy_host: Optional[str] = None,
    proxy_port: Optional[int] = None,
    proxy_username: Optional[str] = None,
    proxy_password: Optional[str] = None,
) -> TransportAdapter:
    """Build an adapter from a config-driven ``transport_type`` string.

    Recognized values:

    - ``"direct"`` — ``DirectAdapter``.
    - ``"socks5"`` — ``SocksAdapter(version=5)``. Requires
      ``proxy_host`` + ``proxy_port``.
    - ``"socks4"`` — ``SocksAdapter(version=4)``. Requires
      ``proxy_host`` + ``proxy_port``. No auth supported.

    :raises TransportConfigError: if ``transport_type`` is unknown or if
        a SOCKS variant is selected without proxy host/port.
    """
    transport_type = transport_type.lower().strip()
    if transport_type in ("direct", "", "none"):
        return DirectAdapter()
    if transport_type in ("socks5", "socks"):
        if not proxy_host or proxy_port is None:
            raise TransportConfigError(
                "socks5 transport requires proxy_host + proxy_port"
            )
        return SocksAdapter(
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            username=proxy_username,
            password=proxy_password,
            version=5,
        )
    if transport_type == "socks4":
        if not proxy_host or proxy_port is None:
            raise TransportConfigError(
                "socks4 transport requires proxy_host + proxy_port"
            )
        return SocksAdapter(
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            version=4,
        )
    raise TransportConfigError(
        f"unknown transport_type {transport_type!r}; "
        f"recognized: direct, socks5, socks4"
    )


__all__ = [
    "TransportAdapter",
    "TransportError",
    "TransportConfigError",
    "TransportConnectError",
    "DirectAdapter",
    "SocksAdapter",
    "SocksProxyConfig",
    "adapter_from_config",
]
