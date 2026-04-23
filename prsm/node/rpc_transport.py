"""R9 Phase 6.2 Task 4 — FTNS RPC endpoint routing over transport.

Completes the R9-SCOPING-1 §5.2 integration surface. The web3.py
settlement layer (StakeManager, ProvenanceRegistry, EmissionClient,
etc.) uses ``Web3(Web3.HTTPProvider(rpc_url))`` to talk to Base L2 RPC
endpoints. By default that's a direct HTTPS connection — for operators
in censoring jurisdictions, it needs to route through the same
TransportAdapter as bootstrap fetching + P2P connections.

Web3.HTTPProvider accepts a ``session=`` kwarg that takes any
``requests.Session``-compatible object. We build a session with
proxies configured from the adapter, then hand it to HTTPProvider —
all subsequent RPC traffic (eth_call, eth_sendRawTransaction, etc.)
flows through the configured transport.

Why requests.Session and not httpx
----------------------------------
web3.py is built on requests internally. Its HTTPProvider expects the
requests-specific Session API (headers / proxies / adapters dict).
httpx.Client is API-incompatible. We keep bootstrap_transport on
httpx (lighter, async-friendly) and RPC on requests (web3-native).

Proxy scheme choice
-------------------
requests + PySocks support ``socks5h://`` (remote DNS) out of the
box. Same URL format as bootstrap_transport. Credentials URL-encoded
per RFC 3986.

Foundation boundary
-------------------
Per R9 §8, the Foundation ships this mechanism neutrally. It does
NOT ship an operator config that selects a specific RPC endpoint
known to be accessible under any specific adversary's blockade. The
operator provides both the RPC URL and the transport; this module
wires them together.

Usage
-----

.. code-block:: python

    from web3 import Web3
    from prsm.node.rpc_transport import make_web3_http_provider
    from prsm.node.transport_adapter import SocksAdapter

    adapter = SocksAdapter("127.0.0.1", 9050)  # operator's Tor daemon

    provider = make_web3_http_provider(
        adapter,
        rpc_url="https://mainnet.base.org",
    )
    web3 = Web3(provider)

    # Drop-in replacement — all subsequent RPC calls route through Tor.
    balance = web3.eth.get_balance("0x...")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from prsm.node.bootstrap_transport import _socks_proxy_url
from prsm.node.transport_adapter import (
    DirectAdapter,
    SocksAdapter,
    TransportAdapter,
    TransportConfigError,
)

if TYPE_CHECKING:
    import requests
    from web3.providers.rpc import HTTPProvider

logger = logging.getLogger(__name__)


# Settlement-layer RPC timeouts are longer than bootstrap because a
# transaction broadcast + confirmation wait can legitimately take
# minutes on L2. Operators override via the provider's request_kwargs.
_DEFAULT_RPC_TIMEOUT = 120.0


def make_requests_session_for_adapter(
    adapter: TransportAdapter,
    *,
    timeout: float = _DEFAULT_RPC_TIMEOUT,
) -> "requests.Session":
    """Build a ``requests.Session`` routed through ``adapter``.

    :param adapter: TransportAdapter. DirectAdapter leaves the session
        with no proxies configured (default behavior).  SocksAdapter
        synthesizes a ``socks5h://`` URL and assigns it to both the
        ``http`` and ``https`` proxy entries on the session.
    :param timeout: Default request timeout, applied as a
        ``Session.request`` default via a small adapter wrapper.
    :returns: ``requests.Session`` ready to hand to
        ``Web3.HTTPProvider(session=...)``.

    :raises TransportConfigError: if the adapter type is unknown.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise TransportConfigError(
            "RPC-over-transport requires requests. "
            "Install with: pip install 'requests[socks]'"
        ) from exc

    session = requests.Session()

    if isinstance(adapter, DirectAdapter):
        # No proxies configured; session is a vanilla requests.Session.
        proxy_url: Optional[str] = None
    elif isinstance(adapter, SocksAdapter):
        proxy_url = _socks_proxy_url(adapter)
        session.proxies.update({
            "http": proxy_url,
            "https": proxy_url,
        })
        # PySocks needs to be available or requests will raise
        # InvalidSchema on the first proxied call. Fail fast with a
        # clear message at session-creation time if the operator's
        # env is missing the dep.
        try:
            import socks  # noqa: F401 - just checking availability
        except ImportError as exc:  # pragma: no cover
            raise TransportConfigError(
                "SOCKS proxy support for requests requires PySocks. "
                "Install with: pip install 'requests[socks]'"
            ) from exc
    else:
        logger.warning(
            "make_requests_session_for_adapter: unknown adapter type %r; "
            "falling back to direct session", adapter.name,
        )
        proxy_url = None

    # Store adapter metadata on the session for telemetry / debugging.
    # Namespaced attribute so we don't collide with requests internals.
    session._prsm_transport_adapter = adapter.name  # type: ignore[attr-defined]
    session._prsm_default_timeout = timeout  # type: ignore[attr-defined]
    session._prsm_proxy_url = proxy_url  # type: ignore[attr-defined]

    return session


def make_web3_http_provider(
    adapter: TransportAdapter,
    rpc_url: str,
    *,
    timeout: float = _DEFAULT_RPC_TIMEOUT,
    request_kwargs: Optional[dict] = None,
) -> "HTTPProvider":
    """Build a ``Web3.HTTPProvider`` routed through ``adapter``.

    Convenience wrapper that combines ``make_requests_session_for_adapter``
    with Web3's provider construction.

    :param adapter: TransportAdapter.
    :param rpc_url: JSON-RPC endpoint URL (e.g., ``https://mainnet.base.org``).
    :param timeout: Default request timeout in seconds.
    :param request_kwargs: Additional kwargs passed through to
        ``HTTPProvider(request_kwargs=...)``. The ``timeout`` key in
        this dict overrides the top-level ``timeout`` parameter if
        both are provided.
    :returns: ``web3.providers.rpc.HTTPProvider`` ready to hand to
        ``Web3(...)``.
    """
    try:
        from web3 import Web3
    except ImportError as exc:  # pragma: no cover
        raise TransportConfigError(
            "RPC provider construction requires web3. "
            "Install with: pip install web3"
        ) from exc

    if not rpc_url or not isinstance(rpc_url, str):
        raise TransportConfigError(
            f"rpc_url must be a non-empty string; got {rpc_url!r}"
        )

    session = make_requests_session_for_adapter(adapter, timeout=timeout)

    # Merge timeout into request_kwargs (if not already set).
    merged_kwargs = dict(request_kwargs or {})
    merged_kwargs.setdefault("timeout", timeout)

    return Web3.HTTPProvider(
        endpoint_uri=rpc_url,
        session=session,
        request_kwargs=merged_kwargs,
    )


__all__ = [
    "make_requests_session_for_adapter",
    "make_web3_http_provider",
]
