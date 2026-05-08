"""Aggregate endpoint resolver — node_id → aggregator base URL.

Closes the placeholder ``lambda node_id: f"https://{node_id}/compute/aggregate"``
in ``node.py`` B7 wiring. That lambda assumed the node_id (a 32-char
hex SHA256 prefix) was a valid hostname, which it never is.

ProviderListing intentionally does not carry an endpoint URL — the
gossip layer advertises capacity/price/identity, not network address.
Resolving a node_id to an HTTP base URL therefore needs an explicit
mechanism. This module provides three composable backends:

  1. ``StaticMapEndpointResolver`` — operator-supplied
     ``{node_id: base_url}`` dict. The cleanest production path:
     deploy-time config maps known aggregator nodes to their
     ``/compute/aggregate`` endpoints.

  2. ``TransportPeerEndpointResolver`` — derives base URL from the
     ``WebSocketTransport.get_peer(node_id).address`` (host:port).
     Fallback when no static map entry exists. Note: the WS port
     and the HTTP-aggregate port are NOT necessarily the same in
     production; ``aggregate_port=`` lets operators override.

  3. ``ChainedEndpointResolver`` — tries an ordered list of resolvers
     and returns the first hit. Production wires
     ``[static_map, transport_peer]`` so operator-curated entries
     win over auto-derived ones.

Failure mode: any unresolved node_id raises
``AggregateEndpointUnresolvedError``, which
``HttpAggregateTransport`` already maps to
``AggregateTransportError`` via its endpoint-resolver try/except
(see http_aggregate_transport.py:147-156).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol


_VALID_SCHEMES = ("http", "https")


class AggregateEndpointUnresolvedError(Exception):
    """Raised when no resolver can map node_id to a base URL."""


class _PeerRegistry(Protocol):
    """Minimal contract this module needs from the WebSocketTransport."""

    def get_peer(self, node_id: str): ...


class StaticMapEndpointResolver:
    """Resolves node_id to base URL from an operator-supplied dict.

    Construction validates that every URL has an http(s) scheme so
    operator typos can't accidentally produce a plaintext or
    scheme-less URL that ``HttpAggregateTransport`` would then
    silently reject.
    """

    def __init__(self, mapping: Dict[str, str]):
        for node_id, url in mapping.items():
            if not url.startswith(_VALID_SCHEMES[0] + "://") and not url.startswith(
                _VALID_SCHEMES[1] + "://"
            ):
                raise ValueError(
                    f"endpoint URL for node_id={node_id!r} must have an "
                    f"http:// or https:// scheme, got {url!r}"
                )
        self._mapping = dict(mapping)

    def __call__(self, node_id: str) -> str:
        url = self._mapping.get(node_id)
        if url is None:
            raise AggregateEndpointUnresolvedError(
                f"no static-map entry for node_id={node_id!r}"
            )
        return url


class TransportPeerEndpointResolver:
    """Resolves node_id by consulting the transport's peer registry.

    The peer's ``address`` field is a ``host:port`` string (per
    prsm/node/transport.py:389). This resolver splits it and rebuilds
    a URL with the configured scheme + optional port override.

    ``aggregate_port=None`` reuses the WS peer port — fine for
    single-process dev nodes that serve both WS and HTTP from one
    uvicorn instance, but production deployments typically separate
    them and should pass an explicit port.
    """

    def __init__(
        self,
        transport: _PeerRegistry,
        scheme: str = "https",
        aggregate_port: Optional[int] = None,
    ):
        if scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"scheme must be http or https, got {scheme!r}"
            )
        if aggregate_port is not None and not (0 < aggregate_port < 65536):
            raise ValueError(
                f"aggregate_port must be in 1..65535, got {aggregate_port}"
            )
        self._transport = transport
        self._scheme = scheme
        self._aggregate_port = aggregate_port

    def __call__(self, node_id: str) -> str:
        peer = self._transport.get_peer(node_id)
        if peer is None:
            raise AggregateEndpointUnresolvedError(
                f"no peer in transport registry for node_id={node_id!r}"
            )
        addr = getattr(peer, "address", "")
        if ":" not in addr:
            raise AggregateEndpointUnresolvedError(
                f"peer for node_id={node_id!r} has malformed address={addr!r}"
            )
        host, _, port_str = addr.rpartition(":")
        if not host or not port_str.isdigit():
            raise AggregateEndpointUnresolvedError(
                f"peer for node_id={node_id!r} has malformed address={addr!r}"
            )
        port = self._aggregate_port if self._aggregate_port is not None else int(port_str)
        return f"{self._scheme}://{host}:{port}"


class ChainedEndpointResolver:
    """Tries an ordered list of resolvers; returns the first hit.

    Resolvers earlier in the list win — production wires the
    operator-curated ``StaticMapEndpointResolver`` first so it
    overrides the transport-derived fallback.
    """

    def __init__(self, resolvers: List):
        if not resolvers:
            raise ValueError("ChainedEndpointResolver: resolvers must not be empty")
        self._resolvers = list(resolvers)

    def __call__(self, node_id: str) -> str:
        last_err: Optional[AggregateEndpointUnresolvedError] = None
        for resolver in self._resolvers:
            try:
                return resolver(node_id)
            except AggregateEndpointUnresolvedError as exc:
                last_err = exc
                continue
        # All misses — raise the last so operators see at least one
        # specific reason in logs.
        assert last_err is not None
        raise AggregateEndpointUnresolvedError(
            f"no resolver matched node_id={node_id!r}; last reason: {last_err}"
        )
