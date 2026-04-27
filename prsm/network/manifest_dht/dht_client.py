"""
Manifest DHT — client.

Phase 3.x.5 Task 3.

The client side of the manifest DHT. Composes the Kademlia routing
primitives from ``prsm.compute.collaboration.p2p.node_discovery``
with the Phase 3.x.5 wire protocol (Task 1) to:

  - announce(model_id, manifest_path) — local-only state update
    (delegates to LocalManifestIndex; no network broadcast in v1)
  - find_providers(model_id) → list[ProviderInfo]
    Single-round Kademlia query to the K closest peers
  - fetch_manifest(provider, model_id) → ModelManifest
    Raw RPC fetch; NO anchor verification at this layer
  - get_manifest(model_id) → ModelManifest
    Composes find + fetch + Phase 3.x.3 anchor verification.
    Returns the FIRST provider whose bytes verify under the anchored
    publisher key. Tampered or unanchored manifests are dropped,
    next provider is tried, eventually raises ``ManifestNotFoundError``.

Anchor verification is **mandatory** at the get_manifest layer. The
client refuses to construct without an anchor — there is no
"trust the network" mode. This invariant is the central guarantee
of Phase 3.x.5: any DHT-fetched manifest you accept has been
authenticated against the on-chain publisher key.

Iterative-refinement Kademlia (asking responders for THEIR closest
peers in a second round) is a v2 feature. v1 does single-round
querying — sufficient for the small operator set per design plan §1.1.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Callable, List, Optional, Protocol

from prsm.compute.model_registry.models import ModelManifest
from prsm.network.manifest_dht.local_index import LocalManifestIndex
from prsm.network.manifest_dht.protocol import (
    DHT_PROTOCOL_VERSION,
    ErrorCode,
    ErrorResponse,
    FetchManifestRequest,
    FindProvidersRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    ManifestResponse,
    MessageType,
    ProtocolError,
    ProviderInfo,
    ProvidersResponse,
    UnknownMessageTypeError,
    encode_message,
    parse_message,
)


logger = logging.getLogger(__name__)


# Default K (Kademlia replication / closest-peer count). Matches the
# constant in prsm.compute.collaboration.p2p.node_discovery.KademliaDHT.
DEFAULT_K = 3


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class DHTClientError(Exception):
    """Base error for any DHT client failure."""


class ManifestNotFoundError(DHTClientError):
    """No verified manifest found for the requested ``model_id``.

    Raised when:
      - find_providers returned no providers, OR
      - every provider's response failed anchor verification, OR
      - every fetch attempt timed out / errored.
    """


class TransportFailureError(DHTClientError):
    """Underlying ``send_message`` callable raised, or the response
    couldn't be parsed as a DHT protocol message. Distinct from
    ``ManifestNotFoundError`` so callers can distinguish "unreachable
    peer" from "verifiably absent manifest."""


# --------------------------------------------------------------------------
# Routing table protocol
# --------------------------------------------------------------------------


class RoutingTable(Protocol):
    """Subset of ``KademliaDHT`` the DHT client depends on.

    Allows test injection of fake routing tables without depending on
    the full ``KademliaDHT`` class. Production code wires in the
    existing ``prsm.compute.collaboration.p2p.node_discovery.KademliaDHT``,
    which already implements this shape.
    """

    def find_closest_peers(self, target_id: str, count: int = 20) -> List[Any]:
        """Return up to ``count`` peers closest in XOR distance to
        ``target_id`` (a hex string in PRSM's node_id form). Each
        returned peer must expose ``.node_id: str`` and ``.address: str``
        (where address is "host:port"). The existing PeerNode class
        satisfies this."""


# --------------------------------------------------------------------------
# send_message callable signature
# --------------------------------------------------------------------------


# Synchronous request/response over the underlying P2P transport.
# (address, request_bytes) → response_bytes. Production wires this
# to TCP/SOCKS via prsm.node.transport_adapter; tests inject a fake.
# Raising indicates a transport-level failure (connection refused,
# timeout, etc.); a successful return value is the raw response
# payload that this client will parse.
SendMessageFn = Callable[[str, bytes], bytes]


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class ManifestDHTClient:
    """Cross-node fetcher for Phase 3.x.2 model manifests.

    Trust model invariant: anchor verification is REQUIRED. The
    constructor refuses to build a client without an ``anchor``, so
    callers can't accidentally use a "trust the network" mode.
    """

    def __init__(
        self,
        local_index: LocalManifestIndex,
        routing_table: RoutingTable,
        send_message: SendMessageFn,
        anchor: Any,  # PublisherKeyAnchorClient duck-typed via .lookup
        *,
        my_node_id: str,
        my_address: str,
        k: int = DEFAULT_K,
    ) -> None:
        if anchor is None:
            raise RuntimeError(
                "ManifestDHTClient requires an anchor — there is no "
                "trust-the-network mode. Anchor every DHT-fetched "
                "manifest via Phase 3.x.3 verify_manifest_with_anchor."
            )
        if not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "anchor must expose .lookup(node_id) → Optional[str]; "
                f"got {type(anchor).__name__}"
            )
        if not isinstance(my_node_id, str) or not my_node_id:
            raise ValueError("my_node_id must be a non-empty string")
        if not isinstance(my_address, str) or ":" not in my_address:
            raise ValueError(
                f"my_address must be 'host:port', got {my_address!r}"
            )

        self._local_index = local_index
        self._routing_table = routing_table
        self._send_message = send_message
        self._anchor = anchor
        self._my_node_id = my_node_id
        self._my_address = my_address
        self._k = k

    # -- announce (local-only in v1) ----------------------------------------

    def announce(self, model_id: str, manifest_path) -> None:
        """Tell the local index this node serves ``model_id``.

        v1: local-only state update. No network broadcast — peers
        discover providers via Kademlia ``find_providers`` queries.
        Kept as a method on the client (rather than callers writing
        directly to the local index) to leave a hook for v2's
        PubSub-style invalidation if it ever ships.
        """
        self._local_index.register(model_id, manifest_path)

    # -- find_providers (Kademlia single-round) -----------------------------

    def find_providers(self, model_id: str) -> List[ProviderInfo]:
        """Return the list of nodes (other than self) that may serve
        ``model_id``.

        v1: single-round query. Computes the target_id as
        ``sha256(model_id)[:32]`` (same form as PRSM node_ids), gets
        the K closest peers from the routing table, and asks each
        in parallel-of-spirit (sequentially for v1; async upgrade
        is a v2 feature). Each peer's ``ProvidersResponse.providers``
        is merged into the result list.

        Self is included if the local index has the manifest — the
        caller may then short-circuit the network fetch entirely.

        Iterative refinement (asking the responders for THEIR closest
        peers) is a v2 feature — see design plan §3.2.
        """
        target_id = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:32]
        candidate_peers = self._routing_table.find_closest_peers(
            target_id, count=self._k
        )

        providers: List[ProviderInfo] = []
        seen: set[tuple[str, str]] = set()

        # Self-as-provider: if the local index has it, include this node.
        if self._local_index.lookup(model_id) is not None:
            self_provider = ProviderInfo(
                node_id=self._my_node_id, address=self._my_address
            )
            providers.append(self_provider)
            seen.add((self_provider.node_id, self_provider.address))

        # Ask each candidate peer once. v1 = single round.
        for peer in candidate_peers:
            peer_node_id = getattr(peer, "node_id", None)
            peer_address = getattr(peer, "address", None)
            if not peer_node_id or not peer_address:
                continue
            # Skip self if it's in the routing table (most Kademlia
            # implementations include self; we deduplicate here rather
            # than depending on routing-table internals).
            if peer_node_id == self._my_node_id:
                continue

            try:
                response = self._send_request(
                    peer_address,
                    FindProvidersRequest(
                        model_id=model_id,
                        request_id=_new_request_id(),
                    ),
                )
            except TransportFailureError as exc:
                # One peer being unreachable is not fatal; continue.
                logger.debug(
                    "find_providers: peer %s at %s failed: %s",
                    peer_node_id, peer_address, exc,
                )
                continue

            if not isinstance(response, ProvidersResponse):
                logger.debug(
                    "find_providers: peer %s at %s returned %s instead of "
                    "ProvidersResponse",
                    peer_node_id, peer_address, type(response).__name__,
                )
                continue

            for p in response.providers:
                key = (p.node_id, p.address)
                if key in seen or p.node_id == self._my_node_id:
                    continue
                seen.add(key)
                providers.append(p)

        return providers

    # -- fetch_manifest (raw RPC; no verify) --------------------------------

    def fetch_manifest(
        self, provider: ProviderInfo, model_id: str
    ) -> ModelManifest:
        """Fetch the manifest for ``model_id`` from one provider.

        Does NOT verify against the anchor — that's
        ``get_manifest``'s job. Callers using ``fetch_manifest``
        directly are responsible for verification.

        Raises:
            ManifestNotFoundError: provider responded with NOT_FOUND,
                or returned a malformed manifest dict.
            TransportFailureError: send_message raised, or response
                couldn't be parsed as a DHT protocol message.
        """
        request = FetchManifestRequest(
            model_id=model_id, request_id=_new_request_id()
        )
        response = self._send_request(provider.address, request)

        if isinstance(response, ErrorResponse):
            if response.code == ErrorCode.NOT_FOUND.value:
                raise ManifestNotFoundError(
                    f"provider {provider.node_id} at {provider.address} "
                    f"reports {model_id!r} not found"
                )
            raise TransportFailureError(
                f"provider {provider.node_id} returned error "
                f"{response.code}: {response.message}"
            )

        if not isinstance(response, ManifestResponse):
            raise TransportFailureError(
                f"provider {provider.node_id} returned "
                f"{type(response).__name__} instead of ManifestResponse"
            )

        try:
            return ModelManifest.from_dict(response.manifest)
        except (TypeError, ValueError, KeyError) as exc:
            raise ManifestNotFoundError(
                f"provider {provider.node_id} returned malformed manifest: {exc}"
            ) from exc

    # -- get_manifest (compose + verify) ------------------------------------

    def get_manifest(self, model_id: str) -> ModelManifest:
        """Find + fetch + anchor-verify.

        Returns the first provider's manifest whose signature verifies
        under the anchored publisher key. Tampered, unanchored, or
        otherwise-failing manifests are dropped; the next provider is
        tried. Exhausting all providers without a verified manifest
        raises ``ManifestNotFoundError``.

        Anchor verification is the trust anchor here — any node can
        serve any bytes, but only bytes signed under the on-chain
        publisher key are accepted.
        """
        providers = self.find_providers(model_id)
        if not providers:
            raise ManifestNotFoundError(
                f"no providers found for model_id={model_id!r}"
            )

        # Late import to avoid a circular dep with the publisher_key_anchor
        # package (its verifiers.py imports model_registry; this module is
        # below model_registry). Cheap; runs once per get_manifest.
        from prsm.security.publisher_key_anchor.exceptions import (
            AnchorRPCError,
        )
        from prsm.security.publisher_key_anchor.verifiers import (
            verify_manifest_with_anchor,
        )

        last_error: Optional[Exception] = None
        for provider in providers:
            # Self-fetch shortcut: if this node is itself a provider,
            # read straight from the local index rather than RPC-to-self.
            try:
                if provider.node_id == self._my_node_id:
                    manifest = self._fetch_local(model_id)
                else:
                    manifest = self.fetch_manifest(provider, model_id)
            except ManifestNotFoundError as exc:
                last_error = exc
                continue
            except TransportFailureError as exc:
                last_error = exc
                logger.debug(
                    "get_manifest: transport error from provider %s: %s",
                    provider.node_id, exc,
                )
                continue

            # Substitution defense (HIGH-2 from Phase 3.x.5 round 1
            # review): a malicious provider can return a *validly-signed*
            # manifest under a DIFFERENT model_id than what we asked for.
            # Anchor verification passes (the signature is genuine), but
            # the bytes do not describe the model the caller wanted.
            # Reject the substitution and try the next provider.
            if manifest.model_id != model_id:
                logger.warning(
                    "get_manifest: provider %s returned manifest with "
                    "model_id=%r, expected %r — rejecting substitution "
                    "attempt",
                    provider.node_id, manifest.model_id, model_id,
                )
                last_error = ManifestNotFoundError(
                    f"provider {provider.node_id} returned mismatched "
                    f"model_id (got {manifest.model_id!r}, "
                    f"expected {model_id!r})"
                )
                continue

            try:
                verified = verify_manifest_with_anchor(
                    manifest, self._anchor
                )
            except AnchorRPCError as exc:
                # HIGH-3: anchor RPC failure during one provider's
                # verify must not abort the whole get_manifest. Treat
                # as transient; continue to the next provider. If every
                # provider hits the same anchor RPC failure, we surface
                # ManifestNotFoundError below — which is the correct
                # caller-facing error (the model could not be obtained).
                logger.warning(
                    "get_manifest: anchor RPC error verifying provider "
                    "%s: %s — treating as transient, trying next provider",
                    provider.node_id, exc,
                )
                last_error = exc
                continue

            if not verified:
                logger.debug(
                    "get_manifest: provider %s returned manifest that failed "
                    "anchor verification — trying next provider",
                    provider.node_id,
                )
                last_error = ManifestNotFoundError(
                    f"provider {provider.node_id} bytes failed anchor verify"
                )
                continue

            return manifest

        raise ManifestNotFoundError(
            f"no provider returned a verified manifest for {model_id!r}; "
            f"last error: {last_error}"
        )

    # -- internals ----------------------------------------------------------

    def _fetch_local(self, model_id: str) -> ModelManifest:
        """Read the manifest directly from the local index without
        going through the network."""
        path = self._local_index.lookup(model_id)
        if path is None:
            raise ManifestNotFoundError(
                f"local index claims to serve {model_id!r} but lookup "
                f"returned None — index drift"
            )
        try:
            import json
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestNotFoundError(
                f"local manifest at {path} unreadable: {exc}"
            ) from exc
        try:
            return ModelManifest.from_dict(data)
        except (TypeError, ValueError, KeyError) as exc:
            raise ManifestNotFoundError(
                f"local manifest at {path} malformed: {exc}"
            ) from exc

    def _send_request(self, address: str, request: Any) -> Any:
        """Encode + send + parse one RPC. Surfaces transport errors as
        ``TransportFailureError`` and protocol-version mismatches as
        ``TransportFailureError`` (peer-skip semantics, not a hard
        failure of the calling operation)."""
        encoded = encode_message(request)
        try:
            raw_response = self._send_message(address, encoded)
        except Exception as exc:
            raise TransportFailureError(
                f"send_message to {address} raised: {exc}"
            ) from exc

        try:
            response = parse_message(raw_response)
        except (
            MalformedMessageError,
            UnknownMessageTypeError,
            IncompatibleProtocolVersionError,
        ) as exc:
            raise TransportFailureError(
                f"response from {address} unparseable: {exc}"
            ) from exc
        except ProtocolError as exc:
            raise TransportFailureError(
                f"response from {address} protocol error: {exc}"
            ) from exc

        # Correlation: response.request_id must match request.request_id.
        # Detects stale / out-of-order responses; treat as transport
        # failure since we can't trust the bytes anyway.
        response_request_id = getattr(response, "request_id", None)
        request_request_id = getattr(request, "request_id", None)
        if response_request_id != request_request_id:
            raise TransportFailureError(
                f"response from {address} has request_id="
                f"{response_request_id!r}; expected {request_request_id!r}"
            )

        return response


def _new_request_id() -> str:
    """Per-RPC unique correlator. UUID4 is overkill in entropy but the
    cheap way to guarantee uniqueness without per-client state."""
    return uuid.uuid4().hex
