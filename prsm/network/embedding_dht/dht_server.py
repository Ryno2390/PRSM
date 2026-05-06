"""
Embedding DHT — server.

PRSM-PROV-1 Item 3 Task 3.

The server side of the embedding DHT. Pure request/reply: receives
``FindEmbeddingRequest`` / ``FetchEmbeddingRequest`` over the wire,
returns the matching response (or an ``ErrorResponse`` on failure).

Single entry point ``handle(request_bytes) → response_bytes`` so the
PRSM RPC dispatcher (or any test harness) can plug in without caring
about the internal message taxonomy. The server NEVER initiates a
request — it only responds. Provider/Embedding/Error responses
arriving at this entry point are misuse and yield MALFORMED_REQUEST.

Trust model: the server NEVER signs anything itself. When fulfilling
``FetchEmbeddingRequest``, it relays the creator-signed
``LocalEmbeddingRecord`` it has stored. A node serving someone else's
embedding cannot forge the signature; verification happens at the
client (see ``EmbeddingDHTClient``) against the on-chain
``PublisherKeyAnchor`` for the content's creator pubkey.

Concurrency: same single-writer-per-node assumption as Phase 3.x.5
ManifestDHTServer. Multiple ``handle()`` calls in flight against a
quiescent index don't need locks.
"""

from __future__ import annotations

import logging
from typing import Any, List

from prsm.network.embedding_dht.local_index import LocalEmbeddingIndex
from prsm.network.embedding_dht.protocol import (
    EmbeddingProvidersResponse,
    EmbeddingResponse,
    ErrorCode,
    ErrorResponse,
    FetchEmbeddingRequest,
    FindEmbeddingRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    MessageType,
    ProtocolError,
    ProviderInfo,
    UnknownMessageTypeError,
    encode_message,
    parse_message,
)
from prsm.network.manifest_dht.dht_client import (
    DEFAULT_K,
    RoutingTable,
)


logger = logging.getLogger(__name__)


# Sentinel used as request_id on ErrorResponses generated when the
# server couldn't parse the request bytes (so it has no real
# request_id to correlate with).
UNKNOWN_REQUEST_ID = "<unknown>"


class EmbeddingDHTServer:
    """RPC handler for the embedding DHT.

    Constructed once per node alongside ``EmbeddingDHTClient``. Wired
    into the existing PRSM RPC dispatcher pattern: bytes in → bytes
    out.
    """

    def __init__(
        self,
        local_index: LocalEmbeddingIndex,
        routing_table: RoutingTable,
        *,
        my_node_id: str,
        my_address: str,
        k: int = DEFAULT_K,
    ) -> None:
        if not isinstance(my_node_id, str) or not my_node_id:
            raise ValueError("my_node_id must be a non-empty string")
        if not isinstance(my_address, str) or ":" not in my_address:
            raise ValueError(
                f"my_address must be 'host:port', got {my_address!r}"
            )
        self._local_index = local_index
        self._routing_table = routing_table
        self._my_node_id = my_node_id
        self._my_address = my_address
        self._k = k

    # -- entry point -------------------------------------------------------

    def handle(self, request_bytes: bytes) -> bytes:
        """Top-level dispatch: parse, route, encode response.

        Always returns bytes (an ``ErrorResponse`` on every failure
        mode); never raises. Caller is the transport layer and is
        not expected to handle exceptions from us.
        """
        try:
            message = parse_message(request_bytes)
        except IncompatibleProtocolVersionError as exc:
            return encode_message(ErrorResponse(
                request_id=UNKNOWN_REQUEST_ID,
                code=ErrorCode.UNSUPPORTED_VERSION.value,
                message=str(exc),
            ))
        except UnknownMessageTypeError as exc:
            return encode_message(ErrorResponse(
                request_id=UNKNOWN_REQUEST_ID,
                code=ErrorCode.MALFORMED_REQUEST.value,
                message=str(exc),
            ))
        except (MalformedMessageError, ProtocolError) as exc:
            return encode_message(ErrorResponse(
                request_id=UNKNOWN_REQUEST_ID,
                code=ErrorCode.MALFORMED_REQUEST.value,
                message=str(exc),
            ))

        if isinstance(message, FindEmbeddingRequest):
            return self._handle_find(message)
        if isinstance(message, FetchEmbeddingRequest):
            return self._handle_fetch(message)

        # Response shapes arriving here are misuse: the server only
        # responds, it never receives responses. Match the
        # ManifestDHTServer convention.
        request_id = getattr(message, "request_id", UNKNOWN_REQUEST_ID)
        if not request_id:
            request_id = UNKNOWN_REQUEST_ID
        return encode_message(ErrorResponse(
            request_id=request_id,
            code=ErrorCode.MALFORMED_REQUEST.value,
            message=(
                f"server received {type(message).__name__} as a request; "
                f"server only handles {MessageType.FIND_EMBEDDING.value} "
                f"and {MessageType.FETCH_EMBEDDING.value}"
            ),
        ))

    # -- handlers ---------------------------------------------------------

    def _handle_find(self, request: FindEmbeddingRequest) -> bytes:
        """Reply with up to k providers known to serve
        (content_hash, model_id).

        We always include ourselves first if we have the embedding
        locally — minimizes RTT for a likely-hit lookup. Then fill
        from the routing table's closest peers to the content_hash.

        Note: this is a "who has it" lookup, not a content-routed
        Kademlia store. The routing table returns peers near
        ``content_hash`` because that's the same keyspace partition
        we use for find_node — but we don't restrict our reply to
        peers who definitely have the embedding (we don't know
        about their local indexes). The client iterates if a
        FETCH from a candidate fails.
        """
        providers: List[ProviderInfo] = []

        if self._local_index.has(request.content_hash, request.model_id):
            providers.append(ProviderInfo(
                node_id=self._my_node_id,
                address=self._my_address,
            ))

        # Fill from routing table. find_closest_peers returns objects
        # exposing .node_id and .address — same shape ManifestDHTServer
        # depends on.
        try:
            peers = self._routing_table.find_closest_peers(
                request.content_hash, count=self._k
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "routing table lookup failed for content_hash=%s: %s",
                request.content_hash[:12], exc,
            )
            peers = []

        seen_ids = {p.node_id for p in providers}
        for peer in peers:
            node_id = getattr(peer, "node_id", None)
            address = getattr(peer, "address", None)
            if (
                not isinstance(node_id, str) or not node_id
                or not isinstance(address, str) or ":" not in address
            ):
                continue
            if node_id in seen_ids:
                continue
            try:
                providers.append(ProviderInfo(
                    node_id=node_id, address=address,
                ))
            except MalformedMessageError:
                # Skip malformed peers rather than failing the whole
                # response — same defensive posture as ManifestDHTServer.
                continue
            seen_ids.add(node_id)
            if len(providers) >= self._k + 1:
                # +1 because self is allowed to be in the list above
                # the k cap — symmetric with the manifest_dht behavior.
                break

        return encode_message(EmbeddingProvidersResponse(
            request_id=request.request_id,
            providers=tuple(providers),
        ))

    def _handle_fetch(self, request: FetchEmbeddingRequest) -> bytes:
        """Reply with the locally-stored embedding for
        (content_hash, model_id), or NOT_FOUND."""
        record = self._local_index.lookup(
            request.content_hash, request.model_id,
        )
        if record is None:
            return encode_message(ErrorResponse(
                request_id=request.request_id,
                code=ErrorCode.NOT_FOUND.value,
                message=(
                    f"no embedding for content_hash={request.content_hash[:12]} "
                    f"under model_id={request.model_id!r}"
                ),
            ))

        # Relay the creator-signed payload verbatim. Server doesn't
        # verify the signature — the client does, against the on-chain
        # anchor for the content's creator. Storing-then-relaying is
        # a transparent role for the server.
        return encode_message(EmbeddingResponse(
            request_id=request.request_id,
            content_hash=record.content_hash,
            model_id=record.model_id,
            dimension=record.dimension,
            dtype=record.dtype,
            vector_b64=record.vector_b64,
            creator_id=record.creator_id,
            created_at=record.created_at,
            signature_b64=record.signature_b64,
        ))
