"""
Manifest DHT — server.

Phase 3.x.5 Task 4.

The server side of the manifest DHT. Pure request/reply: receives
``FindProvidersRequest`` / ``FetchManifestRequest`` over the wire,
returns the matching response (or an ``ErrorResponse`` on failure).

Single entry point ``handle(request_bytes) → response_bytes`` so the
PRSM RPC dispatcher (or any test harness) can plug in without
caring about the internal message taxonomy. The server NEVER
initiates a request — it only responds. ``ProvidersResponse`` /
``ManifestResponse`` / ``ErrorResponse`` arriving at this entry point
are misuse and yield ``MALFORMED_REQUEST``.

Concurrency: the server itself holds no mutable state. It reads
through ``LocalManifestIndex.lookup`` and ``RoutingTable.find_closest_peers``,
both of which are concurrent-read safe under the project-wide
single-writer-per-node invariant (Phase 3.x.2 / 3.x.4 / 3.x.5
share this). The registry's ``_fetch_manifest_via_dht`` is a writer
to the same ``LocalManifestIndex`` — concurrent server reads against
that writer rely on the single-writer assumption holding at the
operator level (no parallel registries or DHT-fetch loops sharing one
index). Multiple ``handle()`` calls in flight against a quiescent
index don't need locks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from prsm.network.manifest_dht.dht_client import (
    DEFAULT_K,
    RoutingTable,
)
from prsm.network.manifest_dht.local_index import LocalManifestIndex
from prsm.network.manifest_dht.protocol import (
    ErrorCode,
    ErrorResponse,
    FetchManifestRequest,
    FindProvidersRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    ManifestResponse,
    ProtocolError,
    ProviderInfo,
    ProvidersResponse,
    UnknownMessageTypeError,
    encode_message,
    parse_message,
)


logger = logging.getLogger(__name__)


# Sentinel used as ``request_id`` on ErrorResponses generated when the
# server couldn't parse the request bytes (so it has no real request_id
# to correlate with). Non-empty so the protocol's non-empty-string
# invariant on request_id holds; client-side correlation will see this
# as a mismatch and surface it as TransportFailureError, which is the
# correct semantics — the caller hit unparseable bytes from the peer.
UNKNOWN_REQUEST_ID = "<unknown>"


class ManifestDHTServer:
    """RPC handler for the manifest DHT.

    Constructed once per node (alongside the corresponding
    ``ManifestDHTClient``). Wired into the existing PRSM RPC
    dispatcher pattern: bytes in → bytes out.
    """

    def __init__(
        self,
        local_index: LocalManifestIndex,
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
        """Parse + dispatch + encode. Returns response bytes.

        NEVER raises — every failure path is mapped to an encoded
        ``ErrorResponse``. The transport-layer caller can write the
        bytes back to the wire without exception handling.
        """
        try:
            request = parse_message(request_bytes)
        except IncompatibleProtocolVersionError as exc:
            return self._error(
                UNKNOWN_REQUEST_ID, ErrorCode.UNSUPPORTED_VERSION, str(exc)
            )
        except (MalformedMessageError, UnknownMessageTypeError) as exc:
            return self._error(
                UNKNOWN_REQUEST_ID, ErrorCode.MALFORMED_REQUEST, str(exc)
            )
        except ProtocolError as exc:
            # Catch-all for any other protocol-layer failure
            return self._error(
                UNKNOWN_REQUEST_ID, ErrorCode.MALFORMED_REQUEST, str(exc)
            )
        except Exception as exc:  # noqa: BLE001
            # Truly unexpected — log + return INTERNAL_ERROR with a
            # generic message (don't leak exception details to peers).
            logger.exception(
                "ManifestDHTServer.handle: unexpected parse failure"
            )
            return self._error(
                UNKNOWN_REQUEST_ID,
                ErrorCode.INTERNAL_ERROR,
                "internal error during parse",
            )

        # HIGH-1 (Phase 3.x.5 round 1 review): the dispatch must honor
        # the never-raises invariant from the docstring. The handler
        # methods catch their declared failure paths, but we still wrap
        # the dispatch in an outer try/except so a future regression
        # (a new uncaught exception in either handler, or in
        # `encode_message` itself when the on-disk manifest contains a
        # non-serializable value) cannot escape `handle()`.
        request_id = getattr(request, "request_id", "") or UNKNOWN_REQUEST_ID
        try:
            if isinstance(request, FindProvidersRequest):
                return self._handle_find_providers(request)
            if isinstance(request, FetchManifestRequest):
                return self._handle_fetch_manifest(request)

            # Response-type messages (ProvidersResponse, ManifestResponse,
            # ErrorResponse) shouldn't arrive at the server entry point.
            # Treat as malformed-misuse.
            return self._error(
                request_id,
                ErrorCode.MALFORMED_REQUEST,
                f"server received non-request message type: "
                f"{type(request).__name__}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ManifestDHTServer.handle: unexpected dispatch failure "
                "for request_id=%r",
                request_id,
            )
            return self._error(
                request_id,
                ErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: "
                f"{exc.__class__.__name__}",
            )

    # -- handlers ----------------------------------------------------------

    def _handle_find_providers(
        self, request: FindProvidersRequest
    ) -> bytes:
        """Build a ProvidersResponse:
        - Self if the local index has the model
        - K closest peers from the routing table (excluding self)
        """
        providers: list[ProviderInfo] = []
        seen: set[tuple[str, str]] = set()

        # Self-as-provider check
        if self._local_index.lookup(request.model_id) is not None:
            self_provider = ProviderInfo(
                node_id=self._my_node_id, address=self._my_address
            )
            providers.append(self_provider)
            seen.add((self_provider.node_id, self_provider.address))

        # Add closest peers
        target_id = hashlib.sha256(
            request.model_id.encode("utf-8")
        ).hexdigest()[:32]
        try:
            candidates = self._routing_table.find_closest_peers(
                target_id, count=self._k
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "find_providers: routing-table lookup raised"
            )
            return self._error(
                request.request_id,
                ErrorCode.INTERNAL_ERROR,
                f"routing-table lookup failed: {exc}",
            )

        for peer in candidates:
            peer_node_id = getattr(peer, "node_id", None)
            peer_address = getattr(peer, "address", None)
            if not peer_node_id or not peer_address:
                continue
            if peer_node_id == self._my_node_id:
                continue
            if not isinstance(peer_node_id, str) or not isinstance(
                peer_address, str
            ):
                continue
            if ":" not in peer_address:
                # Routing table returned a malformed peer; skip rather
                # than fail the whole lookup.
                continue
            key = (peer_node_id, peer_address)
            if key in seen:
                continue
            seen.add(key)
            providers.append(
                ProviderInfo(node_id=peer_node_id, address=peer_address)
            )

        response = ProvidersResponse(
            request_id=request.request_id,
            providers=tuple(providers),
        )
        return encode_message(response)

    def _handle_fetch_manifest(
        self, request: FetchManifestRequest
    ) -> bytes:
        """Look up model_id in the local index; if present, read the
        manifest.json and serve it. Otherwise return NOT_FOUND."""
        manifest_path = self._local_index.lookup(request.model_id)
        if manifest_path is None:
            return self._error(
                request.request_id,
                ErrorCode.NOT_FOUND,
                f"model_id {request.model_id!r} not in local index",
            )

        try:
            data = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            # UnicodeDecodeError catches the case where the on-disk
            # manifest is not valid UTF-8 (binary content, partial
            # write, or operator dropped a non-text file at the path).
            logger.warning(
                "fetch_manifest: local manifest at %s unreadable: %s",
                manifest_path, exc,
            )
            return self._error(
                request.request_id,
                ErrorCode.INTERNAL_ERROR,
                f"local manifest unreadable: {exc.__class__.__name__}",
            )

        if not isinstance(data, dict):
            logger.warning(
                "fetch_manifest: local manifest at %s parsed to %s, "
                "expected dict",
                manifest_path, type(data).__name__,
            )
            return self._error(
                request.request_id,
                ErrorCode.INTERNAL_ERROR,
                "local manifest is not a JSON dict",
            )

        response = ManifestResponse(
            request_id=request.request_id, manifest=data
        )
        return encode_message(response)

    # -- internals ---------------------------------------------------------

    def _error(
        self, request_id: str, code: ErrorCode, message: str
    ) -> bytes:
        """Build + encode an ErrorResponse. Used for every non-happy
        path so the transport caller never has to handle exceptions."""
        return encode_message(
            ErrorResponse(
                request_id=request_id,
                code=code.value,
                message=message,
            )
        )
