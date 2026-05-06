"""DHTRequestRouter — multiplexes DHT request bytes to the right server.

A node that runs both PRSM DHTs (Phase 3.x.5 ManifestDHT for model
manifests, PRSM-PROV-1 Item 3 EmbeddingDHT for content embeddings) over
a single transport listener uses this router to dispatch to the right
server based on the JSON ``type`` field in the request envelope.

Building block for the DHT-transport-integration sprint that's
prerequisite for ManifestDHT T7 and EmbeddingDHT T3.8 going live as
real 3-node E2E tests. The TCP listener itself is out of scope for
this module — that's tracked separately. Once the listener lands,
its only DHT-related job is to call ``router.handle(request_bytes)``
and write the response back to the wire.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from prsm.network.embedding_dht.protocol import (
    MessageType as EmbeddingMessageType,
)
from prsm.network.manifest_dht.protocol import (
    MessageType as ManifestMessageType,
)

logger = logging.getLogger(__name__)


# Compute the type-set memberships once at import. Both registries
# include "error" (response-side message), but errors never arrive at
# server entry, so the overlap is safe to ignore — we resolve it in
# favor of the manifest server below to be deterministic.
_MANIFEST_TYPES = frozenset(t.value for t in ManifestMessageType)
_EMBEDDING_TYPES = frozenset(t.value for t in EmbeddingMessageType)


def _envelope_error(reason: str, code: str = "MALFORMED_REQUEST") -> bytes:
    """Generic error envelope used when no DHT-specific ErrorResponse
    can be constructed (i.e. the router can't tell which protocol to
    use). Stays JSON-shaped so callers can still decode it; carries the
    same "type"/"code"/"message" fields both DHT protocols use, so
    log-aggregation glob patterns from either side still match.
    """
    return json.dumps(
        {
            "type": "error",
            "version": 1,
            "request_id": "",
            "code": code,
            "message": reason,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class DHTRequestRouter:
    """Routes an incoming DHT request envelope to the correct server.

    Construction takes one or both of ``manifest_server`` /
    ``embedding_server``. A router with only one server still works —
    requests for the unregistered DHT receive an ``UNSUPPORTED_VERSION``
    error envelope so a bootstrap node that runs only ManifestDHT (or
    only EmbeddingDHT) doesn't crash on cross-protocol traffic.

    Both DHT servers' ``handle()`` are documented as bytes-in /
    bytes-out and NEVER raise. The router preserves that contract: any
    exception from a downstream server is caught and converted to an
    ``INTERNAL_ERROR`` envelope so the listener loop can write the
    response unconditionally.
    """

    def __init__(
        self,
        *,
        manifest_server: Optional[object] = None,
        embedding_server: Optional[object] = None,
    ) -> None:
        if manifest_server is None and embedding_server is None:
            raise ValueError(
                "DHTRequestRouter requires at least one server registered "
                "(manifest_server, embedding_server, or both)"
            )
        self._manifest_server = manifest_server
        self._embedding_server = embedding_server

    def handle(self, request_bytes: bytes) -> bytes:
        """Parse type field; dispatch to matching server. Never raises.

        Returns a generic JSON error envelope when:
          - the request isn't valid JSON or isn't a dict
          - the JSON has no string ``type`` field
          - the type doesn't match any registered server's protocol
          - the matched server's handler raises (defensive — both
            servers' handle() are documented as never-raises)
        """
        try:
            envelope = json.loads(request_bytes)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            return _envelope_error(f"JSON parse failed: {exc}")
        if not isinstance(envelope, dict):
            return _envelope_error(
                f"top-level message must be a dict, got "
                f"{type(envelope).__name__}",
            )
        msg_type = envelope.get("type")
        if not isinstance(msg_type, str):
            return _envelope_error(
                f"missing or non-string 'type' field; got {msg_type!r}",
            )

        target = self._resolve_target(msg_type)
        if target is None:
            return _envelope_error(
                f"no DHT server registered for type {msg_type!r}",
                code="UNSUPPORTED_VERSION",
            )

        try:
            response = target.handle(request_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "DHTRequestRouter: %s.handle() raised unexpectedly for "
                "type=%r — both servers' handle() are documented as "
                "never-raises; this indicates a regression",
                type(target).__name__,
                msg_type,
            )
            return _envelope_error(
                f"internal error during dispatch: {exc.__class__.__name__}",
                code="INTERNAL_ERROR",
            )

        if not isinstance(response, (bytes, bytearray)):
            logger.error(
                "DHTRequestRouter: %s.handle() returned non-bytes "
                "(%s) — this indicates a regression",
                type(target).__name__,
                type(response).__name__,
            )
            return _envelope_error(
                f"server returned non-bytes response: "
                f"{type(response).__name__}",
                code="INTERNAL_ERROR",
            )
        return bytes(response)

    def _resolve_target(self, msg_type: str):
        if msg_type in _MANIFEST_TYPES and self._manifest_server is not None:
            return self._manifest_server
        if msg_type in _EMBEDDING_TYPES and self._embedding_server is not None:
            return self._embedding_server
        return None
