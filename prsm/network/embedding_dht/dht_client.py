"""
Embedding DHT — client.

PRSM-PROV-1 Item 3 Task 4.

Client side of the embedding DHT. Two operations:

  - ``find_providers(content_hash, model_id) -> List[ProviderInfo]``:
    Returns up to k peers known to serve (content_hash, model_id).
    Pure routing — no signature verification needed at this step.

  - ``fetch_embedding(provider, content_hash, model_id) -> bytes``:
    Fetches the vector from a specific peer. Verifies the
    creator-signed payload against the on-chain ``PublisherKeyAnchor``
    BEFORE returning the bytes. Rejects unsigned-or-wrong-signed
    responses, which is the design's poisoning defense.

Trust model invariant: ``EmbeddingDHTClient`` REFUSES to construct
without a verifier. Same posture as ``ManifestDHTClient`` — there is
no "trust the network" mode, because cross-node embedding poisoning
is the entire attack class this DHT exists to defend against.

The verifier is duck-typed via two callables:

  - ``creator_pubkey_for(content_hash) -> bytes | None``:
    Returns the canonical creator pubkey (32 bytes for Ed25519) for
    the content's on-chain registration. Production wires this to the
    ``PublisherKeyAnchor`` lookup; tests inject a stub.

  - ``verify_signature(pubkey_bytes, message_bytes, signature_bytes) -> bool``:
    Verifies an Ed25519 signature. Production wires this to the
    project's ``cryptography`` library wrapper.

Splitting these out keeps the client fully testable without dragging
the on-chain anchor or cryptography into the unit-test surface.
"""

from __future__ import annotations

import base64
import binascii
import logging
import secrets
from typing import Any, Callable, List, Optional

from prsm.network.embedding_dht.protocol import (
    EmbeddingProvidersResponse,
    EmbeddingResponse,
    ErrorCode,
    ErrorResponse,
    FetchEmbeddingRequest,
    FindEmbeddingRequest,
    IncompatibleProtocolVersionError,
    MalformedMessageError,
    ProviderInfo,
    UnknownMessageTypeError,
    canonical_signing_payload,
    encode_message,
    parse_message,
)
from prsm.network.manifest_dht.dht_client import (
    DEFAULT_K,
    RoutingTable,
    SendMessageFn,
)


logger = logging.getLogger(__name__)


# Verifier callables — see module docstring.
CreatorPubkeyLookupFn = Callable[[str], Optional[bytes]]
VerifySignatureFn = Callable[[bytes, bytes, bytes], bool]


class EmbeddingDHTClientError(Exception):
    """Base for embedding DHT client failures."""


class EmbeddingNotFoundError(EmbeddingDHTClientError):
    """Server replied NOT_FOUND or no provider had the embedding."""


class TransportFailureError(EmbeddingDHTClientError):
    """Underlying transport failed (timeout, connection refused, etc.)."""


class SignatureVerificationError(EmbeddingDHTClientError):
    """Returned ``EmbeddingResponse`` failed creator-signature
    verification. The server is either compromised, malicious, or
    serving stale data signed under a rotated key. In all three
    cases the right response is to reject and retry against another
    provider."""


class EmbeddingDHTClient:
    """Cross-node fetcher for content embeddings.

    Trust model invariant: verifier is REQUIRED.
    """

    def __init__(
        self,
        routing_table: RoutingTable,
        send_message: SendMessageFn,
        creator_pubkey_for: CreatorPubkeyLookupFn,
        verify_signature: VerifySignatureFn,
        *,
        my_node_id: str,
        my_address: str,
        k: int = DEFAULT_K,
    ) -> None:
        if creator_pubkey_for is None or verify_signature is None:
            raise RuntimeError(
                "EmbeddingDHTClient requires a verifier — there is no "
                "trust-the-network mode. Wire creator_pubkey_for to "
                "PublisherKeyAnchor and verify_signature to "
                "Ed25519 verify."
            )
        if not isinstance(my_node_id, str) or not my_node_id:
            raise ValueError("my_node_id must be a non-empty string")
        if not isinstance(my_address, str) or ":" not in my_address:
            raise ValueError(
                f"my_address must be 'host:port', got {my_address!r}"
            )
        self._routing_table = routing_table
        self._send_message = send_message
        self._creator_pubkey_for = creator_pubkey_for
        self._verify_signature = verify_signature
        self._my_node_id = my_node_id
        self._my_address = my_address
        self._k = k

    # -- public API -------------------------------------------------------

    def find_providers(
        self, content_hash: str, model_id: str,
    ) -> List[ProviderInfo]:
        """Ask up to k peers near ``content_hash`` for who has the
        embedding under ``model_id``. Returns the union of all returned
        provider lists, deduplicated by node_id."""
        try:
            peers = self._routing_table.find_closest_peers(
                content_hash, count=self._k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "routing-table find_closest_peers failed: %s", exc,
            )
            return []

        request = FindEmbeddingRequest(
            content_hash=content_hash,
            model_id=model_id,
            request_id=_new_request_id(),
        )
        wire = encode_message(request)

        seen: dict[str, ProviderInfo] = {}
        for peer in peers:
            address = getattr(peer, "address", None)
            if not isinstance(address, str) or ":" not in address:
                continue
            try:
                resp_bytes = self._send_message(address, wire)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "find_providers send failed to %s: %s",
                    address, exc,
                )
                continue
            try:
                resp = parse_message(resp_bytes)
            except (
                MalformedMessageError,
                UnknownMessageTypeError,
                IncompatibleProtocolVersionError,
            ) as exc:
                logger.debug(
                    "find_providers parse failed from %s: %s",
                    address, exc,
                )
                continue
            if isinstance(resp, ErrorResponse):
                continue
            if not isinstance(resp, EmbeddingProvidersResponse):
                continue
            for p in resp.providers:
                if p.node_id and p.node_id not in seen:
                    seen[p.node_id] = p

        return list(seen.values())

    def fetch_embedding(
        self,
        provider: ProviderInfo,
        content_hash: str,
        model_id: str,
    ) -> EmbeddingResponse:
        """Fetch the embedding from a specific provider and verify
        the creator signature.

        Raises ``EmbeddingNotFoundError`` if the provider doesn't have
        the embedding, ``TransportFailureError`` on send/parse failure,
        ``SignatureVerificationError`` if the response fails signature
        verification against the canonical creator pubkey.
        """
        request = FetchEmbeddingRequest(
            content_hash=content_hash,
            model_id=model_id,
            request_id=_new_request_id(),
        )
        try:
            resp_bytes = self._send_message(
                provider.address, encode_message(request),
            )
        except Exception as exc:  # noqa: BLE001
            raise TransportFailureError(
                f"send to {provider.address} failed: {exc}"
            ) from exc
        try:
            resp = parse_message(resp_bytes)
        except (
            MalformedMessageError,
            UnknownMessageTypeError,
            IncompatibleProtocolVersionError,
        ) as exc:
            raise TransportFailureError(
                f"parse failed from {provider.address}: {exc}"
            ) from exc

        if isinstance(resp, ErrorResponse):
            if resp.code == ErrorCode.NOT_FOUND.value:
                raise EmbeddingNotFoundError(
                    f"{provider.node_id} has no embedding for "
                    f"({content_hash[:12]}, {model_id})"
                )
            raise TransportFailureError(
                f"{provider.node_id} returned error code "
                f"{resp.code}: {resp.message}"
            )
        if not isinstance(resp, EmbeddingResponse):
            raise TransportFailureError(
                f"{provider.node_id} returned unexpected message "
                f"type {type(resp).__name__}"
            )

        # Cross-check the response actually answers OUR request.
        # A malicious peer could otherwise serve a (content_hash,
        # model_id) we didn't ask for, hoping the caller updates the
        # local index under the wrong key.
        if resp.content_hash != content_hash:
            raise SignatureVerificationError(
                f"response content_hash={resp.content_hash!r} != "
                f"requested {content_hash!r}"
            )
        if resp.model_id != model_id:
            raise SignatureVerificationError(
                f"response model_id={resp.model_id!r} != "
                f"requested {model_id!r}"
            )

        self._verify_creator_signature(resp)
        return resp

    # -- internal --------------------------------------------------------

    def _verify_creator_signature(self, resp: EmbeddingResponse) -> None:
        """Verify ``resp.signature_b64`` against the creator's
        on-chain anchored Ed25519 pubkey. Raises
        ``SignatureVerificationError`` on any failure."""
        pubkey = self._creator_pubkey_for(resp.content_hash)
        if pubkey is None:
            raise SignatureVerificationError(
                f"no on-chain creator pubkey for content_hash="
                f"{resp.content_hash[:12]}; refusing to trust "
                f"unanchored embedding"
            )
        if not isinstance(pubkey, (bytes, bytearray)):
            raise SignatureVerificationError(
                f"creator_pubkey_for returned non-bytes "
                f"({type(pubkey).__name__})"
            )

        try:
            sig_bytes = base64.b64decode(resp.signature_b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SignatureVerificationError(
                f"signature_b64 not valid base64: {exc}"
            ) from exc

        try:
            vector_bytes = base64.b64decode(resp.vector_b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SignatureVerificationError(
                f"vector_b64 not valid base64: {exc}"
            ) from exc
        if len(vector_bytes) != resp.dimension * 4:
            raise SignatureVerificationError(
                f"vector_b64 decodes to {len(vector_bytes)} bytes; "
                f"expected dimension*4={resp.dimension * 4}"
            )

        message = canonical_signing_payload(
            content_hash=resp.content_hash,
            model_id=resp.model_id,
            dimension=resp.dimension,
            dtype=resp.dtype,
            vector_bytes=vector_bytes,
            created_at=resp.created_at,
        )

        try:
            ok = self._verify_signature(bytes(pubkey), message, sig_bytes)
        except Exception as exc:  # noqa: BLE001
            raise SignatureVerificationError(
                f"verify_signature raised: {exc}"
            ) from exc

        if not ok:
            raise SignatureVerificationError(
                f"creator-signature verification FAILED for "
                f"content_hash={resp.content_hash[:12]} under "
                f"model_id={resp.model_id!r}"
            )


# -- helpers --------------------------------------------------------------


def _new_request_id() -> str:
    """16-byte hex request id, same shape as Phase 3.x.5."""
    return secrets.token_hex(16)
