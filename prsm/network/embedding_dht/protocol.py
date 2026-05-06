"""
Embedding DHT — wire-format protocol.

PRSM-PROV-1 Item 3 Task 1.

Defines the JSON-over-transport messages exchanged between
``EmbeddingDHTClient`` and ``EmbeddingDHTServer`` for cross-node
content-embedding gossip:

  Requests (client → server):
    - FindEmbeddingRequest  — "who has the embedding for (content_hash, model_id)?"
    - FetchEmbeddingRequest — "send me the embedding for this (content_hash, model_id)"

  Responses (server → client):
    - EmbeddingProvidersResponse — list of (node_id, address) pairs
    - EmbeddingResponse          — vector bytes + model metadata + creator signature
    - ErrorResponse              — coded failure (NOT_FOUND, MALFORMED, etc.)

Wire format is JSON, dispatched by ``type`` field. Each message
carries a ``protocol_version`` so a peer running an incompatible
version is rejected at parse time rather than producing confusing
shape errors deeper in the stack.

Critical design constraint — model_id keyspace partitioning:

Embedding vectors from different models live in different vector
spaces; their cosine similarities are meaningless. The DHT keys on
the (content_hash, model_id) tuple, not content_hash alone. A node
running ``sentence-transformers/all-MiniLM-L6-v2`` cannot use a vector
produced by ``openai/text-embedding-ada-002`` for dedup — it gets
back NOT_FOUND under its own model_id and falls through to the
upload-as-new path. This is intentional. Cross-model alignment
(Procrustes / linear projection) is research-grade and explicitly
out of scope for PRSM-PROV-1 (deferred to R10).

Poisoning defense — payload signature:

Unlike ManifestDHT (where the manifest itself carries a Phase 3.x.2
signature verified via the Phase 3.x.3 anchor), an embedding vector
has no "natural" signing structure. We add one explicitly:
``EmbeddingResponse.signature_b64`` is an Ed25519 signature by the
*original creator* (not the serving node) over the canonical bytes:

  domain_tag || content_hash_bytes || model_id_utf8 || dim_be ||
  dtype_utf8 || vector_bytes || created_at_be

A node that serves someone else's embedding cannot forge the
signature — the creator's pubkey is anchored on-chain via
``PublisherKeyAnchor`` (Phase 3.x.3). Verifiers reject responses
whose signature does not match the on-chain creator pubkey for the
content_hash. This closes the "malicious peer serves random vector"
attack surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple


# Wire-format version. Bump when any message dataclass adds, removes,
# or reshapes a field. Peers running mismatched versions reject each
# other at parse time — see ``parse_message``.
EMBEDDING_DHT_PROTOCOL_VERSION = 1

# Hard cap on incoming wire-message size. Largest expected payload is
# EmbeddingResponse with a 8192-dim float32 vector = 32 KB raw +
# base64 overhead (~33%) ≈ 42 KB. Headers, signatures, model_id push
# total to ~64 KB worst case. 256 KB gives 4× headroom and matches
# ManifestDHT for operational consistency. Anything larger is rejected
# at parse time before json.loads allocates — closes the OOM-via-huge-
# payload DoS surface.
MAX_MESSAGE_BYTES = 256 * 1024

# Per-list cap on EmbeddingProvidersResponse.providers. Bounds the
# response-side amplification — a hostile peer cannot return a
# million-entry list to exhaust caller memory parsing it.
MAX_PROVIDERS_PER_RESPONSE = 1024

# Per-vector hard cap on dimensions. OpenAI ada-002 is 1536; MiniLM
# is 384; the largest published model dimension we'd reasonably
# accept is BGE-large at 1024. 8192 is a generous cap that leaves
# room for any model an operator might register without letting a
# hostile peer ship a 10M-dim vector to exhaust memory at decode time.
MAX_VECTOR_DIMENSION = 8192

# Domain tag prepended to the canonical signing payload. Domain
# separation prevents a creator's Ed25519 key from being tricked into
# signing something interpretable as a different protocol message
# type (e.g. an InferenceReceipt). Length-prefix-free format is fine
# here because the signing payload uses fixed-width big-endian ints
# for variable-length fields.
SIGNING_DOMAIN_TAG = b"PRSM-PROV-1/EmbeddingResponse/v1"

# Allowed embedding dtype strings. Restrict to float32 for now —
# float16 / bfloat16 add cross-platform-determinism risk that isn't
# justified by the bandwidth savings at the dimensions we care about.
# Forward-compatible: peers running a future version with float16 in
# the registry would advertise themselves with a higher
# protocol_version, triggering the version-mismatch reject path.
ALLOWED_DTYPES = frozenset({"float32"})

# Tag strings used as the ``type`` field on every wire message.
# Public so callers can build ad-hoc messages without importing the
# concrete dataclass (handy for fuzz / chaos testing).
class MessageType(str, Enum):
    FIND_EMBEDDING = "find_embedding"
    FETCH_EMBEDDING = "fetch_embedding"
    EMBEDDING_PROVIDERS = "embedding_providers"
    EMBEDDING_RESPONSE = "embedding_response"
    ERROR = "error"


# Coded error reasons returned by the server. Strings (not ints) so
# log lines + tcpdump traces are readable.
class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ProtocolError(Exception):
    """Base exception for any DHT protocol-level failure."""


class UnknownMessageTypeError(ProtocolError):
    """Received a message whose ``type`` field is not a known
    ``MessageType``. Treat as malformed / hostile peer."""


class MalformedMessageError(ProtocolError):
    """A required field was missing, the wrong type, or the wrapper
    JSON couldn't even be parsed. Caller should respond with an
    ``ErrorResponse(MALFORMED_REQUEST)`` and probably drop the peer."""


class IncompatibleProtocolVersionError(ProtocolError):
    """Peer's ``protocol_version`` doesn't match the local
    ``EMBEDDING_DHT_PROTOCOL_VERSION``. Caller should reject the
    message (and ideally skip the peer for a configurable cool-off)."""


# --------------------------------------------------------------------------
# Message dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FindEmbeddingRequest:
    """Client → server: "Who can serve the embedding for
    (content_hash, model_id)?"

    The server should return its locally-known providers (typically
    the server itself if it has the embedding, plus any peers it
    knows from prior find_embedding iterations) up to a bounded
    count K. Iteration / convergence is the client's responsibility
    (Kademlia-style).
    """

    content_hash: str   # 0x-prefixed hex; matches ProvenanceRegistry contentHash
    model_id: str       # e.g. "openai/text-embedding-ada-002"
    request_id: str
    protocol_version: int = EMBEDDING_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.FIND_EMBEDDING.value

    def __post_init__(self) -> None:
        _validate_str_field("content_hash", self.content_hash)
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "content_hash": self.content_hash,
            "model_id": self.model_id,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FindEmbeddingRequest":
        _expect_type(data, MessageType.FIND_EMBEDDING)
        return cls(
            content_hash=_required_str(data, "content_hash"),
            model_id=_required_str(data, "model_id"),
            request_id=_required_str(data, "request_id"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class FetchEmbeddingRequest:
    """Client → server: "Send me the embedding for
    (content_hash, model_id)."

    Server returns ``EmbeddingResponse`` if it has the embedding
    locally, ``ErrorResponse(NOT_FOUND)`` otherwise. Server does
    NOT chase the DHT on behalf of the client — find_embedding is
    the iteration primitive.
    """

    content_hash: str
    model_id: str
    request_id: str
    protocol_version: int = EMBEDDING_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.FETCH_EMBEDDING.value

    def __post_init__(self) -> None:
        _validate_str_field("content_hash", self.content_hash)
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "content_hash": self.content_hash,
            "model_id": self.model_id,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FetchEmbeddingRequest":
        _expect_type(data, MessageType.FETCH_EMBEDDING)
        return cls(
            content_hash=_required_str(data, "content_hash"),
            model_id=_required_str(data, "model_id"),
            request_id=_required_str(data, "request_id"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class ProviderInfo:
    """One provider's identity + address; sub-record of
    EmbeddingProvidersResponse."""

    node_id: str
    address: str  # "host:port"

    def __post_init__(self) -> None:
        _validate_str_field("node_id", self.node_id)
        _validate_str_field("address", self.address)
        if ":" not in self.address:
            raise MalformedMessageError(
                f"address must be 'host:port', got {self.address!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "address": self.address}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderInfo":
        return cls(
            node_id=_required_str(data, "node_id"),
            address=_required_str(data, "address"),
        )


@dataclass(frozen=True)
class EmbeddingProvidersResponse:
    """Server → client: list of providers for the requested
    (content_hash, model_id)."""

    request_id: str
    providers: Tuple[ProviderInfo, ...]
    protocol_version: int = EMBEDDING_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.EMBEDDING_PROVIDERS.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)
        if not isinstance(self.providers, tuple):
            object.__setattr__(self, "providers", tuple(self.providers))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "providers": [p.to_dict() for p in self.providers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingProvidersResponse":
        _expect_type(data, MessageType.EMBEDDING_PROVIDERS)
        providers_raw = data.get("providers")
        if not isinstance(providers_raw, list):
            raise MalformedMessageError(
                f"providers must be a list, got "
                f"{type(providers_raw).__name__}"
            )
        if len(providers_raw) > MAX_PROVIDERS_PER_RESPONSE:
            raise MalformedMessageError(
                f"providers list exceeds MAX_PROVIDERS_PER_RESPONSE "
                f"({len(providers_raw)} > {MAX_PROVIDERS_PER_RESPONSE})"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            providers=tuple(
                ProviderInfo.from_dict(p) for p in providers_raw
            ),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class EmbeddingResponse:
    """Server → client: a single embedding vector with metadata and
    creator signature.

    The vector is transmitted as base64-encoded raw float32 bytes
    (little-endian, since numpy's default on every platform PRSM
    targets is little-endian; numpy's ``frombuffer`` interprets bytes
    according to the platform native order, but we serialize via
    ``.tobytes()`` which is platform-native — see
    ``vector_bytes_canonical`` for the canonical serialization
    function used at sign + verify time).

    The signature binds (content_hash, model_id, dim, dtype,
    vector_bytes, created_at) under the creator's Ed25519 key. The
    creator's pubkey is anchored on-chain via
    ``PublisherKeyAnchor`` (Phase 3.x.3); a verifying node looks up
    the canonical creator pubkey for ``content_hash`` and rejects
    the response if the signature does not verify.
    """

    request_id: str
    content_hash: str   # 0x-prefixed hex
    model_id: str
    dimension: int      # vector length, e.g. 1536 or 384
    dtype: str          # "float32" (only allowed value in v1)
    vector_b64: str     # base64-encoded raw bytes, length = dimension * 4
    creator_id: str     # node_id or DID of the original embedder; used
                        # to scope the on-chain pubkey lookup
    created_at: float   # unix epoch seconds; bound into signature
    signature_b64: str  # base64-encoded Ed25519 signature, 64 bytes
    protocol_version: int = EMBEDDING_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.EMBEDDING_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("content_hash", self.content_hash)
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("dtype", self.dtype)
        _validate_str_field("vector_b64", self.vector_b64)
        _validate_str_field("creator_id", self.creator_id)
        _validate_str_field("signature_b64", self.signature_b64)
        _validate_version(self.protocol_version)

        if not isinstance(self.dimension, int):
            raise MalformedMessageError(
                f"dimension must be int, got {type(self.dimension).__name__}"
            )
        if self.dimension <= 0:
            raise MalformedMessageError(
                f"dimension must be positive, got {self.dimension}"
            )
        if self.dimension > MAX_VECTOR_DIMENSION:
            raise MalformedMessageError(
                f"dimension exceeds MAX_VECTOR_DIMENSION "
                f"({self.dimension} > {MAX_VECTOR_DIMENSION})"
            )

        if self.dtype not in ALLOWED_DTYPES:
            raise MalformedMessageError(
                f"dtype must be one of {sorted(ALLOWED_DTYPES)}, "
                f"got {self.dtype!r}"
            )

        if not isinstance(self.created_at, (int, float)):
            raise MalformedMessageError(
                f"created_at must be numeric, got "
                f"{type(self.created_at).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "content_hash": self.content_hash,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "vector_b64": self.vector_b64,
            "creator_id": self.creator_id,
            "created_at": self.created_at,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingResponse":
        _expect_type(data, MessageType.EMBEDDING_RESPONSE)
        created_at = data.get("created_at")
        if not isinstance(created_at, (int, float)):
            raise MalformedMessageError(
                f"created_at must be numeric, got "
                f"{type(created_at).__name__}"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            content_hash=_required_str(data, "content_hash"),
            model_id=_required_str(data, "model_id"),
            dimension=_required_int(data, "dimension"),
            dtype=_required_str(data, "dtype"),
            vector_b64=_required_str(data, "vector_b64"),
            creator_id=_required_str(data, "creator_id"),
            created_at=float(created_at),
            signature_b64=_required_str(data, "signature_b64"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class ErrorResponse:
    """Server → client: a coded failure to either request type."""

    request_id: str
    code: str
    message: str
    protocol_version: int = EMBEDDING_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.ERROR.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("code", self.code)
        if not isinstance(self.message, str):
            raise MalformedMessageError(
                f"message must be a string, got "
                f"{type(self.message).__name__}"
            )
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "code": self.code,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorResponse":
        _expect_type(data, MessageType.ERROR)
        return cls(
            request_id=_required_str(data, "request_id"),
            code=_required_str(data, "code"),
            message=data.get("message", ""),
            protocol_version=_required_int(data, "protocol_version"),
        )


# --------------------------------------------------------------------------
# Canonical signing payload
# --------------------------------------------------------------------------


def canonical_signing_payload(
    content_hash: str,
    model_id: str,
    dimension: int,
    dtype: str,
    vector_bytes: bytes,
    created_at: float,
) -> bytes:
    """Build the canonical bytes that an EmbeddingResponse signature
    must cover.

    Layout:
      SIGNING_DOMAIN_TAG (32 bytes)
      || len(content_hash) as uint32 BE || content_hash (utf-8 bytes)
      || len(model_id)     as uint32 BE || model_id     (utf-8 bytes)
      || dimension         as uint32 BE
      || len(dtype)        as uint32 BE || dtype        (utf-8 bytes)
      || len(vector_bytes) as uint32 BE || vector_bytes
      || created_at_ms     as uint64 BE  (milliseconds since epoch)

    Length-prefixing every variable-length field prevents domain-tag
    confusion attacks (e.g. moving the boundary between content_hash
    and model_id while leaving the concatenated bytes the same).

    created_at is reduced to integer milliseconds for determinism —
    re-serializing a float across processes can lose precision; ms
    granularity is finer than gossip propagation already provides.
    """
    if not isinstance(content_hash, str) or not content_hash:
        raise MalformedMessageError("content_hash must be non-empty string")
    if not isinstance(model_id, str) or not model_id:
        raise MalformedMessageError("model_id must be non-empty string")
    if not isinstance(dimension, int) or dimension <= 0:
        raise MalformedMessageError("dimension must be positive int")
    if dimension > MAX_VECTOR_DIMENSION:
        raise MalformedMessageError(
            f"dimension exceeds MAX_VECTOR_DIMENSION "
            f"({dimension} > {MAX_VECTOR_DIMENSION})"
        )
    if dtype not in ALLOWED_DTYPES:
        raise MalformedMessageError(
            f"dtype must be one of {sorted(ALLOWED_DTYPES)}, "
            f"got {dtype!r}"
        )
    if not isinstance(vector_bytes, (bytes, bytearray)):
        raise MalformedMessageError("vector_bytes must be bytes")
    if len(vector_bytes) != dimension * 4:
        raise MalformedMessageError(
            f"vector_bytes length {len(vector_bytes)} does not match "
            f"dimension*4={dimension * 4} for float32"
        )
    if not isinstance(created_at, (int, float)):
        raise MalformedMessageError("created_at must be numeric")

    content_hash_b = content_hash.encode("utf-8")
    model_id_b = model_id.encode("utf-8")
    dtype_b = dtype.encode("utf-8")
    created_at_ms = int(created_at * 1000)

    parts: list[bytes] = [
        SIGNING_DOMAIN_TAG.ljust(32, b"\x00"),
        len(content_hash_b).to_bytes(4, "big"),
        content_hash_b,
        len(model_id_b).to_bytes(4, "big"),
        model_id_b,
        dimension.to_bytes(4, "big"),
        len(dtype_b).to_bytes(4, "big"),
        dtype_b,
        len(vector_bytes).to_bytes(4, "big"),
        bytes(vector_bytes),
        created_at_ms.to_bytes(8, "big", signed=False),
    ]
    return b"".join(parts)


# --------------------------------------------------------------------------
# Top-level dispatcher
# --------------------------------------------------------------------------


# Map message type tag → from_dict constructor.
MESSAGE_TYPE_REGISTRY = {
    MessageType.FIND_EMBEDDING.value: FindEmbeddingRequest.from_dict,
    MessageType.FETCH_EMBEDDING.value: FetchEmbeddingRequest.from_dict,
    MessageType.EMBEDDING_PROVIDERS.value:
        EmbeddingProvidersResponse.from_dict,
    MessageType.EMBEDDING_RESPONSE.value: EmbeddingResponse.from_dict,
    MessageType.ERROR.value: ErrorResponse.from_dict,
}


def parse_message(payload: bytes) -> Any:
    """Decode a wire-format JSON payload into the matching dataclass.

    Raises ``MalformedMessageError`` on JSON parse failure or
    missing/wrong-typed required fields.
    Raises ``UnknownMessageTypeError`` if the ``type`` field doesn't
    match a known ``MessageType``.
    Raises ``IncompatibleProtocolVersionError`` if
    ``protocol_version != EMBEDDING_DHT_PROTOCOL_VERSION``.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise MalformedMessageError(
            f"payload must be bytes, got {type(payload).__name__}"
        )
    if len(payload) > MAX_MESSAGE_BYTES:
        raise MalformedMessageError(
            f"payload exceeds MAX_MESSAGE_BYTES "
            f"({len(payload)} > {MAX_MESSAGE_BYTES})"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MalformedMessageError(f"JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise MalformedMessageError(
            f"top-level message must be a dict, got "
            f"{type(data).__name__}"
        )

    msg_type = data.get("type")
    if not isinstance(msg_type, str):
        raise MalformedMessageError(
            f"missing or non-string 'type' field; got {msg_type!r}"
        )

    constructor = MESSAGE_TYPE_REGISTRY.get(msg_type)
    if constructor is None:
        raise UnknownMessageTypeError(
            f"unknown message type {msg_type!r}; "
            f"known: {sorted(MESSAGE_TYPE_REGISTRY)}"
        )

    version = data.get("protocol_version")
    if (
        isinstance(version, int)
        and version != EMBEDDING_DHT_PROTOCOL_VERSION
    ):
        raise IncompatibleProtocolVersionError(
            f"peer protocol_version={version}; "
            f"local EMBEDDING_DHT_PROTOCOL_VERSION="
            f"{EMBEDDING_DHT_PROTOCOL_VERSION}"
        )

    return constructor(data)


def encode_message(message: Any) -> bytes:
    """Encode a message dataclass to wire bytes (JSON, UTF-8)."""
    if not hasattr(message, "to_dict"):
        raise MalformedMessageError(
            f"message must have to_dict(); got {type(message).__name__}"
        )
    return json.dumps(message.to_dict(), sort_keys=True).encode("utf-8")


# --------------------------------------------------------------------------
# Internal validation helpers
# --------------------------------------------------------------------------


def _validate_str_field(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise MalformedMessageError(
            f"{name} must be a string, got {type(value).__name__}"
        )
    if not value:
        raise MalformedMessageError(f"{name} must be non-empty")


def _validate_version(version: Any) -> None:
    if not isinstance(version, int):
        raise MalformedMessageError(
            f"protocol_version must be an int, got "
            f"{type(version).__name__}"
        )


def _expect_type(data: Dict[str, Any], expected: MessageType) -> None:
    actual = data.get("type")
    if actual != expected.value:
        raise MalformedMessageError(
            f"expected type={expected.value!r}, got {actual!r}"
        )


def _required_str(data: Dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise MalformedMessageError(
            f"{field} must be a string, got {type(value).__name__}"
        )
    if not value:
        raise MalformedMessageError(f"{field} must be non-empty")
    return value


def _required_int(data: Dict[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int):
        raise MalformedMessageError(
            f"{field} must be an int, got {type(value).__name__}"
        )
    return value
