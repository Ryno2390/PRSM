"""
Manifest DHT — wire-format protocol.

Phase 3.x.5 Task 1.

Defines the JSON-over-transport messages exchanged between
``ManifestDHTClient`` and ``ManifestDHTServer``:

  Requests (client → server):
    - FindProvidersRequest — "who can serve this model_id?"
    - FetchManifestRequest — "send me the manifest for this model_id"

  Responses (server → client):
    - ProvidersResponse    — list of (node_id, address) pairs
    - ManifestResponse     — single manifest dict (ModelManifest.to_dict)
    - ErrorResponse        — coded failure (NOT_FOUND, MALFORMED, etc.)

Wire format is JSON, dispatched by ``type`` field. Each message
carries a ``protocol_version`` so a peer running an incompatible
version is rejected at parse time rather than producing confusing
shape errors deeper in the stack.

We don't sign DHT messages themselves — the manifest payload inside
``ManifestResponse`` carries its own Ed25519 signature (Phase 3.x.2),
which the receiver verifies via the on-chain anchor (Phase 3.x.3).
The DHT layer is just routing + bytes; trust comes from the artifact,
not the transport.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple


# Wire-format version. Bump when any message dataclass adds, removes,
# or reshapes a field. Peers running mismatched versions reject each
# other at parse time — see ``parse_message``.
DHT_PROTOCOL_VERSION = 1

# Hard cap on incoming wire-message size. Manifests in practice are
# 1–10 KB (one entry per shard, ~150 bytes each); 256 KB gives ~25×
# headroom over the 10 KB p95 and ~16× over a hypothetical 16K-shard
# manifest. Anything larger is rejected at parse time before json.loads
# allocates — closes the OOM-via-huge-payload DoS surface (Phase 3.x.5
# round 1 review MEDIUM-1).
MAX_MESSAGE_BYTES = 256 * 1024

# Per-list cap on ProvidersResponse.providers. Bounds the
# response-side amplification — a hostile peer cannot return a
# million-entry list to exhaust caller memory parsing it.
MAX_PROVIDERS_PER_RESPONSE = 1024


# Tag strings used as the ``type`` field on every wire message.
# Public so callers can build ad-hoc messages without importing the
# concrete dataclass (handy for fuzz / chaos testing).
class MessageType(str, Enum):
    FIND_PROVIDERS = "find_providers"
    FETCH_MANIFEST = "fetch_manifest"
    PROVIDERS_RESPONSE = "providers_response"
    MANIFEST_RESPONSE = "manifest_response"
    ERROR = "error"


# Coded error reasons returned by the server. Strings (not ints) so
# log lines + tcpdump traces are readable; not so many codes that
# string-vs-enum overhead matters.
class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
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
    ``DHT_PROTOCOL_VERSION``. Caller should reject the message
    (and ideally skip the peer for a configurable cool-off)."""


# --------------------------------------------------------------------------
# Message dataclasses
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FindProvidersRequest:
    """Client → server: "Who can serve this model_id?"

    The server should return its locally-known providers
    (typically the server itself if it has the manifest, plus any
    peers it knows from prior find_providers iterations) up to a
    bounded count K. Iteration / convergence is the client's
    responsibility (Kademlia-style).
    """

    model_id: str
    request_id: str
    protocol_version: int = DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.FIND_PROVIDERS.value

    def __post_init__(self) -> None:
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FindProvidersRequest":
        _expect_type(data, MessageType.FIND_PROVIDERS)
        return cls(
            model_id=_required_str(data, "model_id"),
            request_id=_required_str(data, "request_id"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class FetchManifestRequest:
    """Client → server: "Send me the manifest for this model_id."

    Server returns ``ManifestResponse`` if it has the manifest
    locally, ``ErrorResponse(NOT_FOUND)`` otherwise. Server does
    NOT chase the DHT on behalf of the client — find_providers
    is the iteration primitive.
    """

    model_id: str
    request_id: str
    protocol_version: int = DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.FETCH_MANIFEST.value

    def __post_init__(self) -> None:
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "model_id": self.model_id,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FetchManifestRequest":
        _expect_type(data, MessageType.FETCH_MANIFEST)
        return cls(
            model_id=_required_str(data, "model_id"),
            request_id=_required_str(data, "request_id"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class ProviderInfo:
    """One provider's identity + address; sub-record of ProvidersResponse."""

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
class ProvidersResponse:
    """Server → client: list of providers for the requested model_id."""

    request_id: str
    providers: Tuple[ProviderInfo, ...]
    protocol_version: int = DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.PROVIDERS_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)
        if not isinstance(self.providers, tuple):
            # Coerce list → tuple (frozen dataclass; use object.__setattr__)
            object.__setattr__(self, "providers", tuple(self.providers))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "providers": [p.to_dict() for p in self.providers],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvidersResponse":
        _expect_type(data, MessageType.PROVIDERS_RESPONSE)
        providers_raw = data.get("providers")
        if not isinstance(providers_raw, list):
            raise MalformedMessageError(
                f"providers must be a list, got {type(providers_raw).__name__}"
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
class ManifestResponse:
    """Server → client: a single manifest as a dict.

    The ``manifest`` field is a ``ModelManifest.to_dict()`` result.
    The DHT protocol does not validate the manifest's internal
    shape — that's the caller's job, by passing the dict to
    ``ModelManifest.from_dict()`` and then verifying via the
    Phase 3.x.3 anchor.
    """

    request_id: str
    manifest: Dict[str, Any]
    protocol_version: int = DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.MANIFEST_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)
        if not isinstance(self.manifest, dict):
            raise MalformedMessageError(
                f"manifest must be a dict, got {type(self.manifest).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "manifest": self.manifest,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestResponse":
        _expect_type(data, MessageType.MANIFEST_RESPONSE)
        manifest = data.get("manifest")
        if not isinstance(manifest, dict):
            raise MalformedMessageError(
                f"manifest must be a dict, got {type(manifest).__name__}"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            manifest=manifest,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class ErrorResponse:
    """Server → client: a coded failure to either request type.

    ``code`` is a stable string from ``ErrorCode``; ``message`` is
    a human-readable diagnostic that callers MAY include in logs but
    MUST NOT switch on.
    """

    request_id: str
    code: str
    message: str
    protocol_version: int = DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = MessageType.ERROR.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("code", self.code)
        # message can be empty string but must be a string
        if not isinstance(self.message, str):
            raise MalformedMessageError(
                f"message must be a string, got {type(self.message).__name__}"
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
# Top-level dispatcher
# --------------------------------------------------------------------------


# Map message type tag → from_dict constructor. Public so callers
# building introspection or fuzzers can enumerate the wire format.
MESSAGE_TYPE_REGISTRY = {
    MessageType.FIND_PROVIDERS.value: FindProvidersRequest.from_dict,
    MessageType.FETCH_MANIFEST.value: FetchManifestRequest.from_dict,
    MessageType.PROVIDERS_RESPONSE.value: ProvidersResponse.from_dict,
    MessageType.MANIFEST_RESPONSE.value: ManifestResponse.from_dict,
    MessageType.ERROR.value: ErrorResponse.from_dict,
}


def parse_message(payload: bytes) -> Any:
    """Decode a wire-format JSON payload into the matching dataclass.

    Raises ``MalformedMessageError`` on JSON parse failure or
    missing/wrong-typed required fields.
    Raises ``UnknownMessageTypeError`` if the ``type`` field doesn't
    match a known ``MessageType``.
    Raises ``IncompatibleProtocolVersionError`` if
    ``protocol_version != DHT_PROTOCOL_VERSION``.
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
            f"top-level message must be a dict, got {type(data).__name__}"
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

    # Version check happens INSIDE _required_int via the dataclass
    # __post_init__, but that raises MalformedMessageError. Surface
    # the version mismatch as the dedicated exception type so callers
    # can react differently (skip peer for cool-off, etc.) rather
    # than treating it as a malformed-message hostile-peer signal.
    version = data.get("protocol_version")
    if (
        isinstance(version, int)
        and version != DHT_PROTOCOL_VERSION
    ):
        raise IncompatibleProtocolVersionError(
            f"peer protocol_version={version}; "
            f"local DHT_PROTOCOL_VERSION={DHT_PROTOCOL_VERSION}"
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
            f"protocol_version must be an int, got {type(version).__name__}"
        )
    # Note: the version-mismatch check is in parse_message rather than
    # here, so a sender constructing a future-version message in tests
    # doesn't blow up at __post_init__ time.


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
