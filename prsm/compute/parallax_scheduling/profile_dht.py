"""
Profile DHT — PRSM-native publication of (τ_g,ℓ, ρ_g,g') over Phase 6 transport.

Phase 3.x.6 Task 4 — concrete ``ProfileSource`` implementation that
satisfies the Protocol defined in ``prsm_request_router``. Replaces
the upstream Lattica dependency with PRSM's audited Phase 6 transport
(TCP/SOCKS adapters + EIP-4361 wallet binding from Phase 4).

Mirrors Phase 3.x.5 manifest DHT patterns:
  - JSON-over-transport wire format with version
  - Bounded message + payload sizes
  - Server's ``handle(bytes) → bytes`` never raises
  - Anchor-verify on read (rejects entries whose signature doesn't
    verify under the on-chain pubkey from Phase 3.x.3)
  - Per-publisher signed entries; tampering breaks the signature
  - Timestamped entries with TTL-based staleness eviction

Differs from manifest DHT in:
  - Profiles are small (~1 KB), high-frequency (republish every 2s)
    rather than large (~10 KB) and rare. Wire-format size cap is
    accordingly tighter (64 KB vs 256 KB).
  - Profiles are owned by their publisher; no "many providers serve
    one content key" semantics. The DHT is a per-node-id cache, not
    a content-addressed lookup.
  - Lookups are direct: query "what's node X's latest profile?",
    not "who serves model Y?". Matches the upstream paper §3.3
    semantics.

What this module owns:
  - Wire format + codec + protocol version
  - ``ProfileEntry`` signed dataclass + canonical signing payload
  - ``ProfileDHTServer`` — handles publish + fetch RPCs
  - ``ProfileDHTClient`` — publishes own profile, fetches peers'
  - ``ProfileDHT`` — combines both; implements ``ProfileSource``

What this module does NOT own:
  - Periodic republish scheduling — see ``republish_loop`` for a
    reference implementation; production callers use their own
    scheduler (asyncio task, threading.Timer, k8s CronJob, etc.)
  - The transport itself — ``send_message`` callable injected by
    caller (Phase 6 ``TransportAdapter``)
  - Anchor wiring — caller injects an object exposing
    ``lookup(node_id) → Optional[str]``
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from prsm.compute.parallax_scheduling.prsm_request_router import (
    ProfileSnapshot,
)
from prsm.node.identity import NodeIdentity, verify_signature


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Protocol constants
# ──────────────────────────────────────────────────────────────────────────


PROFILE_DHT_PROTOCOL_VERSION = 1
"""Wire-format version. Bump when any message dataclass changes shape."""

MAX_MESSAGE_BYTES = 64 * 1024
"""Hard cap on incoming wire bytes. Profiles are small; this gives
~64x headroom over a typical 1KB profile entry. Anything larger is
rejected at parse time before json.loads allocates."""

DEFAULT_TTL_SECONDS = 5.0
"""Cache entries older than this are treated as stale. 2.5× the
default publish interval — gives one missed republish window before
eviction. Operators can tune."""

DEFAULT_PUBLISH_INTERVAL_SECONDS = 2.0
"""Per Parallax paper §3.3: republish every 1-2 seconds."""


# ──────────────────────────────────────────────────────────────────────────
# Message types + error codes
# ──────────────────────────────────────────────────────────────────────────


class ProfileMessageType(str, Enum):
    PUBLISH_PROFILE = "publish_profile"
    PUBLISH_RESPONSE = "publish_response"
    FETCH_PROFILE = "fetch_profile"
    FETCH_RESPONSE = "fetch_response"
    ERROR = "error"


class ProfileErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    UNREGISTERED_PUBLISHER = "UNREGISTERED_PUBLISHER"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    STALE_ENTRY = "STALE_ENTRY"


# ──────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────


class ProfileProtocolError(Exception):
    """Base error for profile-DHT wire-format failures."""


class ProfileMalformedError(ProfileProtocolError):
    """Wire bytes did not parse as a valid profile-DHT message."""


class ProfileUnknownTypeError(ProfileProtocolError):
    """Message ``type`` field doesn't match a known ``ProfileMessageType``."""


class ProfileVersionMismatchError(ProfileProtocolError):
    """Peer's ``protocol_version`` doesn't match local."""


class ProfileSignatureError(Exception):
    """Profile signature failed verification under the anchored pubkey."""


# ──────────────────────────────────────────────────────────────────────────
# Signed profile entry — the payload that lives in the DHT
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignedProfileEntry:
    """A node's published profile, with Ed25519 signature.

    The signing payload is canonical-JSON of ``{node_id,
    layer_latency_ms, rtt_to_peers, timestamp_unix}`` — sorted keys,
    no whitespace. Anchor lookup at verify time resolves the
    publisher's pubkey; the signature is checked against THAT, not
    against any pubkey embedded in the entry. This prevents a
    malicious publisher from including their own pubkey + signature
    that "verifies" trivially.
    """

    node_id: str
    layer_latency_ms: float
    rtt_to_peers: Dict[str, float]
    timestamp_unix: float
    signature_b64: str

    def __post_init__(self) -> None:
        _validate_str_field("node_id", self.node_id)
        _validate_str_field("signature_b64", self.signature_b64)
        if not isinstance(self.layer_latency_ms, (int, float)):
            raise ProfileMalformedError(
                f"layer_latency_ms must be numeric, "
                f"got {type(self.layer_latency_ms).__name__}"
            )
        if self.layer_latency_ms < 0:
            raise ProfileMalformedError(
                f"layer_latency_ms must be non-negative, got "
                f"{self.layer_latency_ms}"
            )
        if not isinstance(self.rtt_to_peers, dict):
            raise ProfileMalformedError(
                f"rtt_to_peers must be dict, "
                f"got {type(self.rtt_to_peers).__name__}"
            )
        for peer_id, rtt in self.rtt_to_peers.items():
            if not isinstance(peer_id, str) or not peer_id:
                raise ProfileMalformedError(
                    f"rtt_to_peers key must be non-empty str, got {peer_id!r}"
                )
            if not isinstance(rtt, (int, float)) or rtt < 0:
                raise ProfileMalformedError(
                    f"rtt_to_peers[{peer_id!r}] must be non-negative numeric, "
                    f"got {rtt!r}"
                )
        if not isinstance(self.timestamp_unix, (int, float)):
            raise ProfileMalformedError(
                f"timestamp_unix must be numeric, "
                f"got {type(self.timestamp_unix).__name__}"
            )

    @staticmethod
    def signing_payload(
        node_id: str,
        layer_latency_ms: float,
        rtt_to_peers: Dict[str, float],
        timestamp_unix: float,
    ) -> bytes:
        """Build the canonical bytes signed by the publisher.

        Order matters: dict iteration order is implementation-defined,
        but ``json.dumps(..., sort_keys=True)`` makes it deterministic.
        Both signer and verifier MUST construct identical bytes from
        the same logical payload.
        """
        # Sort rtt_to_peers keys for determinism.
        sorted_rtts = {k: rtt_to_peers[k] for k in sorted(rtt_to_peers)}
        payload = {
            "node_id": node_id,
            "layer_latency_ms": float(layer_latency_ms),
            "rtt_to_peers": sorted_rtts,
            "timestamp_unix": float(timestamp_unix),
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def sign(
        cls,
        identity: NodeIdentity,
        layer_latency_ms: float,
        rtt_to_peers: Dict[str, float],
        timestamp_unix: Optional[float] = None,
    ) -> "SignedProfileEntry":
        """Construct + sign a fresh entry under ``identity``."""
        ts = timestamp_unix if timestamp_unix is not None else time.time()
        payload = cls.signing_payload(
            identity.node_id, layer_latency_ms, rtt_to_peers, ts
        )
        sig = identity.sign(payload)
        return cls(
            node_id=identity.node_id,
            layer_latency_ms=float(layer_latency_ms),
            rtt_to_peers=dict(rtt_to_peers),
            timestamp_unix=ts,
            signature_b64=sig,
        )

    def verify_with_anchor(self, anchor: Any) -> bool:
        """Verify this entry's signature using the publisher's pubkey
        resolved from the on-chain anchor.

        Returns False (not an exception) for any failure path so the
        caller can simply skip rejected entries — matches the
        manifest-DHT pattern. Distinct from anchor-RPC errors which
        raise (those are transient infrastructure, not "this entry
        is bad").
        """
        if anchor is None or not hasattr(anchor, "lookup"):
            return False
        publisher_pubkey_b64 = anchor.lookup(self.node_id)
        if not publisher_pubkey_b64:
            # Unregistered publisher → cannot verify. Different from
            # "wrong sig" — caller may want to log specifically.
            return False
        payload = self.signing_payload(
            self.node_id,
            self.layer_latency_ms,
            self.rtt_to_peers,
            self.timestamp_unix,
        )
        return verify_signature(
            publisher_pubkey_b64, payload, self.signature_b64
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "layer_latency_ms": self.layer_latency_ms,
            "rtt_to_peers": dict(self.rtt_to_peers),
            "timestamp_unix": self.timestamp_unix,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignedProfileEntry":
        rtts_raw = data.get("rtt_to_peers", {})
        if not isinstance(rtts_raw, dict):
            raise ProfileMalformedError(
                f"rtt_to_peers must be dict, got "
                f"{type(rtts_raw).__name__}"
            )
        return cls(
            node_id=_required_str(data, "node_id"),
            layer_latency_ms=_required_number(data, "layer_latency_ms"),
            rtt_to_peers={str(k): float(v) for k, v in rtts_raw.items()},
            timestamp_unix=_required_number(data, "timestamp_unix"),
            signature_b64=_required_str(data, "signature_b64"),
        )

    def to_snapshot(self) -> ProfileSnapshot:
        """Convert verified entry to the router-side snapshot type."""
        return ProfileSnapshot(
            node_id=self.node_id,
            layer_latency_ms=self.layer_latency_ms,
            rtt_to_peers=dict(self.rtt_to_peers),
            timestamp_unix=self.timestamp_unix,
        )


# ──────────────────────────────────────────────────────────────────────────
# Wire messages
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PublishProfileRequest:
    """Client → server: 'here is my latest profile, cache it.'

    Server validates the signature against the anchor and accepts
    or rejects. Acceptance does NOT mean other peers will see the
    entry — replication / propagation is up to the operator's
    chosen membership scheme.
    """

    request_id: str
    entry: SignedProfileEntry
    protocol_version: int = PROFILE_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ProfileMessageType.PUBLISH_PROFILE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "entry": self.entry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublishProfileRequest":
        _expect_type(data, ProfileMessageType.PUBLISH_PROFILE)
        entry_raw = data.get("entry")
        if not isinstance(entry_raw, dict):
            raise ProfileMalformedError(
                f"entry must be dict, got {type(entry_raw).__name__}"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            entry=SignedProfileEntry.from_dict(entry_raw),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class PublishResponse:
    request_id: str
    accepted: bool
    protocol_version: int = PROFILE_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ProfileMessageType.PUBLISH_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "accepted": bool(self.accepted),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublishResponse":
        _expect_type(data, ProfileMessageType.PUBLISH_RESPONSE)
        return cls(
            request_id=_required_str(data, "request_id"),
            accepted=bool(data.get("accepted", False)),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class FetchProfileRequest:
    request_id: str
    node_id: str  # which publisher's profile to fetch
    protocol_version: int = PROFILE_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ProfileMessageType.FETCH_PROFILE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("node_id", self.node_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FetchProfileRequest":
        _expect_type(data, ProfileMessageType.FETCH_PROFILE)
        return cls(
            request_id=_required_str(data, "request_id"),
            node_id=_required_str(data, "node_id"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class FetchResponse:
    request_id: str
    entry: Optional[SignedProfileEntry]  # None on cache miss
    protocol_version: int = PROFILE_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ProfileMessageType.FETCH_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "entry": self.entry.to_dict() if self.entry is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FetchResponse":
        _expect_type(data, ProfileMessageType.FETCH_RESPONSE)
        entry_raw = data.get("entry")
        entry: Optional[SignedProfileEntry] = None
        if entry_raw is not None:
            if not isinstance(entry_raw, dict):
                raise ProfileMalformedError(
                    f"entry must be dict or null, "
                    f"got {type(entry_raw).__name__}"
                )
            entry = SignedProfileEntry.from_dict(entry_raw)
        return cls(
            request_id=_required_str(data, "request_id"),
            entry=entry,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class ErrorResponse:
    request_id: str
    code: str
    message: str
    protocol_version: int = PROFILE_DHT_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ProfileMessageType.ERROR.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("code", self.code)
        if not isinstance(self.message, str):
            raise ProfileMalformedError(
                f"message must be string, got {type(self.message).__name__}"
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
        _expect_type(data, ProfileMessageType.ERROR)
        return cls(
            request_id=_required_str(data, "request_id"),
            code=_required_str(data, "code"),
            message=data.get("message", ""),
            protocol_version=_required_int(data, "protocol_version"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Codec
# ──────────────────────────────────────────────────────────────────────────


_MESSAGE_TYPE_REGISTRY = {
    ProfileMessageType.PUBLISH_PROFILE.value: PublishProfileRequest.from_dict,
    ProfileMessageType.PUBLISH_RESPONSE.value: PublishResponse.from_dict,
    ProfileMessageType.FETCH_PROFILE.value: FetchProfileRequest.from_dict,
    ProfileMessageType.FETCH_RESPONSE.value: FetchResponse.from_dict,
    ProfileMessageType.ERROR.value: ErrorResponse.from_dict,
}


def parse_message(payload: bytes) -> Any:
    """Decode a wire-format payload into the matching dataclass.

    Mirrors the manifest-DHT parser. Size cap fires before
    ``json.loads`` allocates; version mismatch raises a dedicated
    exception so callers can distinguish from malformed input.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise ProfileMalformedError(
            f"payload must be bytes, got {type(payload).__name__}"
        )
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProfileMalformedError(
            f"payload exceeds MAX_MESSAGE_BYTES "
            f"({len(payload)} > {MAX_MESSAGE_BYTES})"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProfileMalformedError(f"JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileMalformedError(
            f"top-level message must be a dict, got {type(data).__name__}"
        )
    msg_type = data.get("type")
    if not isinstance(msg_type, str):
        raise ProfileMalformedError(
            f"missing or non-string 'type' field; got {msg_type!r}"
        )
    constructor = _MESSAGE_TYPE_REGISTRY.get(msg_type)
    if constructor is None:
        raise ProfileUnknownTypeError(
            f"unknown message type {msg_type!r}; "
            f"known: {sorted(_MESSAGE_TYPE_REGISTRY)}"
        )
    version = data.get("protocol_version")
    if (
        isinstance(version, int)
        and version != PROFILE_DHT_PROTOCOL_VERSION
    ):
        raise ProfileVersionMismatchError(
            f"peer protocol_version={version}; "
            f"local PROFILE_DHT_PROTOCOL_VERSION={PROFILE_DHT_PROTOCOL_VERSION}"
        )
    return constructor(data)


def encode_message(message: Any) -> bytes:
    if not hasattr(message, "to_dict"):
        raise ProfileMalformedError(
            f"message must have to_dict(); got {type(message).__name__}"
        )
    return json.dumps(message.to_dict(), sort_keys=True).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────────────────


_UNKNOWN_REQUEST_ID = "<unknown>"


class ProfileDHTServer:
    """Per-node server-side cache + RPC handler.

    Maintains an in-memory ``node_id → SignedProfileEntry`` map. New
    publishes overwrite older ones for the same node_id IF the new
    entry's timestamp is more recent and the signature verifies under
    the anchor. Fetches return the cached entry or a NOT_FOUND error.

    ``handle(bytes) → bytes`` NEVER raises — every failure path is
    mapped to an encoded ErrorResponse so the transport caller never
    needs to handle exceptions.
    """

    def __init__(
        self,
        anchor: Any,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if anchor is None or not hasattr(anchor, "lookup"):
            raise RuntimeError(
                "ProfileDHTServer requires an anchor with .lookup(node_id) → "
                f"Optional[str]; got {type(anchor).__name__}"
            )
        self._anchor = anchor
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._cache: Dict[str, SignedProfileEntry] = {}
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────

    def handle(self, request_bytes: bytes) -> bytes:
        """Parse → dispatch → encode response. Never raises."""
        try:
            request = parse_message(request_bytes)
        except ProfileVersionMismatchError as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                ProfileErrorCode.UNSUPPORTED_VERSION,
                str(exc),
            )
        except (ProfileMalformedError, ProfileUnknownTypeError) as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                ProfileErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except ProfileProtocolError as exc:
            return self._error(
                _UNKNOWN_REQUEST_ID,
                ProfileErrorCode.MALFORMED_REQUEST,
                str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ProfileDHTServer.handle: unexpected parse failure")
            return self._error(
                _UNKNOWN_REQUEST_ID,
                ProfileErrorCode.INTERNAL_ERROR,
                f"internal error: {exc.__class__.__name__}",
            )

        request_id = (
            getattr(request, "request_id", "") or _UNKNOWN_REQUEST_ID
        )
        try:
            if isinstance(request, PublishProfileRequest):
                return self._handle_publish(request)
            if isinstance(request, FetchProfileRequest):
                return self._handle_fetch(request)
            return self._error(
                request_id,
                ProfileErrorCode.MALFORMED_REQUEST,
                f"server received non-request type: {type(request).__name__}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ProfileDHTServer.handle: unexpected dispatch failure for "
                "request_id=%r",
                request_id,
            )
            return self._error(
                request_id,
                ProfileErrorCode.INTERNAL_ERROR,
                f"internal error during dispatch: {exc.__class__.__name__}",
            )

    def get_cached(self, node_id: str) -> Optional[SignedProfileEntry]:
        """Direct cache read — for in-process use (tests, the
        ``ProfileSource`` wrapper). Returns None if absent or stale."""
        with self._lock:
            entry = self._cache.get(node_id)
        if entry is None:
            return None
        if self._is_stale(entry):
            return None
        return entry

    def evict_stale(self) -> int:
        """Drop all entries past TTL. Returns count evicted. Operators
        may call this from a background thread; it's also called
        opportunistically by ``get_cached``."""
        with self._lock:
            stale_ids = [
                nid for nid, entry in self._cache.items()
                if self._is_stale(entry)
            ]
            for nid in stale_ids:
                del self._cache[nid]
        return len(stale_ids)

    # ── handlers ──────────────────────────────────────────────────────

    def _handle_publish(self, request: PublishProfileRequest) -> bytes:
        entry = request.entry
        # Anchor-verify before accepting. False → reject with
        # SIGNATURE_INVALID OR UNREGISTERED_PUBLISHER (we can't always
        # tell which without re-querying anchor; surface as
        # SIGNATURE_INVALID — operators check anchor side-channel).
        if not entry.verify_with_anchor(self._anchor):
            # Disambiguate the common case: publisher not registered.
            looked_up = self._anchor.lookup(entry.node_id)
            if not looked_up:
                return self._error(
                    request.request_id,
                    ProfileErrorCode.UNREGISTERED_PUBLISHER,
                    f"publisher {entry.node_id!r} not on anchor",
                )
            return self._error(
                request.request_id,
                ProfileErrorCode.SIGNATURE_INVALID,
                f"profile signature for {entry.node_id!r} did not verify",
            )

        # Stale-on-arrival check — entry's own timestamp says it's
        # already past TTL. Reject; sender's clock is broken or
        # they're replaying old data.
        if self._is_stale(entry):
            return self._error(
                request.request_id,
                ProfileErrorCode.STALE_ENTRY,
                f"entry timestamp {entry.timestamp_unix} is past TTL",
            )

        # Latest-timestamp wins per-node.
        with self._lock:
            existing = self._cache.get(entry.node_id)
            if existing is not None and existing.timestamp_unix >= entry.timestamp_unix:
                # Out-of-order delivery; older publish lost the race.
                # Not an error — just don't overwrite.
                return encode_message(
                    PublishResponse(
                        request_id=request.request_id, accepted=False
                    )
                )
            self._cache[entry.node_id] = entry

        return encode_message(
            PublishResponse(request_id=request.request_id, accepted=True)
        )

    def _handle_fetch(self, request: FetchProfileRequest) -> bytes:
        entry = self.get_cached(request.node_id)
        return encode_message(
            FetchResponse(request_id=request.request_id, entry=entry)
        )

    # ── internals ─────────────────────────────────────────────────────

    def _is_stale(self, entry: SignedProfileEntry) -> bool:
        return (self._clock() - entry.timestamp_unix) > self._ttl

    def _error(
        self,
        request_id: str,
        code: ProfileErrorCode,
        message: str,
    ) -> bytes:
        return encode_message(
            ErrorResponse(
                request_id=request_id,
                code=code.value,
                message=message,
            )
        )


# ──────────────────────────────────────────────────────────────────────────
# Client + ProfileSource implementation
# ──────────────────────────────────────────────────────────────────────────


# Synchronous request/response: (peer_address, request_bytes) → response_bytes.
# Production wiring: Phase 6 transport adapter. Tests inject a fake.
SendMessageFn = Callable[[str, bytes], bytes]


class ProfileDHT:
    """Combined client + server + ``ProfileSource`` impl.

    Each operator runs one ``ProfileDHT`` per node. It:
      - Maintains a ``ProfileDHTServer`` for incoming requests
      - Publishes own profile to a configured peer set
      - Fetches peers' profiles into the local server cache
      - Implements ``ProfileSource`` Protocol so ``RequestRouter``
        can consume it directly

    Peer set is operator-configured (out of scope for this module —
    typically derived from the manifest DHT's known publisher set
    in the same region).
    """

    def __init__(
        self,
        identity: NodeIdentity,
        anchor: Any,
        send_message: SendMessageFn,
        peers: List[str],  # peer addresses; "host:port" form
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        publish_interval_seconds: float = DEFAULT_PUBLISH_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._identity = identity
        self._anchor = anchor
        self._send = send_message
        self._peers = list(peers)
        self._publish_interval = float(publish_interval_seconds)
        self._clock = clock
        self.server = ProfileDHTServer(
            anchor=anchor, ttl_seconds=ttl_seconds, clock=clock
        )

    @property
    def identity(self) -> NodeIdentity:
        return self._identity

    def update_peers(self, peers: List[str]) -> None:
        """Replace the peer set. Called by the orchestrator when the
        manifest DHT's known-peer view changes (member join/leave)."""
        self._peers = list(peers)

    def publish_self(
        self,
        layer_latency_ms: float,
        rtt_to_peers: Dict[str, float],
    ) -> int:
        """Sign + broadcast own profile to all peers.

        Returns the count of peers that accepted (best-effort —
        unreachable peers / version-mismatched peers are skipped, not
        a failure of the call). Caller invokes periodically per
        ``publish_interval_seconds``.
        """
        entry = SignedProfileEntry.sign(
            identity=self._identity,
            layer_latency_ms=layer_latency_ms,
            rtt_to_peers=rtt_to_peers,
            timestamp_unix=self._clock(),
        )
        # Self-cache so local fetches see own latest immediately.
        with self.server._lock:  # noqa: SLF001
            self.server._cache[entry.node_id] = entry  # noqa: SLF001

        accepted = 0
        request = PublishProfileRequest(
            request_id=_new_request_id(self._clock),
            entry=entry,
        )
        encoded = encode_message(request)
        for peer in self._peers:
            try:
                response_bytes = self._send(peer, encoded)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "publish_self: peer %s send failed: %s", peer, exc
                )
                continue
            try:
                response = parse_message(response_bytes)
            except ProfileProtocolError as exc:
                logger.debug(
                    "publish_self: peer %s response unparseable: %s",
                    peer, exc,
                )
                continue
            if isinstance(response, PublishResponse) and response.accepted:
                accepted += 1
        return accepted

    def fetch_peer(self, node_id: str) -> Optional[SignedProfileEntry]:
        """Query peers in order until one returns a verified entry
        for ``node_id``. Caches the verified entry locally on success.
        Returns None if no peer has it or all responses fail
        verification.
        """
        # Local cache hit short-circuits.
        cached = self.server.get_cached(node_id)
        if cached is not None:
            return cached

        request = FetchProfileRequest(
            request_id=_new_request_id(self._clock),
            node_id=node_id,
        )
        encoded = encode_message(request)
        for peer in self._peers:
            try:
                response_bytes = self._send(peer, encoded)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "fetch_peer: peer %s send failed: %s", peer, exc
                )
                continue
            try:
                response = parse_message(response_bytes)
            except ProfileProtocolError as exc:
                logger.debug(
                    "fetch_peer: peer %s response unparseable: %s",
                    peer, exc,
                )
                continue
            if not isinstance(response, FetchResponse):
                continue
            if response.entry is None:
                continue
            # Verify before trusting peer-served bytes.
            if not response.entry.verify_with_anchor(self._anchor):
                logger.debug(
                    "fetch_peer: peer %s served entry for %s that failed "
                    "anchor verify; trying next peer",
                    peer, node_id,
                )
                continue
            # Cache locally so subsequent get_snapshot hits the cache.
            with self.server._lock:  # noqa: SLF001
                existing = self.server._cache.get(node_id)  # noqa: SLF001
                if (
                    existing is None
                    or existing.timestamp_unix < response.entry.timestamp_unix
                ):
                    self.server._cache[node_id] = response.entry  # noqa: SLF001
            return response.entry
        return None

    # ── ProfileSource Protocol ────────────────────────────────────────

    def get_snapshot(self, node_id: str) -> Optional[ProfileSnapshot]:
        """Return the latest verified non-stale snapshot for ``node_id``.

        Resolution order:
          1. Local server cache (verified at publish/fetch time).
          2. Fall through to fetch_peer — which will populate the
             cache on success.

        Stale entries are eviction-checked by the server's
        ``get_cached`` path; we don't double-check here.
        """
        cached = self.server.get_cached(node_id)
        if cached is not None:
            return cached.to_snapshot()
        fetched = self.fetch_peer(node_id)
        if fetched is None:
            return None
        return fetched.to_snapshot()


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────


def _validate_str_field(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ProfileMalformedError(
            f"{name} must be str, got {type(value).__name__}"
        )
    if not value:
        raise ProfileMalformedError(f"{name} must be non-empty")


def _validate_version(version: Any) -> None:
    if not isinstance(version, int):
        raise ProfileMalformedError(
            f"protocol_version must be int, got {type(version).__name__}"
        )


def _expect_type(data: Dict[str, Any], expected: ProfileMessageType) -> None:
    actual = data.get("type")
    if actual != expected.value:
        raise ProfileMalformedError(
            f"expected type={expected.value!r}, got {actual!r}"
        )


def _required_str(data: Dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ProfileMalformedError(
            f"{field_name} must be non-empty str"
        )
    return value


def _required_int(data: Dict[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    if not isinstance(value, int):
        raise ProfileMalformedError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    return value


def _required_number(data: Dict[str, Any], field_name: str) -> float:
    value = data.get(field_name)
    if not isinstance(value, (int, float)):
        raise ProfileMalformedError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    return float(value)


def _new_request_id(clock: Callable[[], float] = time.time) -> str:
    """Lightweight monotonic-enough request_id. Tests can swap clock
    for determinism."""
    import secrets
    return f"prof-{int(clock() * 1000)}-{secrets.token_hex(4)}"
