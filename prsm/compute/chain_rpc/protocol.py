"""Phase 3.x.7 Task 1 — Cross-host ChainExecutor wire protocol.

The wire layer for activation handoff between consecutive stages of
a `GPUChain` (Phase 3.x.6). Mirrors the Phase 3.x.5 manifest DHT and
Phase 3.x.6 profile DHT patterns:
  - JSON-canonical-bytes wire format with version
  - Bounded message size (handshake messages only — large activation
    blobs ride the Phase 6 streaming chunker out-of-band)
  - All wire dataclasses are immutable (`@dataclass(frozen=True)`)
  - Anchor-verify-on-read for `HandoffToken`: pubkey resolved from the
    Phase 3.x.3 anchor at verify time, never trusted from any embedded
    field

Three wire types:

  RunLayerSliceRequest    Executor (or upstream stage) → stage. Carries
                          model_id + layer_range + activation blob +
                          signed `HandoffToken`. Stage validates the
                          token against the anchor before executing.

  RunLayerSliceResponse   Stage → executor on success. Carries downstream
                          activation blob + per-stage TEE attestation +
                          stage's signature over the response.

  StageError              Stage → executor on any failure. Structured
                          enum-coded error so the client can route to
                          a structured InferenceResult.failure rather
                          than parsing prose.

Plus the load-bearing trust artifact:

  HandoffToken            Settler-signed credential authorizing one
                          specific stage to execute one specific
                          (request, layer-range) at the settler's
                          expense. Bound to chain_stage_index +
                          request_id; not replayable across requests
                          or stages.

Activation tensor encoding sits in `activation_codec.py` (Task 3); the
wire types here just carry the encoded bytes + shape + dtype string.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, verify_signature
from prsm.node.shard_streaming import ShardManifest


# ──────────────────────────────────────────────────────────────────────────
# Protocol constants
# ──────────────────────────────────────────────────────────────────────────


CHAIN_RPC_PROTOCOL_VERSION = 2
"""Local wire-format version. Bumped to 2 in Phase 3.x.7.1 to add the
optional ``activation_manifest`` field on ``RunLayerSliceRequest`` /
``Response`` for the chunked-streaming path. v1 inline messages
remain byte-identical for the signing payload (the new field is
omitted from the payload when ``activation_manifest is None``), so
inline interop with v1 peers is preserved."""

SUPPORTED_PROTOCOL_VERSIONS: frozenset = frozenset({1, 2})
"""Protocol versions the parser accepts. v2 nodes accept both v1 and
v2 messages from peers (servers are backward-compat). v1 nodes
accept only v1; a v2 client sending a v2-formatted message to a v1
server gets rejected at the v1 parser. Production rolling-deploy
flows handle this by upgrading servers before clients."""

MAX_HANDSHAKE_BYTES = 64 * 1024 * 1024
"""Hard cap on a JSON-encoded chain-RPC message.

v1 inlines activation blobs hex-encoded inside the JSON envelope.
Hex doubles size + JSON adds modest framing overhead; a 64 MiB cap
swallows ~25 MiB of raw activation bytes — enough for typical LLM
activations (e.g., 2048-token × 4096-dim float16 ≈ 16 MiB raw,
≈ 33 MiB hex+JSON).

Future work (Phase 3.x.7.x): wire the Phase 6 streaming chunker
(``activation_codec.chunk_activation``) for activations exceeding
``CHUNK_THRESHOLD_BYTES`` so the wire layer doesn't have to allocate
the full hex string in memory. The codec is shipped (Task 3); only
the transport-side wiring remains. Until then, the inline path
handles the production-sized activation regime by raising the cap.

Stages enforcing a different ceiling can wrap parse_message in their
own size check; the wire protocol's contract is "messages larger
than this are rejected at parse time," not "messages must fit this
cap."
"""


# ──────────────────────────────────────────────────────────────────────────
# Message types + error codes
# ──────────────────────────────────────────────────────────────────────────


class ChainRpcMessageType(str, Enum):
    RUN_LAYER_SLICE_REQUEST = "run_layer_slice_request"
    RUN_LAYER_SLICE_RESPONSE = "run_layer_slice_response"
    STAGE_ERROR = "stage_error"
    ACTIVATION_CHUNK = "activation_chunk"  # v2: streamed-path chunk frame
    # Phase 3.x.8 — streaming-token output (additive on protocol v2):
    TOKEN_FRAME = "token_frame"  # incremental token output from tail stage
    STREAM_FINAL_FRAME = "stream_final_frame"  # terminal frame; embeds signed response


class StageErrorCode(str, Enum):
    """Structured error codes for cross-host stage failures.

    Values are stable strings (kept lowercase + dash-free) for forward
    compatibility — adding new codes is non-breaking, but renaming an
    existing code requires a protocol-version bump.
    """

    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    INVALID_TOKEN = "INVALID_TOKEN"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    SHARD_MISSING = "SHARD_MISSING"
    TIER_GATE = "TIER_GATE"
    LAYER_RANGE_INVALID = "LAYER_RANGE_INVALID"
    ACTIVATION_INVALID = "ACTIVATION_INVALID"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ──────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────


class ChainRpcProtocolError(Exception):
    """Base error for chain-RPC wire-format failures."""


class ChainRpcMalformedError(ChainRpcProtocolError):
    """Wire bytes did not parse as a valid chain-RPC message."""


class ChainRpcUnknownTypeError(ChainRpcProtocolError):
    """Message ``type`` field doesn't match a known ``ChainRpcMessageType``."""


class ChainRpcVersionMismatchError(ChainRpcProtocolError):
    """Peer's ``protocol_version`` doesn't match local."""


# ──────────────────────────────────────────────────────────────────────────
# HandoffToken — the load-bearing trust artifact
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HandoffToken:
    """Settler-signed credential authorizing one stage to execute one
    (request, layer-range) at the settler's expense.

    The ``settler_node_id`` is the node that's paying for the chain —
    typically the same node that signs the final ``InferenceReceipt``.
    Stages verify the token's signature against the settler's pubkey
    via the Phase 3.x.3 anchor before doing any work.

    Bindings:
      - ``request_id``: a stage cannot reuse a token across requests
      - ``chain_stage_index``: a stage cannot reuse a token at a
        different position in the chain
      - ``deadline_unix``: stale tokens are rejected; bounds the
        replay window even if a token leaks

    Security model: forging a token requires forging an Ed25519
    signature against an anchor-registered pubkey. A stage that
    accepts a forged token has not been compromised — its pre-work
    validation simply failed. We treat forged tokens as protocol
    bugs rather than attestation failures (Phase 7.1 challenges fire
    on output mismatch, not on token-validation failure).
    """

    request_id: str
    settler_node_id: str
    chain_stage_index: int
    chain_total_stages: int
    deadline_unix: float
    signature_b64: str

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("settler_node_id", self.settler_node_id)
        _validate_str_field("signature_b64", self.signature_b64)
        # Reject bool: isinstance(True, int) is True in Python; without
        # this guard a peer sending {"chain_stage_index": false} would
        # produce a token with bool-typed numeric fields.
        if (
            not isinstance(self.chain_stage_index, int)
            or isinstance(self.chain_stage_index, bool)
            or self.chain_stage_index < 0
        ):
            raise ChainRpcMalformedError(
                f"chain_stage_index must be non-negative int, got "
                f"{self.chain_stage_index!r}"
            )
        if (
            not isinstance(self.chain_total_stages, int)
            or isinstance(self.chain_total_stages, bool)
            or self.chain_total_stages <= 0
        ):
            raise ChainRpcMalformedError(
                f"chain_total_stages must be positive int, got "
                f"{self.chain_total_stages!r}"
            )
        if self.chain_stage_index >= self.chain_total_stages:
            raise ChainRpcMalformedError(
                f"chain_stage_index ({self.chain_stage_index}) must be < "
                f"chain_total_stages ({self.chain_total_stages})"
            )
        if not isinstance(self.deadline_unix, (int, float)):
            raise ChainRpcMalformedError(
                f"deadline_unix must be numeric, got "
                f"{type(self.deadline_unix).__name__}"
            )

    @staticmethod
    def signing_payload(
        request_id: str,
        settler_node_id: str,
        chain_stage_index: int,
        chain_total_stages: int,
        deadline_unix: float,
    ) -> bytes:
        """Canonical bytes signed by the settler. Both signer and
        verifier MUST construct identical bytes from the same logical
        token, so all numeric fields are coerced to their JSON-stable
        types and `sort_keys=True` enforces deterministic key ordering."""
        payload = {
            "request_id": request_id,
            "settler_node_id": settler_node_id,
            "chain_stage_index": int(chain_stage_index),
            "chain_total_stages": int(chain_total_stages),
            "deadline_unix": float(deadline_unix),
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def sign(
        cls,
        identity: NodeIdentity,
        request_id: str,
        chain_stage_index: int,
        chain_total_stages: int,
        deadline_unix: float,
    ) -> "HandoffToken":
        """Mint + sign a fresh token under the settler ``identity``."""
        payload = cls.signing_payload(
            request_id,
            identity.node_id,
            chain_stage_index,
            chain_total_stages,
            deadline_unix,
        )
        sig = identity.sign(payload)
        return cls(
            request_id=request_id,
            settler_node_id=identity.node_id,
            chain_stage_index=int(chain_stage_index),
            chain_total_stages=int(chain_total_stages),
            deadline_unix=float(deadline_unix),
            signature_b64=sig,
        )

    def verify_with_anchor(self, anchor: Any) -> bool:
        """Verify this token's signature using the settler's pubkey
        resolved from the on-chain anchor.

        Returns False (not an exception) for any failure path so the
        caller can simply route to ``StageError(INVALID_TOKEN)`` —
        matches the Phase 3.x.5 / 3.x.6 anchor-verify pattern. Distinct
        from anchor-RPC errors which raise (those are transient
        infrastructure, not "this token is bad").

        Anchor-verify-on-read invariant: the pubkey is resolved from
        the anchor via ``settler_node_id``. The token does NOT carry
        an embedded pubkey field; a malicious settler cannot include
        their own pubkey + signature that "verifies" trivially.
        """
        if anchor is None or not hasattr(anchor, "lookup"):
            return False
        settler_pubkey_b64 = anchor.lookup(self.settler_node_id)
        if not settler_pubkey_b64:
            return False
        payload = self.signing_payload(
            self.request_id,
            self.settler_node_id,
            self.chain_stage_index,
            self.chain_total_stages,
            self.deadline_unix,
        )
        return verify_signature(
            settler_pubkey_b64, payload, self.signature_b64
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "settler_node_id": self.settler_node_id,
            "chain_stage_index": self.chain_stage_index,
            "chain_total_stages": self.chain_total_stages,
            "deadline_unix": self.deadline_unix,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffToken":
        return cls(
            request_id=_required_str(data, "request_id"),
            settler_node_id=_required_str(data, "settler_node_id"),
            chain_stage_index=_required_int(data, "chain_stage_index"),
            chain_total_stages=_required_int(data, "chain_total_stages"),
            deadline_unix=_required_number(data, "deadline_unix"),
            signature_b64=_required_str(data, "signature_b64"),
        )


# ──────────────────────────────────────────────────────────────────────────
# ShardManifest serialization helpers
# (Phase 6 ShardManifest is owned by node/shard_streaming; we serialize
# it inline here to keep the wire format self-contained.)
# ──────────────────────────────────────────────────────────────────────────


def _shard_manifest_to_dict(m: ShardManifest) -> Dict[str, Any]:
    return {
        "shard_id": m.shard_id,
        "payload_sha256": m.payload_sha256,
        "payload_bytes": m.payload_bytes,
        "total_chunks": m.total_chunks,
        "chunk_bytes": m.chunk_bytes,
    }


def _shard_manifest_from_dict(data: Dict[str, Any]) -> ShardManifest:
    if not isinstance(data, dict):
        raise ChainRpcMalformedError(
            f"shard_manifest must be dict, got {type(data).__name__}"
        )
    return ShardManifest(
        shard_id=_required_str(data, "shard_id"),
        payload_sha256=_required_str(data, "payload_sha256"),
        payload_bytes=_required_int(data, "payload_bytes"),
        total_chunks=_required_int(data, "total_chunks"),
        chunk_bytes=_required_int(data, "chunk_bytes"),
    )


def _validate_shard_manifest(m: ShardManifest, *, field: str) -> None:
    """Structural sanity checks on a manifest before signing /
    verification. Stricter than ShardManifest's own validation;
    catches obvious malformations a peer might inject."""
    if not isinstance(m, ShardManifest):
        raise ChainRpcMalformedError(
            f"{field} must be ShardManifest, got {type(m).__name__}"
        )
    if m.payload_bytes < 0:
        raise ChainRpcMalformedError(
            f"{field}.payload_bytes must be non-negative, got {m.payload_bytes}"
        )
    if m.total_chunks < 0:
        raise ChainRpcMalformedError(
            f"{field}.total_chunks must be non-negative, got {m.total_chunks}"
        )
    if m.chunk_bytes <= 0:
        raise ChainRpcMalformedError(
            f"{field}.chunk_bytes must be positive, got {m.chunk_bytes}"
        )
    if not m.shard_id:
        raise ChainRpcMalformedError(
            f"{field}.shard_id must be non-empty"
        )
    # payload_sha256 must look like a sha256 hex string.
    if (
        not isinstance(m.payload_sha256, str)
        or len(m.payload_sha256) != 64
        or any(c not in "0123456789abcdef" for c in m.payload_sha256)
    ):
        raise ChainRpcMalformedError(
            f"{field}.payload_sha256 must be 64-char lowercase hex, "
            f"got {m.payload_sha256!r}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Wire messages
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivationChunk:
    """Per-chunk frame for the v2 streaming path.

    Wraps a single Phase 6 ``ShardChunk`` for cross-host transport.
    Bound to its parent ``RunLayerSliceRequest`` / ``Response`` via
    ``request_id`` so a relay can't splice chunks from one stream
    into another.

    Chunks are small (≤1 MiB by default — well under
    ``MAX_HANDSHAKE_BYTES``). The streaming transport sends them as
    individual frames over a bidi gRPC stream; the wire format here
    just describes the on-the-wire encoding of one frame.
    """

    request_id: str
    sequence: int  # 0-indexed; matches ShardChunk.sequence
    data: bytes
    chunk_sha256: str  # hex digest of `data`
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.ACTIVATION_CHUNK.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("chunk_sha256", self.chunk_sha256)
        _validate_version(self.protocol_version)
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise ChainRpcMalformedError(
                f"sequence must be non-negative int, got {self.sequence!r}"
            )
        if not isinstance(self.data, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"data must be bytes, got {type(self.data).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "data_hex": bytes(self.data).hex(),
            "chunk_sha256": self.chunk_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivationChunk":
        _expect_type(data, ChainRpcMessageType.ACTIVATION_CHUNK)
        try:
            data_bytes = bytes.fromhex(_required_str(data, "data_hex"))
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"data_hex is not valid hex: {exc}"
            ) from exc
        return cls(
            request_id=_required_str(data, "request_id"),
            sequence=_required_int(data, "sequence"),
            data=data_bytes,
            chunk_sha256=_required_str(data, "chunk_sha256"),
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class RunLayerSliceRequest:
    """Executor (or upstream stage) → stage: 'run layers
    [layer_range[0], layer_range[1]) on this activation'."""

    request_id: str
    model_id: str
    layer_range: Tuple[int, int]  # half-open [start, end)
    privacy_tier: PrivacyLevel
    content_tier: ContentTier
    activation_blob: bytes
    activation_shape: Tuple[int, ...]
    activation_dtype: str
    upstream_token: HandoffToken
    deadline_unix: float
    # v2 streaming: when present, ``activation_blob`` MUST be empty
    # (b"") and the bytes ride out-of-band as ``ActivationChunk``
    # frames over the streaming transport. ``activation_shape`` +
    # ``activation_dtype`` still describe the post-assembly tensor
    # so the server can decode without an extra round-trip. The
    # manifest's ``payload_sha256`` is the cryptographic commitment
    # to the to-be-assembled bytes.
    activation_manifest: Optional[ShardManifest] = None
    # Phase 3.x.8 — streaming-token output. When True AND the
    # dispatched stage is the chain tail (``is_final_stage`` at the
    # server side), the server returns a TokenFrame stream followed
    # by a StreamFinalFrame instead of a unary RunLayerSliceResponse.
    # When True on a non-tail stage the server rejects with
    # MALFORMED_REQUEST. When False (default), the wire format is
    # byte-identical to a pre-3.x.8 message — the field is OMITTED
    # from the canonical JSON entirely (mirroring the
    # activation_manifest conditional encoding pattern).
    streaming: bool = False
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("model_id", self.model_id)
        _validate_str_field("activation_dtype", self.activation_dtype)
        _validate_version(self.protocol_version)
        if not isinstance(self.layer_range, tuple) or len(self.layer_range) != 2:
            raise ChainRpcMalformedError(
                f"layer_range must be (start, end) tuple, got "
                f"{self.layer_range!r}"
            )
        start, end = self.layer_range
        if not isinstance(start, int) or not isinstance(end, int):
            raise ChainRpcMalformedError(
                f"layer_range entries must be int, got ({type(start).__name__}, "
                f"{type(end).__name__})"
            )
        if start < 0 or end <= start:
            raise ChainRpcMalformedError(
                f"layer_range must satisfy 0 <= start < end, got "
                f"({start}, {end})"
            )
        if not isinstance(self.activation_blob, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"activation_blob must be bytes, got "
                f"{type(self.activation_blob).__name__}"
            )
        if not isinstance(self.activation_shape, tuple):
            raise ChainRpcMalformedError(
                f"activation_shape must be tuple, got "
                f"{type(self.activation_shape).__name__}"
            )
        for dim in self.activation_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ChainRpcMalformedError(
                    f"activation_shape entries must be positive int, got "
                    f"{dim!r}"
                )
        if not isinstance(self.privacy_tier, PrivacyLevel):
            raise ChainRpcMalformedError(
                f"privacy_tier must be PrivacyLevel, got "
                f"{type(self.privacy_tier).__name__}"
            )
        if not isinstance(self.content_tier, ContentTier):
            raise ChainRpcMalformedError(
                f"content_tier must be ContentTier, got "
                f"{type(self.content_tier).__name__}"
            )
        if not isinstance(self.upstream_token, HandoffToken):
            raise ChainRpcMalformedError(
                f"upstream_token must be HandoffToken, got "
                f"{type(self.upstream_token).__name__}"
            )
        # Cross-field consistency: token's request_id must match the
        # request's request_id. Forging this would require forging the
        # token signature, but we still want a parse-time fast-fail.
        if self.upstream_token.request_id != self.request_id:
            raise ChainRpcMalformedError(
                f"upstream_token.request_id ({self.upstream_token.request_id!r}) "
                f"must match request.request_id ({self.request_id!r})"
            )
        if not isinstance(self.deadline_unix, (int, float)):
            raise ChainRpcMalformedError(
                f"deadline_unix must be numeric, got "
                f"{type(self.deadline_unix).__name__}"
            )
        # v2 inline-XOR-streamed integrity check. Exactly one of the
        # two payload paths must be active: inline blob (non-empty) OR
        # streamed manifest. Both-or-neither is malformed.
        #   - manifest present + blob non-empty → ambiguous (M3 round-1)
        #   - manifest absent  + blob empty     → no payload (M3 round-1)
        # ``activation_shape`` entries are positive ints (>0), so any
        # well-formed inline activation has at least dtype.itemsize
        # bytes and an empty blob is structurally meaningless.
        if self.activation_manifest is not None:
            _validate_shard_manifest(
                self.activation_manifest, field="activation_manifest"
            )
            if self.activation_blob:
                raise ChainRpcMalformedError(
                    "streamed mode: activation_blob must be empty when "
                    "activation_manifest is present (chunks ride out-"
                    "of-band)"
                )
        elif not self.activation_blob:
            raise ChainRpcMalformedError(
                "request has neither inline activation_blob nor "
                "activation_manifest — exactly one payload path is "
                "required"
            )
        # Phase 3.x.8 — streaming flag must be a real bool (not int).
        # Catches the same M1-style coercion hole the protocol_version
        # check defends against.
        if not isinstance(self.streaming, bool):
            raise ChainRpcMalformedError(
                f"streaming must be bool, got {type(self.streaming).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        # activation_blob → hex for JSON-safety. For streamed (v2)
        # requests, activation_blob is empty bytes and the manifest
        # describes the chunked payload.
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "layer_range": list(self.layer_range),
            "privacy_tier": self.privacy_tier.value,
            "content_tier": self.content_tier.value,
            "activation_blob_hex": bytes(self.activation_blob).hex(),
            "activation_shape": list(self.activation_shape),
            "activation_dtype": self.activation_dtype,
            "upstream_token": self.upstream_token.to_dict(),
            "deadline_unix": self.deadline_unix,
        }
        if self.activation_manifest is not None:
            out["activation_manifest"] = _shard_manifest_to_dict(
                self.activation_manifest
            )
        # Conditional encoding: streaming=False omits the field
        # entirely so a v2-but-pre-3.x.8 serializer's bytes are
        # byte-identical to a 3.x.8 serializer's. Only streaming=True
        # adds the key.
        if self.streaming:
            out["streaming"] = True
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunLayerSliceRequest":
        _expect_type(data, ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST)
        token_raw = data.get("upstream_token")
        if not isinstance(token_raw, dict):
            raise ChainRpcMalformedError(
                f"upstream_token must be dict, got {type(token_raw).__name__}"
            )
        layer_range_raw = data.get("layer_range")
        if not isinstance(layer_range_raw, list) or len(layer_range_raw) != 2:
            raise ChainRpcMalformedError(
                f"layer_range must be [start, end] list, got {layer_range_raw!r}"
            )
        shape_raw = data.get("activation_shape")
        if not isinstance(shape_raw, list):
            raise ChainRpcMalformedError(
                f"activation_shape must be list, got {type(shape_raw).__name__}"
            )
        # activation_blob_hex is REQUIRED but MAY be empty (streamed
        # mode carries the bytes out-of-band as ActivationChunk frames).
        blob_hex = data.get("activation_blob_hex")
        if not isinstance(blob_hex, str):
            raise ChainRpcMalformedError(
                "activation_blob_hex must be string"
            )
        try:
            blob_bytes = bytes.fromhex(blob_hex)
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"activation_blob_hex is not valid hex: {exc}"
            ) from exc
        try:
            privacy = PrivacyLevel(_required_str(data, "privacy_tier"))
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"privacy_tier invalid: {exc}"
            ) from exc
        try:
            content = ContentTier(_required_str(data, "content_tier"))
        except ValueError as exc:
            raise ChainRpcMalformedError(
                f"content_tier invalid: {exc}"
            ) from exc
        manifest_raw = data.get("activation_manifest")
        manifest: Optional[ShardManifest] = None
        if manifest_raw is not None:
            manifest = _shard_manifest_from_dict(manifest_raw)
        # streaming flag (Phase 3.x.8): default False when absent
        # (preserves byte-equivalence with v2-pre-3.x.8 messages).
        # When present, MUST be a real bool — int 0/1 rejected to
        # preserve M1-style coercion-safety invariant.
        streaming_raw = data.get("streaming", False)
        if not isinstance(streaming_raw, bool):
            raise ChainRpcMalformedError(
                f"streaming must be bool, got "
                f"{type(streaming_raw).__name__}"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            model_id=_required_str(data, "model_id"),
            layer_range=(int(layer_range_raw[0]), int(layer_range_raw[1])),
            privacy_tier=privacy,
            content_tier=content,
            activation_blob=blob_bytes,
            activation_shape=tuple(int(d) for d in shape_raw),
            activation_dtype=_required_str(data, "activation_dtype"),
            upstream_token=HandoffToken.from_dict(token_raw),
            deadline_unix=_required_number(data, "deadline_unix"),
            activation_manifest=manifest,
            streaming=streaming_raw,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class RunLayerSliceResponse:
    """Stage → executor on success."""

    request_id: str
    activation_blob: bytes
    activation_shape: Tuple[int, ...]
    activation_dtype: str
    duration_seconds: float
    tee_attestation: bytes
    tee_type: TEEType
    epsilon_spent: float
    stage_signature_b64: str
    stage_node_id: str
    # v2 streaming: when present, ``activation_blob`` MUST be empty
    # and the bytes ride out-of-band as ActivationChunk frames.
    activation_manifest: Optional[ShardManifest] = None
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("activation_dtype", self.activation_dtype)
        _validate_str_field("stage_signature_b64", self.stage_signature_b64)
        _validate_str_field("stage_node_id", self.stage_node_id)
        _validate_version(self.protocol_version)
        if not isinstance(self.activation_blob, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"activation_blob must be bytes, got "
                f"{type(self.activation_blob).__name__}"
            )
        if not isinstance(self.activation_shape, tuple):
            raise ChainRpcMalformedError(
                f"activation_shape must be tuple, got "
                f"{type(self.activation_shape).__name__}"
            )
        for dim in self.activation_shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ChainRpcMalformedError(
                    f"activation_shape entries must be positive int, got "
                    f"{dim!r}"
                )
        if not isinstance(self.duration_seconds, (int, float)) or self.duration_seconds < 0:
            raise ChainRpcMalformedError(
                f"duration_seconds must be non-negative numeric, got "
                f"{self.duration_seconds!r}"
            )
        if not isinstance(self.tee_attestation, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"tee_attestation must be bytes, got "
                f"{type(self.tee_attestation).__name__}"
            )
        if not isinstance(self.tee_type, TEEType):
            raise ChainRpcMalformedError(
                f"tee_type must be TEEType, got "
                f"{type(self.tee_type).__name__}"
            )
        if not isinstance(self.epsilon_spent, (int, float)):
            raise ChainRpcMalformedError(
                f"epsilon_spent must be numeric, got "
                f"{type(self.epsilon_spent).__name__}"
            )
        # v2 inline-XOR-streamed integrity check (M3 round-1 mirror).
        # Exactly one payload path active. ``activation_shape`` entries
        # are positive ints, so any well-formed inline response carries
        # at least dtype.itemsize bytes — empty + no manifest = no
        # payload.
        if self.activation_manifest is not None:
            _validate_shard_manifest(
                self.activation_manifest, field="activation_manifest"
            )
            if self.activation_blob:
                raise ChainRpcMalformedError(
                    "streamed mode: activation_blob must be empty when "
                    "activation_manifest is present"
                )
        elif not self.activation_blob:
            raise ChainRpcMalformedError(
                "response has neither inline activation_blob nor "
                "activation_manifest — exactly one payload path is "
                "required"
            )

    @staticmethod
    def signing_payload(
        request_id: str,
        activation_blob: bytes,
        activation_shape: Tuple[int, ...],
        activation_dtype: str,
        duration_seconds: float,
        tee_attestation: bytes,
        tee_type: TEEType,
        epsilon_spent: float,
        stage_node_id: str,
        activation_manifest: Optional[ShardManifest] = None,
    ) -> bytes:
        """Canonical bytes the stage signs over the response.

        The signature commits the stage to its activation output bytes
        + attestation + claimed timing. A malicious downstream that
        tampered with the response (e.g. swapped attestation) would
        invalidate the signature; the executor catches this at
        anchor-verify time and surfaces it as Phase 7.1 challenge
        material.

        v1↔v2 byte-equivalence for inline messages: when
        ``activation_manifest`` is None, the field is OMITTED from the
        canonical JSON entirely — so a v2 caller signing inline gets
        the same payload bytes a v1 caller would. Only streamed
        responses carry the additional ``activation_manifest_envelope``
        key, which commits the stage to ALL FIVE manifest fields
        (shard_id + payload_sha256 + payload_bytes + total_chunks +
        chunk_bytes). Without committing the envelope's shape fields,
        a network-level relay could inflate ``payload_bytes`` /
        ``total_chunks`` while leaving ``payload_sha256`` intact and
        trigger a client-side memory DoS during reassembly — the
        H2 round-1 finding from Phase 3.x.7.1 Task 6 review.
        """
        payload = {
            "request_id": request_id,
            "activation_blob_hex": bytes(activation_blob).hex(),
            "activation_shape": list(activation_shape),
            "activation_dtype": activation_dtype,
            "duration_seconds": float(duration_seconds),
            "tee_attestation_hex": bytes(tee_attestation).hex(),
            "tee_type": tee_type.value,
            "epsilon_spent": float(epsilon_spent),
            "stage_node_id": stage_node_id,
        }
        # Conditional manifest envelope encoding: omitted entirely when
        # the response is inline (so v1↔v2 inline messages produce
        # byte-identical canonical JSON). Streamed responses commit
        # to ALL five manifest fields — tampering ANY of them
        # invalidates the signature.
        if activation_manifest is not None:
            payload["activation_manifest_envelope"] = {
                "shard_id": activation_manifest.shard_id,
                "payload_sha256": activation_manifest.payload_sha256,
                "payload_bytes": activation_manifest.payload_bytes,
                "total_chunks": activation_manifest.total_chunks,
                "chunk_bytes": activation_manifest.chunk_bytes,
            }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def sign(
        cls,
        identity: NodeIdentity,
        request_id: str,
        activation_blob: bytes,
        activation_shape: Tuple[int, ...],
        activation_dtype: str,
        duration_seconds: float,
        tee_attestation: bytes,
        tee_type: TEEType,
        epsilon_spent: float,
        activation_manifest: Optional[ShardManifest] = None,
    ) -> "RunLayerSliceResponse":
        """Construct + sign a fresh response under the stage ``identity``.

        For streamed responses, pass ``activation_manifest=manifest`` and
        ``activation_blob=b""``. The manifest's ``payload_sha256``
        commits the stage to the to-be-assembled bytes via the
        signing payload."""
        payload = cls.signing_payload(
            request_id,
            activation_blob,
            activation_shape,
            activation_dtype,
            duration_seconds,
            tee_attestation,
            tee_type,
            epsilon_spent,
            identity.node_id,
            activation_manifest=activation_manifest,
        )
        sig = identity.sign(payload)
        return cls(
            request_id=request_id,
            activation_blob=bytes(activation_blob),
            activation_shape=tuple(activation_shape),
            activation_dtype=activation_dtype,
            duration_seconds=float(duration_seconds),
            tee_attestation=bytes(tee_attestation),
            tee_type=tee_type,
            epsilon_spent=float(epsilon_spent),
            stage_signature_b64=sig,
            stage_node_id=identity.node_id,
            activation_manifest=activation_manifest,
        )

    def verify_with_anchor(
        self, anchor: Any, *, expected_stage_node_id: str
    ) -> bool:
        """Verify this response's stage signature against the EXPECTED
        stage's pubkey resolved from the on-chain anchor.

        ``expected_stage_node_id`` MUST be the node_id the caller
        dispatched the request to — typically the GPUChain stage at
        the current chain index. The check rejects when:

          - the response's claimed ``stage_node_id`` doesn't match
            ``expected_stage_node_id`` (defends against an attacker
            with ANY anchor-registered identity replacing the legit
            stage's response with their own genuine signature; without
            this check, the lookup-by-self-claim path would accept
            Mallory's response under Mallory's pubkey because Mallory
            IS anchor-registered)

          - the resolved pubkey doesn't verify the stage's signature

        The Phase 3.x.5 SignedManifestEntry / Phase 3.x.6
        SignedProfileEntry pattern is functionally equivalent: the
        identity to look up is supplied EXTERNALLY by the caller, not
        pulled from the response. RunLayerSliceResponse takes the
        externally-supplied identity as a kwarg to make the contract
        impossible to misuse.

        Returns False on any failure path; raises only on transient
        anchor-RPC infrastructure errors.
        """
        if anchor is None or not hasattr(anchor, "lookup"):
            return False
        if not isinstance(expected_stage_node_id, str) or not expected_stage_node_id:
            return False
        # Anchor-verify against the EXPECTED identity, not the
        # response's self-declared one. If a malicious peer signed
        # with their own (also anchor-registered) key, the lookup
        # below uses the expected node's pubkey, so the bad signature
        # fails verification.
        if self.stage_node_id != expected_stage_node_id:
            return False
        stage_pubkey_b64 = anchor.lookup(expected_stage_node_id)
        if not stage_pubkey_b64:
            return False
        payload = self.signing_payload(
            self.request_id,
            self.activation_blob,
            self.activation_shape,
            self.activation_dtype,
            self.duration_seconds,
            self.tee_attestation,
            self.tee_type,
            self.epsilon_spent,
            self.stage_node_id,
            activation_manifest=self.activation_manifest,
        )
        return verify_signature(
            stage_pubkey_b64, payload, self.stage_signature_b64
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "activation_blob_hex": bytes(self.activation_blob).hex(),
            "activation_shape": list(self.activation_shape),
            "activation_dtype": self.activation_dtype,
            "duration_seconds": self.duration_seconds,
            "tee_attestation_hex": bytes(self.tee_attestation).hex(),
            "tee_type": self.tee_type.value,
            "epsilon_spent": self.epsilon_spent,
            "stage_signature_b64": self.stage_signature_b64,
            "stage_node_id": self.stage_node_id,
        }
        if self.activation_manifest is not None:
            out["activation_manifest"] = _shard_manifest_to_dict(
                self.activation_manifest
            )
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunLayerSliceResponse":
        _expect_type(data, ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE)
        shape_raw = data.get("activation_shape")
        if not isinstance(shape_raw, list):
            raise ChainRpcMalformedError(
                f"activation_shape must be list, got {type(shape_raw).__name__}"
            )
        # activation_blob_hex is REQUIRED but MAY be empty (streamed
        # mode carries the bytes out-of-band).
        blob_hex_raw = data.get("activation_blob_hex")
        if not isinstance(blob_hex_raw, str):
            raise ChainRpcMalformedError(
                "activation_blob_hex must be string"
            )
        try:
            blob_bytes = bytes.fromhex(blob_hex_raw)
            attest_bytes = bytes.fromhex(
                _required_str(data, "tee_attestation_hex")
            )
        except ValueError as exc:
            raise ChainRpcMalformedError(f"hex decode failed: {exc}") from exc
        try:
            tee_type = TEEType(_required_str(data, "tee_type"))
        except ValueError as exc:
            raise ChainRpcMalformedError(f"tee_type invalid: {exc}") from exc
        manifest_raw = data.get("activation_manifest")
        manifest: Optional[ShardManifest] = None
        if manifest_raw is not None:
            manifest = _shard_manifest_from_dict(manifest_raw)
        return cls(
            request_id=_required_str(data, "request_id"),
            activation_blob=blob_bytes,
            activation_shape=tuple(int(d) for d in shape_raw),
            activation_dtype=_required_str(data, "activation_dtype"),
            duration_seconds=_required_number(data, "duration_seconds"),
            tee_attestation=attest_bytes,
            tee_type=tee_type,
            epsilon_spent=_required_number(data, "epsilon_spent"),
            stage_signature_b64=_required_str(data, "stage_signature_b64"),
            stage_node_id=_required_str(data, "stage_node_id"),
            activation_manifest=manifest,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class StageError:
    """Stage → executor on any failure. Structured enum-coded reason."""

    request_id: str
    code: str          # StageErrorCode value (str-Enum, stored as str
                       # so unknown codes from a future protocol round-
                       # trip cleanly without rejection)
    message: str
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.STAGE_ERROR.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_str_field("code", self.code)
        if not isinstance(self.message, str):
            raise ChainRpcMalformedError(
                f"message must be str, got {type(self.message).__name__}"
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
    def from_dict(cls, data: Dict[str, Any]) -> "StageError":
        _expect_type(data, ChainRpcMessageType.STAGE_ERROR)
        return cls(
            request_id=_required_str(data, "request_id"),
            code=_required_str(data, "code"),
            message=data.get("message", "") if isinstance(data.get("message"), str) else "",
            protocol_version=_required_int(data, "protocol_version"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.8 — streaming-token output wire types
# ──────────────────────────────────────────────────────────────────────────


# Permitted finish_reason values on TokenFrame / StreamFinalFrame's
# embedded response. ``None`` means "more frames coming"; a non-None
# value is set on the FINAL frame the runner emits.
_FINISH_REASONS = frozenset({"stop", "max_tokens", "cancelled", "error"})


@dataclass(frozen=True)
class TokenFrame:
    """Incremental token-output frame emitted by the tail stage during
    autoregressive decode (Phase 3.x.8).

    Multiple ``TokenFrame`` frames per stream, ordered strictly by
    ``sequence_index`` (0-indexed, no gaps). The terminal frame on a
    stream is always a ``StreamFinalFrame`` carrying the signed
    ``RunLayerSliceResponse`` over the joined output bytes.

    Bound to its parent request via ``request_id`` so a network relay
    can't splice frames from one stream into another (mirrors the
    relay-defense binding on ``ActivationChunk`` from 3.x.7.1).

    The joined ``text_delta`` across all frames in a stream MUST equal
    the UTF-8 decoding of the embedded response's ``activation_blob``
    in the terminal ``StreamFinalFrame`` — this is how the stage's
    Ed25519 signature commits to the streamed output (existing
    ``RunLayerSliceResponse.signing_payload`` hex-encodes
    ``activation_blob``; no new signing-payload field needed).
    """

    request_id: str
    sequence_index: int  # 0-indexed; strictly increasing across stream
    text_delta: str      # text emitted by this frame (may be empty)
    token_id: Optional[int] = None        # vocab id if available
    finish_reason: Optional[str] = None   # set on the LAST TokenFrame
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.TOKEN_FRAME.value

    def __post_init__(self) -> None:
        _validate_str_field("request_id", self.request_id)
        _validate_version(self.protocol_version)
        if (
            not isinstance(self.sequence_index, int)
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
        ):
            raise ChainRpcMalformedError(
                f"sequence_index must be non-negative int, got "
                f"{self.sequence_index!r}"
            )
        if not isinstance(self.text_delta, str):
            raise ChainRpcMalformedError(
                f"text_delta must be str, got "
                f"{type(self.text_delta).__name__}"
            )
        if self.token_id is not None and (
            not isinstance(self.token_id, int)
            or isinstance(self.token_id, bool)
        ):
            raise ChainRpcMalformedError(
                f"token_id must be int or None, got "
                f"{type(self.token_id).__name__}"
            )
        if self.finish_reason is not None:
            if not isinstance(self.finish_reason, str):
                raise ChainRpcMalformedError(
                    f"finish_reason must be str or None, got "
                    f"{type(self.finish_reason).__name__}"
                )
            if self.finish_reason not in _FINISH_REASONS:
                raise ChainRpcMalformedError(
                    f"finish_reason must be one of "
                    f"{sorted(_FINISH_REASONS)}, got "
                    f"{self.finish_reason!r}"
                )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "sequence_index": self.sequence_index,
            "text_delta": self.text_delta,
        }
        # Optional fields use conditional encoding so frames without
        # token_id / finish_reason don't gain spurious null keys.
        if self.token_id is not None:
            out["token_id"] = self.token_id
        if self.finish_reason is not None:
            out["finish_reason"] = self.finish_reason
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenFrame":
        _expect_type(data, ChainRpcMessageType.TOKEN_FRAME)
        token_id_raw = data.get("token_id")
        if token_id_raw is not None and (
            not isinstance(token_id_raw, int)
            or isinstance(token_id_raw, bool)
        ):
            raise ChainRpcMalformedError(
                f"token_id must be int or null, got "
                f"{type(token_id_raw).__name__}"
            )
        finish_raw = data.get("finish_reason")
        if finish_raw is not None and not isinstance(finish_raw, str):
            raise ChainRpcMalformedError(
                f"finish_reason must be str or null, got "
                f"{type(finish_raw).__name__}"
            )
        text_raw = data.get("text_delta")
        if not isinstance(text_raw, str):
            raise ChainRpcMalformedError(
                "text_delta must be string"
            )
        return cls(
            request_id=_required_str(data, "request_id"),
            sequence_index=_required_int(data, "sequence_index"),
            text_delta=text_raw,
            token_id=token_id_raw,
            finish_reason=finish_raw,
            protocol_version=_required_int(data, "protocol_version"),
        )


@dataclass(frozen=True)
class StreamFinalFrame:
    """Terminal frame on a streaming-token output stream
    (Phase 3.x.8).

    Carries the FULL signed ``RunLayerSliceResponse`` whose
    ``activation_blob`` is the UTF-8-encoded joined output text
    across all preceding ``TokenFrame.text_delta`` values. The
    embedded response's stage signature commits to this joined text
    via the existing ``signing_payload`` (hex-encodes
    ``activation_blob``) — so a relay that tampers any TokenFrame's
    ``text_delta`` causes the joined-bytes hash to diverge from
    what the stage signed, invalidating the stream as a whole.

    The ``activation_shape`` on the embedded response describes the
    joined-text bytes (always 1-D byte tensor of length
    ``len(joined_utf8)``); ``activation_dtype`` is ``"uint8"``.
    """

    response: RunLayerSliceResponse
    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: str = ChainRpcMessageType.STREAM_FINAL_FRAME.value

    def __post_init__(self) -> None:
        _validate_version(self.protocol_version)
        if not isinstance(self.response, RunLayerSliceResponse):
            raise ChainRpcMalformedError(
                f"response must be RunLayerSliceResponse, got "
                f"{type(self.response).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "response": self.response.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamFinalFrame":
        _expect_type(data, ChainRpcMessageType.STREAM_FINAL_FRAME)
        response_raw = data.get("response")
        if not isinstance(response_raw, dict):
            raise ChainRpcMalformedError(
                f"response must be dict, got "
                f"{type(response_raw).__name__}"
            )
        return cls(
            response=RunLayerSliceResponse.from_dict(response_raw),
            protocol_version=_required_int(data, "protocol_version"),
        )


# ──────────────────────────────────────────────────────────────────────────
# Codec
# ──────────────────────────────────────────────────────────────────────────


_MESSAGE_TYPE_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    ChainRpcMessageType.RUN_LAYER_SLICE_REQUEST.value: RunLayerSliceRequest.from_dict,
    ChainRpcMessageType.RUN_LAYER_SLICE_RESPONSE.value: RunLayerSliceResponse.from_dict,
    ChainRpcMessageType.STAGE_ERROR.value: StageError.from_dict,
    ChainRpcMessageType.ACTIVATION_CHUNK.value: ActivationChunk.from_dict,
    ChainRpcMessageType.TOKEN_FRAME.value: TokenFrame.from_dict,
    ChainRpcMessageType.STREAM_FINAL_FRAME.value: StreamFinalFrame.from_dict,
}


def parse_message(payload: bytes) -> Any:
    """Decode wire-format bytes into the matching dataclass.

    Mirrors the manifest-DHT and profile-DHT parsers. Size cap fires
    BEFORE ``json.loads`` allocates; version mismatch raises a
    dedicated exception so callers can distinguish from malformed
    input.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise ChainRpcMalformedError(
            f"payload must be bytes, got {type(payload).__name__}"
        )
    if len(payload) > MAX_HANDSHAKE_BYTES:
        raise ChainRpcMalformedError(
            f"payload exceeds MAX_HANDSHAKE_BYTES "
            f"({len(payload)} > {MAX_HANDSHAKE_BYTES})"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ChainRpcMalformedError(f"JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ChainRpcMalformedError(
            f"top-level message must be dict, got {type(data).__name__}"
        )
    msg_type = data.get("type")
    if not isinstance(msg_type, str):
        raise ChainRpcMalformedError(
            f"missing or non-string 'type' field; got {msg_type!r}"
        )
    constructor = _MESSAGE_TYPE_REGISTRY.get(msg_type)
    if constructor is None:
        raise ChainRpcUnknownTypeError(
            f"unknown message type {msg_type!r}; "
            f"known: {sorted(_MESSAGE_TYPE_REGISTRY)}"
        )
    # v2 nodes accept v1 + v2 (forward-compat for rolling deploys);
    # v1 nodes still reject anything other than 1 because they only
    # know about CHAIN_RPC_PROTOCOL_VERSION = 1. bool rejection
    # preserved (M1 round-1 fix from Phase 3.x.7 Task 8).
    #
    # L1 round-1 (3.x.7.1): missing or non-int protocol_version is a
    # version-negotiation failure, not a malformed-message failure —
    # surface it as ``ChainRpcVersionMismatchError`` so the executor
    # maps it to ``UNSUPPORTED_VERSION`` instead of
    # ``MALFORMED_RESPONSE``.
    if "protocol_version" not in data:
        raise ChainRpcVersionMismatchError(
            "peer message missing 'protocol_version'; "
            f"local SUPPORTED_PROTOCOL_VERSIONS="
            f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
        )
    version = data["protocol_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ChainRpcVersionMismatchError(
            f"peer protocol_version must be int, got "
            f"{type(version).__name__} ({version!r}); "
            f"local SUPPORTED_PROTOCOL_VERSIONS="
            f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
        )
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ChainRpcVersionMismatchError(
            f"peer protocol_version={version}; "
            f"local SUPPORTED_PROTOCOL_VERSIONS="
            f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}"
        )
    return constructor(data)


def encode_message(message: Any) -> bytes:
    """Encode a wire-format dataclass to canonical JSON bytes."""
    if not hasattr(message, "to_dict"):
        raise ChainRpcMalformedError(
            f"message must have to_dict(); got {type(message).__name__}"
        )
    return json.dumps(message.to_dict(), sort_keys=True).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Validation helpers (private)
# ──────────────────────────────────────────────────────────────────────────


def _validate_str_field(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ChainRpcMalformedError(
            f"{name} must be str, got {type(value).__name__}"
        )
    if not value:
        raise ChainRpcMalformedError(f"{name} must be non-empty")


def _validate_version(version: Any) -> None:
    # bool is a subclass of int in Python; reject explicitly so a
    # peer sending {"protocol_version": true} doesn't slip through
    # via True == 1.
    if not isinstance(version, int) or isinstance(version, bool):
        raise ChainRpcMalformedError(
            f"protocol_version must be int, got {type(version).__name__}"
        )


def _expect_type(
    data: Dict[str, Any], expected: ChainRpcMessageType
) -> None:
    actual = data.get("type")
    if actual != expected.value:
        raise ChainRpcMalformedError(
            f"expected type={expected.value!r}, got {actual!r}"
        )


def _required_str(data: Dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value:
        raise ChainRpcMalformedError(
            f"{field_name} must be non-empty str"
        )
    return value


def _required_int(data: Dict[str, Any], field_name: str) -> int:
    value = data.get(field_name)
    # Reject bool explicitly; isinstance(True, int) is True in Python
    # so {"chain_stage_index": true} would otherwise produce a token
    # whose int field is True, polluting downstream telemetry +
    # equality semantics.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChainRpcMalformedError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    return value


def _required_number(data: Dict[str, Any], field_name: str) -> float:
    value = data.get(field_name)
    if not isinstance(value, (int, float)):
        raise ChainRpcMalformedError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    return float(value)
