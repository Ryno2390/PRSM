"""B3.1b — AggregateRequest / AggregateResponse / SignedPartial.

Wire-format dataclasses for the QueryOrchestrator aggregator-handoff
RPC. Per `docs/2026-05-08-aggregate-rpc-design.md`. Pattern-lift from
`prsm/compute/chain_rpc/protocol.py::RunLayerSliceRequest` (Phase
3.x.7) — same `to_dict`/`from_dict` shape, same signing-payload prefix
discipline, same canonical-encoding contract.

Why a separate module: protocol.py is already 2.6K LoC of layer-slice
RPC. Aggregate RPC is a separate product surface; mixing them sprawls
both. The 2 new `ChainRpcMessageType` enum values + 3 new
`StageErrorCode` values DO live in protocol.py (additive enum
extension) so server routing can dispatch on the type field — but
the dataclasses themselves live here.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Tuple

from prsm.compute.chain_rpc.protocol import (
    CHAIN_RPC_PROTOCOL_VERSION,
    ChainRpcMalformedError,
    ChainRpcMessageType,
    _expect_type,
    _required_int,
    _required_number,
    _required_str,
    _validate_str_field,
    _validate_version,
)
from prsm.compute.query_orchestrator.aggregator_selector import (
    AggregationCommit,
)


def _b64(data: bytes) -> str:
    """Base64 encode for canonical wire JSON."""
    return base64.b64encode(data).decode("ascii")


def _b64d(s: str, *, field_name: str) -> bytes:
    """Base64 decode + reject non-string / malformed input."""
    if not isinstance(s, str):
        raise ChainRpcMalformedError(
            f"{field_name} must be base64 str, got {type(s).__name__}"
        )
    try:
        return base64.b64decode(s, validate=True)
    except Exception as exc:
        raise ChainRpcMalformedError(
            f"{field_name} base64 decode failed: {exc}"
        ) from exc


def _validate_fixed_bytes(name: str, value: Any, n: int) -> None:
    if not isinstance(value, (bytes, bytearray)):
        raise ChainRpcMalformedError(
            f"{name} must be bytes, got {type(value).__name__}"
        )
    if len(value) != n:
        raise ChainRpcMalformedError(
            f"{name} must be exactly {n} bytes, got {len(value)}"
        )


# ──────────────────────────────────────────────────────────────────────
# SignedPartial
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignedPartial:
    """Per-shard agent's signed output, addressed to the aggregator.

    The source agent applies DP noise via `prsm/compute/tee/dp_noise.py`,
    sets ``dp_noise_applied=True``, signs the bundle with their Ed25519
    key, and ships. The aggregator's server-side handler verifies the
    signature + the marker before consuming.

    Attributes
    ----------
    shard_cid:
        Which shard this partial was computed against.
    payload:
        DP-noised partial. Aggregator combines without seeing raw
        shard data (A5).
    creator_id:
        Original publisher — flows to RoyaltyDistributor at settlement.
    dp_noise_applied:
        A5 marker. Server refuses-and-raises
        ``StageErrorCode.DP_NOISE_MARKER_MISSING`` when False.
    source_agent_pubkey:
        32-byte Ed25519 pubkey. Server verifies
        ``source_agent_signature`` against this.
    source_agent_signature:
        64-byte Ed25519 signature over the partial's canonical bytes.
    privacy_budget_consumed:
        Epsilon spent on this partial. Aggregator sums these
        (post-Laplace composition) and trips
        ``StageErrorCode.PRIVACY_BUDGET_EXHAUSTED`` if combination
        would exceed the manifest's per-query ceiling.
    """

    shard_cid: str
    payload: bytes
    creator_id: str
    dp_noise_applied: bool
    source_agent_pubkey: bytes
    source_agent_signature: bytes
    privacy_budget_consumed: float

    def __post_init__(self) -> None:
        _validate_str_field("shard_cid", self.shard_cid)
        _validate_str_field("creator_id", self.creator_id)
        if not isinstance(self.payload, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"payload must be bytes, got {type(self.payload).__name__}"
            )
        if not isinstance(self.dp_noise_applied, bool):
            raise ChainRpcMalformedError(
                f"dp_noise_applied must be bool, got "
                f"{type(self.dp_noise_applied).__name__}"
            )
        _validate_fixed_bytes("source_agent_pubkey", self.source_agent_pubkey, 32)
        _validate_fixed_bytes(
            "source_agent_signature", self.source_agent_signature, 64,
        )
        if not isinstance(self.privacy_budget_consumed, (int, float)):
            raise ChainRpcMalformedError(
                f"privacy_budget_consumed must be numeric, got "
                f"{type(self.privacy_budget_consumed).__name__}"
            )
        if float(self.privacy_budget_consumed) < 0.0:
            raise ChainRpcMalformedError(
                f"privacy_budget_consumed must be non-negative, got "
                f"{self.privacy_budget_consumed}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_cid": self.shard_cid,
            "payload": _b64(bytes(self.payload)),
            "creator_id": self.creator_id,
            "dp_noise_applied": self.dp_noise_applied,
            "source_agent_pubkey": _b64(bytes(self.source_agent_pubkey)),
            "source_agent_signature": _b64(bytes(self.source_agent_signature)),
            "privacy_budget_consumed": float(self.privacy_budget_consumed),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignedPartial":
        return cls(
            shard_cid=_required_str(data, "shard_cid"),
            payload=_b64d(data.get("payload", ""), field_name="payload"),
            creator_id=_required_str(data, "creator_id"),
            dp_noise_applied=bool(data.get("dp_noise_applied", False)),
            source_agent_pubkey=_b64d(
                data.get("source_agent_pubkey", ""),
                field_name="source_agent_pubkey",
            ),
            source_agent_signature=_b64d(
                data.get("source_agent_signature", ""),
                field_name="source_agent_signature",
            ),
            privacy_budget_consumed=_required_number(
                data, "privacy_budget_consumed",
            ),
        )

    def canonical_digest(self) -> bytes:
        """SHA-256 over the canonical to_dict-serialized form. Used by
        `AggregateRequest.signing_payload` to commit to the partial
        without copying the full payload bytes into the request's
        signing payload."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).digest()


# ──────────────────────────────────────────────────────────────────────
# AggregateRequest
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AggregateRequest:
    """Prompter → selected aggregator. Carries per-shard partials,
    routing context for the response, and the prompter's signature
    over the full bundle.

    All 32-byte fields are length-validated at construction time —
    short fields raise ``ChainRpcMalformedError`` rather than
    producing a payload that verifies inconsistently.

    See `docs/2026-05-08-aggregate-rpc-design.md` §"AggregateRequest".
    """

    request_id: bytes                  # 32 bytes — unique per (query, attempt)
    query_id: bytes                    # 32 bytes — A9 binding
    manifest_json: str                 # InstructionManifest.to_json()
    partials: Tuple[SignedPartial, ...]
    prompter_pubkey: bytes             # 32 bytes Ed25519
    prompter_node_id: str              # routing only; A2 input
    beacon_used: bytes                 # 32 bytes — A6 forensic anchor
    aggregator_pubkey_hash: bytes      # 32 bytes — A8 identity binding
    ftns_budget: int
    deadline_unix: int
    prompter_signature: bytes          # 64 bytes Ed25519 over signing_payload()

    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: ClassVar[str] = ChainRpcMessageType.AGGREGATE_REQUEST.value
    SIGNING_PREFIX: ClassVar[bytes] = b"prsm:aggregate:v1\n"

    def __post_init__(self) -> None:
        _validate_fixed_bytes("request_id", self.request_id, 32)
        _validate_fixed_bytes("query_id", self.query_id, 32)
        _validate_str_field("manifest_json", self.manifest_json)
        if not isinstance(self.partials, tuple):
            raise ChainRpcMalformedError(
                f"partials must be a tuple, got {type(self.partials).__name__}"
            )
        for i, p in enumerate(self.partials):
            if not isinstance(p, SignedPartial):
                raise ChainRpcMalformedError(
                    f"partials[{i}] must be SignedPartial, got "
                    f"{type(p).__name__}"
                )
        _validate_fixed_bytes("prompter_pubkey", self.prompter_pubkey, 32)
        _validate_str_field("prompter_node_id", self.prompter_node_id)
        _validate_fixed_bytes("beacon_used", self.beacon_used, 32)
        _validate_fixed_bytes(
            "aggregator_pubkey_hash", self.aggregator_pubkey_hash, 32,
        )
        if (
            not isinstance(self.ftns_budget, int)
            or isinstance(self.ftns_budget, bool)
            or self.ftns_budget < 0
        ):
            raise ChainRpcMalformedError(
                f"ftns_budget must be non-negative int, got "
                f"{self.ftns_budget!r}"
            )
        if (
            not isinstance(self.deadline_unix, int)
            or isinstance(self.deadline_unix, bool)
            or self.deadline_unix < 0
        ):
            raise ChainRpcMalformedError(
                f"deadline_unix must be non-negative int, got "
                f"{self.deadline_unix!r}"
            )
        _validate_fixed_bytes(
            "prompter_signature", self.prompter_signature, 64,
        )
        _validate_version(self.protocol_version)

    def signing_payload(self) -> bytes:
        """Canonical bytes the prompter signs.

        Layout:
            SIGNING_PREFIX                  (18 bytes "prsm:aggregate:v1\\n")
            protocol_version                (uvarint)
            request_id                      (32 bytes)
            query_id                        (32 bytes)
            sha256(manifest_json)           (32 bytes)
            len(partials)                   (uvarint)
            partial[i].canonical_digest()   (32 bytes each, in order)
            prompter_pubkey                 (32 bytes)
            len(prompter_node_id) + utf8    (uvarint + bytes)
            beacon_used                     (32 bytes)
            aggregator_pubkey_hash          (32 bytes)
            ftns_budget                     (uvarint)
            deadline_unix                   (uvarint)

        Renaming the prefix or reordering fields invalidates every
        previously-signed request.
        """
        out = bytearray(self.SIGNING_PREFIX)
        out += _uvarint(self.protocol_version)
        out += self.request_id
        out += self.query_id
        out += hashlib.sha256(self.manifest_json.encode("utf-8")).digest()
        out += _uvarint(len(self.partials))
        for p in self.partials:
            out += p.canonical_digest()
        out += self.prompter_pubkey
        nid_bytes = self.prompter_node_id.encode("utf-8")
        out += _uvarint(len(nid_bytes))
        out += nid_bytes
        out += self.beacon_used
        out += self.aggregator_pubkey_hash
        out += _uvarint(self.ftns_budget)
        out += _uvarint(self.deadline_unix)
        return bytes(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": _b64(bytes(self.request_id)),
            "query_id": _b64(bytes(self.query_id)),
            "manifest_json": self.manifest_json,
            "partials": [p.to_dict() for p in self.partials],
            "prompter_pubkey": _b64(bytes(self.prompter_pubkey)),
            "prompter_node_id": self.prompter_node_id,
            "beacon_used": _b64(bytes(self.beacon_used)),
            "aggregator_pubkey_hash": _b64(bytes(self.aggregator_pubkey_hash)),
            "ftns_budget": self.ftns_budget,
            "deadline_unix": self.deadline_unix,
            "prompter_signature": _b64(bytes(self.prompter_signature)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregateRequest":
        _expect_type(data, ChainRpcMessageType.AGGREGATE_REQUEST)
        version = _required_int(data, "protocol_version")
        if version != CHAIN_RPC_PROTOCOL_VERSION:
            raise ChainRpcMalformedError(
                f"protocol_version mismatch: expected "
                f"{CHAIN_RPC_PROTOCOL_VERSION}, got {version}"
            )
        partials_raw = data.get("partials", [])
        if not isinstance(partials_raw, list):
            raise ChainRpcMalformedError(
                f"partials must be list, got {type(partials_raw).__name__}"
            )
        return cls(
            request_id=_b64d(data.get("request_id", ""), field_name="request_id"),
            query_id=_b64d(data.get("query_id", ""), field_name="query_id"),
            manifest_json=_required_str(data, "manifest_json"),
            partials=tuple(SignedPartial.from_dict(p) for p in partials_raw),
            prompter_pubkey=_b64d(
                data.get("prompter_pubkey", ""), field_name="prompter_pubkey",
            ),
            prompter_node_id=_required_str(data, "prompter_node_id"),
            beacon_used=_b64d(data.get("beacon_used", ""), field_name="beacon_used"),
            aggregator_pubkey_hash=_b64d(
                data.get("aggregator_pubkey_hash", ""),
                field_name="aggregator_pubkey_hash",
            ),
            ftns_budget=_required_int(data, "ftns_budget"),
            deadline_unix=_required_int(data, "deadline_unix"),
            prompter_signature=_b64d(
                data.get("prompter_signature", ""),
                field_name="prompter_signature",
            ),
            protocol_version=version,
        )


# ──────────────────────────────────────────────────────────────────────
# AggregateResponse
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AggregateResponse:
    """Aggregator → prompter. Carries the pre-commit (signed BEFORE
    plaintext, A9) and the encrypted plaintext combined output.

    The orchestrator's `swarm_runner.verify_aggregation_commit` checks
    that ``sha256(plaintext_after_decrypt) == commit.result_digest``;
    mismatch slashes the aggregator.

    See `docs/2026-05-08-aggregate-rpc-design.md` §"AggregateResponse".
    """

    request_id: bytes               # 32 bytes — echoed
    query_id: bytes                 # 32 bytes — echoed
    commit: AggregationCommit
    commit_signature: bytes         # 64 bytes Ed25519 over commit.signing_payload()
    encrypted_plaintext: bytes      # x25519+chacha20-poly1305 to prompter pubkey
    nonce: bytes                    # 24 bytes
    aggregator_pubkey: bytes        # 32 bytes Ed25519 — verifier resolves to pubkey_hash
    privacy_budget_consumed: float  # post-combination total
    contributing_creators: Tuple[str, ...]
    completed_unix: int

    protocol_version: int = CHAIN_RPC_PROTOCOL_VERSION

    MESSAGE_TYPE: ClassVar[str] = ChainRpcMessageType.AGGREGATE_RESPONSE.value

    def __post_init__(self) -> None:
        _validate_fixed_bytes("request_id", self.request_id, 32)
        _validate_fixed_bytes("query_id", self.query_id, 32)
        if not isinstance(self.commit, AggregationCommit):
            raise ChainRpcMalformedError(
                f"commit must be AggregationCommit, got "
                f"{type(self.commit).__name__}"
            )
        _validate_fixed_bytes("commit_signature", self.commit_signature, 64)
        if not isinstance(self.encrypted_plaintext, (bytes, bytearray)):
            raise ChainRpcMalformedError(
                f"encrypted_plaintext must be bytes, got "
                f"{type(self.encrypted_plaintext).__name__}"
            )
        _validate_fixed_bytes("nonce", self.nonce, 24)
        _validate_fixed_bytes("aggregator_pubkey", self.aggregator_pubkey, 32)
        if not isinstance(self.privacy_budget_consumed, (int, float)):
            raise ChainRpcMalformedError(
                f"privacy_budget_consumed must be numeric, got "
                f"{type(self.privacy_budget_consumed).__name__}"
            )
        if not isinstance(self.contributing_creators, tuple):
            raise ChainRpcMalformedError(
                f"contributing_creators must be tuple, got "
                f"{type(self.contributing_creators).__name__}"
            )
        for i, c in enumerate(self.contributing_creators):
            if not isinstance(c, str):
                raise ChainRpcMalformedError(
                    f"contributing_creators[{i}] must be str, got "
                    f"{type(c).__name__}"
                )
        if (
            not isinstance(self.completed_unix, int)
            or isinstance(self.completed_unix, bool)
            or self.completed_unix < 0
        ):
            raise ChainRpcMalformedError(
                f"completed_unix must be non-negative int, got "
                f"{self.completed_unix!r}"
            )
        _validate_version(self.protocol_version)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MESSAGE_TYPE,
            "protocol_version": self.protocol_version,
            "request_id": _b64(bytes(self.request_id)),
            "query_id": _b64(bytes(self.query_id)),
            "commit": {
                "query_id": _b64(bytes(self.commit.query_id)),
                "aggregator_pubkey_hash": _b64(
                    bytes(self.commit.aggregator_pubkey_hash),
                ),
                "result_digest": _b64(bytes(self.commit.result_digest)),
            },
            "commit_signature": _b64(bytes(self.commit_signature)),
            "encrypted_plaintext": _b64(bytes(self.encrypted_plaintext)),
            "nonce": _b64(bytes(self.nonce)),
            "aggregator_pubkey": _b64(bytes(self.aggregator_pubkey)),
            "privacy_budget_consumed": float(self.privacy_budget_consumed),
            "contributing_creators": list(self.contributing_creators),
            "completed_unix": self.completed_unix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregateResponse":
        _expect_type(data, ChainRpcMessageType.AGGREGATE_RESPONSE)
        version = _required_int(data, "protocol_version")
        if version != CHAIN_RPC_PROTOCOL_VERSION:
            raise ChainRpcMalformedError(
                f"protocol_version mismatch: expected "
                f"{CHAIN_RPC_PROTOCOL_VERSION}, got {version}"
            )
        commit_raw = data.get("commit")
        if not isinstance(commit_raw, dict):
            raise ChainRpcMalformedError(
                f"commit must be dict, got {type(commit_raw).__name__}"
            )
        commit = AggregationCommit(
            query_id=_b64d(commit_raw.get("query_id", ""), field_name="commit.query_id"),
            aggregator_pubkey_hash=_b64d(
                commit_raw.get("aggregator_pubkey_hash", ""),
                field_name="commit.aggregator_pubkey_hash",
            ),
            result_digest=_b64d(
                commit_raw.get("result_digest", ""),
                field_name="commit.result_digest",
            ),
        )
        creators_raw = data.get("contributing_creators", [])
        if not isinstance(creators_raw, list):
            raise ChainRpcMalformedError(
                f"contributing_creators must be list, got "
                f"{type(creators_raw).__name__}"
            )
        return cls(
            request_id=_b64d(data.get("request_id", ""), field_name="request_id"),
            query_id=_b64d(data.get("query_id", ""), field_name="query_id"),
            commit=commit,
            commit_signature=_b64d(
                data.get("commit_signature", ""), field_name="commit_signature",
            ),
            encrypted_plaintext=_b64d(
                data.get("encrypted_plaintext", ""), field_name="encrypted_plaintext",
            ),
            nonce=_b64d(data.get("nonce", ""), field_name="nonce"),
            aggregator_pubkey=_b64d(
                data.get("aggregator_pubkey", ""), field_name="aggregator_pubkey",
            ),
            privacy_budget_consumed=_required_number(
                data, "privacy_budget_consumed",
            ),
            contributing_creators=tuple(creators_raw),
            completed_unix=_required_int(data, "completed_unix"),
            protocol_version=version,
        )


# ──────────────────────────────────────────────────────────────────────
# uvarint helper (LEB128) — pattern-lift from existing protocol.py
# ──────────────────────────────────────────────────────────────────────


def _uvarint(n: int) -> bytes:
    """LEB128-encoded unsigned varint. Stable across machines.

    Used in signing payloads so the encoded integer width is
    canonical (no big-vs-little-endian ambiguity, no 32-vs-64-bit
    width drift).
    """
    if n < 0:
        raise ValueError(f"uvarint requires non-negative int, got {n}")
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)
