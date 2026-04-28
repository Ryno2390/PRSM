"""Phase 3.x.7 Task 5 — multi-stage TEE attestation aggregation.

Cross-host inference produces one TEE attestation per chain stage.
The Phase 3.x.1 ``InferenceReceipt`` was designed for single-host
inference: one ``tee_type`` + one ``tee_attestation`` blob. Extending
the schema directly would break backward compatibility with the
Phase 2 Ring 8 + Phase 3.x.1 callers that already produce single-stage
receipts.

This module's design:

  - Single-stage receipts continue to use ``tee_attestation`` as raw
    opaque bytes (whatever the platform vendor's attestation report
    contains). Existing callers see no change.

  - Multi-stage receipts encode a per-stage list inside the same
    ``tee_attestation`` field, prefixed with a recognizable magic
    string. Receivers detect the prefix; if absent, the bytes are
    treated as opaque single-stage attestation. If present, the
    JSON envelope is decoded and per-stage entries can be iterated.

  - The receipt's top-level ``tee_type`` is the **worst-case** TEE
    type across stages. SOFTWARE drags hardware down — one software
    stage in an SGX chain → receipt records SOFTWARE. This is the
    conservative policy: verifiers should treat the whole chain as
    only as trustworthy as its weakest link. Operators wanting
    per-stage trust can iterate ``decode_multi_stage_attestation``
    explicitly.

The encoding is intentionally backward-compatible — the
``InferenceReceipt`` dataclass needs no changes, and
``signing_payload()`` continues to hex-encode whatever bytes are in
``tee_attestation`` (so the Ed25519 signature commits the settler to
the per-stage list when the envelope is in use).

What this module does NOT do:
  - It does NOT verify per-stage attestations against platform-vendor
    services (Intel ASP, AMD KDS, etc.). That wiring is post-MVP;
    this module exposes structural validation hooks but the actual
    vendor RPC is out of scope for v1.
  - It does NOT enforce that all stages are present / contiguous;
    callers (the ``RpcChainExecutor``) own that contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from prsm.compute.tee.models import HARDWARE_TEE_TYPES, TEEType


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────


MULTI_STAGE_ATTESTATION_VERSION = 1
"""Wire-format version for the multi-stage envelope. Bump when any
field shape changes."""

MULTI_STAGE_MAGIC_PREFIX = b"PRSM-MS-ATT-V1:"
"""Recognizable byte sequence that marks the JSON envelope. Single-
stage attestations cannot start with this string by convention; if
they do (vanishingly unlikely for real platform-vendor reports), the
caller MUST add a randomization byte at the head — but no real TEE
report we've inspected starts with PRSM-prefixed ASCII."""


# ──────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────


class MultiStageAttestationError(Exception):
    """Base for multi-stage attestation envelope failures."""


class MultiStageMalformedError(MultiStageAttestationError):
    """Envelope bytes failed parse / structural validation."""


# ──────────────────────────────────────────────────────────────────────────
# Per-stage attestation record
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageAttestation:
    """One stage's contribution to the multi-stage envelope.

    Fields:
      stage_index     0-based position in the chain. Receivers can
                      detect missing / out-of-order stages.
      stage_node_id   Identity that produced the attestation. Receivers
                      use this to look up the stage's pubkey on the
                      Phase 3.x.3 anchor when verifying the per-stage
                      RunLayerSliceResponse signature in addition to
                      the platform-vendor attestation.
      tee_type        Per-stage TEE type. Aggregate worst-case across
                      stages drives the receipt's top-level tee_type.
      attestation     Raw bytes — opaque to this layer; the platform
                      vendor's attestation service interprets them.
    """

    stage_index: int
    stage_node_id: str
    tee_type: TEEType
    attestation: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.stage_index, int) or self.stage_index < 0:
            raise MultiStageMalformedError(
                f"stage_index must be non-negative int, got {self.stage_index!r}"
            )
        if not isinstance(self.stage_node_id, str) or not self.stage_node_id:
            raise MultiStageMalformedError(
                f"stage_node_id must be non-empty str, got {self.stage_node_id!r}"
            )
        if not isinstance(self.tee_type, TEEType):
            raise MultiStageMalformedError(
                f"tee_type must be TEEType, got {type(self.tee_type).__name__}"
            )
        if not isinstance(self.attestation, (bytes, bytearray)):
            raise MultiStageMalformedError(
                f"attestation must be bytes, got {type(self.attestation).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "stage_node_id": self.stage_node_id,
            "tee_type": self.tee_type.value,
            "attestation_hex": bytes(self.attestation).hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageAttestation":
        if not isinstance(data, dict):
            raise MultiStageMalformedError(
                f"stage entry must be dict, got {type(data).__name__}"
            )
        try:
            stage_index = int(data["stage_index"])
            stage_node_id = str(data["stage_node_id"])
            tee_type = TEEType(data["tee_type"])
            attestation = bytes.fromhex(data["attestation_hex"])
        except (KeyError, ValueError, TypeError) as exc:
            raise MultiStageMalformedError(
                f"stage entry parse failed: {exc}"
            ) from exc
        return cls(
            stage_index=stage_index,
            stage_node_id=stage_node_id,
            tee_type=tee_type,
            attestation=attestation,
        )


# ──────────────────────────────────────────────────────────────────────────
# Envelope codec
# ──────────────────────────────────────────────────────────────────────────


def encode_multi_stage_attestation(stages: List[StageAttestation]) -> bytes:
    """Encode a per-stage list as a magic-prefixed JSON envelope.

    Empty list raises — a chain with zero stages is a caller bug, not
    a valid receipt state. Callers with single-host inference should
    continue to populate ``tee_attestation`` with raw opaque bytes
    (no envelope) — that's the back-compat path.
    """
    if not stages:
        raise MultiStageAttestationError(
            "encode_multi_stage_attestation requires at least one stage"
        )
    payload = {
        "version": MULTI_STAGE_ATTESTATION_VERSION,
        "stages": [s.to_dict() for s in stages],
    }
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    return MULTI_STAGE_MAGIC_PREFIX + body


def is_multi_stage_attestation(blob: bytes) -> bool:
    """True iff ``blob`` carries the multi-stage magic prefix.

    Cheap O(1) check the receipt-verification layer uses to decide
    whether to invoke ``decode_multi_stage_attestation`` or treat
    ``blob`` as opaque single-stage bytes.
    """
    if not isinstance(blob, (bytes, bytearray)):
        return False
    return bytes(blob).startswith(MULTI_STAGE_MAGIC_PREFIX)


def decode_multi_stage_attestation(
    blob: bytes,
    *,
    expected_stage_count: Optional[int] = None,
) -> Optional[List[StageAttestation]]:
    """Decode a multi-stage envelope into a list of stage attestations.

    Returns ``None`` if ``blob`` doesn't carry the magic prefix —
    indicates a single-stage receipt where the bytes are raw vendor
    attestation. Callers MUST handle the ``None`` case (typically by
    treating the bytes as one opaque attestation).

    Raises ``MultiStageMalformedError`` on a magic-prefixed but
    structurally invalid envelope. The boundary is intentional:
    "no envelope" is a normal back-compat path; "envelope present
    but broken" is a corruption signal worth raising.

    Structural validation enforced:
      - ``stages`` is a non-empty list
      - ``stage_index`` values are exactly 0..N-1 (contiguous, no
        gaps)
      - no duplicate ``stage_index``
      - if ``expected_stage_count`` is given, ``len(stages)`` must
        match it (defends against a settler omitting a SOFTWARE stage
        from the envelope to upgrade the receipt's apparent worst-
        case TEE type)
    """
    if not is_multi_stage_attestation(blob):
        return None
    body = bytes(blob)[len(MULTI_STAGE_MAGIC_PREFIX):]
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MultiStageMalformedError(
            f"envelope JSON parse failed: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise MultiStageMalformedError(
            f"envelope top-level must be dict, got {type(data).__name__}"
        )
    version = data.get("version")
    if version != MULTI_STAGE_ATTESTATION_VERSION:
        raise MultiStageMalformedError(
            f"envelope version {version!r} != local "
            f"{MULTI_STAGE_ATTESTATION_VERSION}"
        )
    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise MultiStageMalformedError(
            "envelope must contain non-empty 'stages' list"
        )
    stages = [StageAttestation.from_dict(s) for s in raw_stages]

    # Structural integrity: stage_index values must be exactly
    # 0..N-1 (contiguous, no gaps, no duplicates). Otherwise a
    # settler could omit a SOFTWARE stage to upgrade the apparent
    # worst-case TEE type seen by a receipt verifier.
    indices = [s.stage_index for s in stages]
    if len(set(indices)) != len(indices):
        raise MultiStageMalformedError(
            f"envelope contains duplicate stage_index values: {sorted(indices)}"
        )
    if sorted(indices) != list(range(len(stages))):
        raise MultiStageMalformedError(
            f"envelope stage_index values must be 0..{len(stages) - 1} "
            f"(contiguous); got {sorted(indices)}"
        )
    if expected_stage_count is not None and len(stages) != expected_stage_count:
        raise MultiStageMalformedError(
            f"envelope has {len(stages)} stages; "
            f"caller expected {expected_stage_count}"
        )

    # Return stages sorted by stage_index for deterministic iteration.
    return sorted(stages, key=lambda s: s.stage_index)


# ──────────────────────────────────────────────────────────────────────────
# Worst-case TEE type
# ──────────────────────────────────────────────────────────────────────────


_WORST_CASE_RANK: Dict[TEEType, int] = {
    TEEType.SOFTWARE: 0,
}
"""Lower rank = worse confidentiality. Only SOFTWARE has a non-default
rank; all hardware TEE types share rank 1 because at this level of
granularity (chain-level worst-case) we don't differentiate among
SGX/TDX/SEV/etc.

If a stage's ``tee_type`` is not in this table, it falls through to
rank 1 (treated as hardware-equivalent). NONE-tier or unknown types
should not appear in this list because they're filtered at the tier-
gate earlier in the chain (Phase 3.x.6 ``TierGateAdapter``)."""


def worst_case_tee_type(stages: List[StageAttestation]) -> TEEType:
    """Return the worst (lowest-confidentiality) TEE type across stages.

    Empty list returns SOFTWARE (the most conservative default). When
    multiple stages share the worst rank, the FIRST stage's type wins
    by tie-break — gives callers a deterministic top-level
    ``tee_type`` for receipts.
    """
    if not stages:
        return TEEType.SOFTWARE
    worst = stages[0].tee_type
    worst_rank = _WORST_CASE_RANK.get(worst, 1)
    for stage in stages[1:]:
        rank = _WORST_CASE_RANK.get(stage.tee_type, 1)
        if rank < worst_rank:
            worst = stage.tee_type
            worst_rank = rank
    return worst


def is_hardware_tee(tee_type: TEEType) -> bool:
    """Convenience: True iff ``tee_type`` is in the hardware-backed
    set (matches Phase 3.x.6 ``TierGateAdapter`` policy)."""
    return tee_type.value in HARDWARE_TEE_TYPES


# ──────────────────────────────────────────────────────────────────────────
# Verification helpers
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageVerificationResult:
    """Outcome of verifying one stage's attestation.

    Fields:
      stage_index       Echoed from the envelope, or ``-1`` for the
                        single-stage opaque-bytes back-compat path.
      stage_node_id     Echoed from the envelope, or empty string for
                        the back-compat path.
      tee_type          The stage's claimed TEE type. For
                        ``is_placeholder=True`` this is ``SOFTWARE``
                        as a conservative default — callers MUST NOT
                        treat this as a real software-tier event.
      structurally_ok   True iff the envelope entry parsed cleanly.
                        v1 stops here; v2 wires platform-vendor RPC.
      vendor_verified   None for v1 (no vendor service wired). v2:
                        True iff the platform-vendor attestation
                        service confirmed the report.
      is_placeholder    True iff this result represents the single-
                        stage opaque-bytes back-compat path (no
                        multi-stage envelope present). Monitoring
                        layers aggregating ``tee_type`` counts MUST
                        skip placeholder results.
      message           Free-form description of the verification
                        outcome (mainly for failure paths).
    """

    stage_index: int
    stage_node_id: str
    tee_type: TEEType
    structurally_ok: bool
    vendor_verified: Optional[bool]
    message: str
    is_placeholder: bool = False


def verify_stage_attestations(
    blob: bytes,
) -> Tuple[bool, List[StageVerificationResult]]:
    """Iterate the multi-stage envelope and produce per-stage results.

    Returns ``(all_ok, results)``:
      - ``all_ok`` is True iff every stage parsed cleanly. v1 doesn't
        wire vendor verification, so structural correctness is the
        only signal.
      - ``results`` is a list (one per stage) with the per-stage
        verification outcome.

    Single-stage opaque-bytes attestations return
    ``(True, [StageVerificationResult(stage_index=0, ...)])`` with the
    structural-ok flag True and ``vendor_verified=None``. Callers who
    need to distinguish single-stage from multi-stage should call
    ``is_multi_stage_attestation(blob)`` first.

    Malformed multi-stage envelopes (magic-prefixed but corrupt)
    return ``(False, [single error result])`` rather than raising —
    the verification API is non-throwing by contract, mirrors
    Phase 3.x.5 / 3.x.6 server-side "never raises" patterns.
    """
    if not is_multi_stage_attestation(blob):
        # Treat as single-stage opaque attestation. We can't infer the
        # stage_node_id or tee_type from raw bytes; report a placeholder
        # entry (is_placeholder=True) so callers can distinguish it
        # from a real multi-stage entry. Monitoring layers aggregating
        # tee_type counts MUST skip placeholder results.
        return True, [StageVerificationResult(
            stage_index=-1,
            stage_node_id="",
            tee_type=TEEType.SOFTWARE,
            structurally_ok=True,
            vendor_verified=None,
            message="single-stage opaque attestation (no envelope)",
            is_placeholder=True,
        )]

    try:
        stages = decode_multi_stage_attestation(blob)
    except MultiStageAttestationError as exc:
        return False, [StageVerificationResult(
            stage_index=-1,
            stage_node_id="",
            tee_type=TEEType.SOFTWARE,
            structurally_ok=False,
            vendor_verified=None,
            message=f"envelope decode failed: {exc}",
        )]

    if stages is None:
        # Should be unreachable given the is_multi_stage_attestation
        # guard above, but treat defensively.
        return False, [StageVerificationResult(
            stage_index=-1,
            stage_node_id="",
            tee_type=TEEType.SOFTWARE,
            structurally_ok=False,
            vendor_verified=None,
            message="decode returned None despite magic prefix present",
        )]

    results = [StageVerificationResult(
        stage_index=s.stage_index,
        stage_node_id=s.stage_node_id,
        tee_type=s.tee_type,
        structurally_ok=True,
        vendor_verified=None,  # v2 wires platform-vendor RPC here
        message="structurally valid; vendor verification not wired (v1)",
    ) for s in stages]
    return True, results
