"""Sprint 312 — pipeline inference receipt + verification.

The receipt is what makes multi-stage inference verifiable
end-to-end. Anyone holding the receipt + the orchestrator's
Ed25519 pubkey can confirm:

  1. The orchestrator signed it (Ed25519 over canonical
     payload).
  2. The activation hash chain is intact: stage K's
     output_activation_hash matches stage K+1's
     input_activation_hash. A MITM that swaps any
     intermediate activation breaks the chain.
  3. The recorded prompt_hash matches stage 0's input
     hash; the recorded output_hash matches the last
     stage's output hash.
  4. The partition_hash matches what the verifier
     expected (caller compares against
     PipelinePartition.partition_hash()).
  5. Each stage's TEE attestation tier meets a
     caller-specified minimum (sprint 305 composition —
     "this whole inference ran on real hardware TEEs,
     not software fallbacks").

The hash chain is the load-bearing security primitive.
Without it, the receipt is just "the orchestrator says
this happened." With it, the receipt is "verifiable
without trusting the orchestrator OR any single stage
operator."
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
)
from prsm.enterprise.federated_learning import (
    _ed25519_b64d,
    _ed25519_b64e,
    _load_ed25519_priv,
    _load_ed25519_pub,
)
from prsm.enterprise.tee_policy import (
    AttestationTier,
    effective_tier_from_result,
    tier_rank,
)


# ── Dataclasses ─────────────────────────────────────


@dataclass
class PerStageReceipt:
    stage_id: int
    layer_indices: List[int]
    input_activation_hash: str
    output_activation_hash: str
    attestation: AttestationVerificationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": int(self.stage_id),
            "layer_indices": list(self.layer_indices),
            "input_activation_hash": (
                self.input_activation_hash
            ),
            "output_activation_hash": (
                self.output_activation_hash
            ),
            "attestation": self.attestation.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "PerStageReceipt":
        att_raw = d.get("attestation") or {}
        return cls(
            stage_id=int(d["stage_id"]),
            layer_indices=list(
                d.get("layer_indices") or [],
            ),
            input_activation_hash=d[
                "input_activation_hash"
            ],
            output_activation_hash=d[
                "output_activation_hash"
            ],
            attestation=AttestationVerificationResult(
                vendor=att_raw.get("vendor", "unknown"),
                vendor_verified=bool(
                    att_raw.get("vendor_verified", False),
                ),
                vendor_data=dict(
                    att_raw.get("vendor_data") or {},
                ),
                signature_chain_ok=bool(
                    att_raw.get(
                        "signature_chain_ok", False,
                    ),
                ),
                error=att_raw.get("error"),
                structural_parse_ok=bool(
                    att_raw.get(
                        "structural_parse_ok", False,
                    ),
                ),
            ),
        )


@dataclass
class PipelineInferenceReceipt:
    prompt_hash: str
    output_hash: str
    partition_hash: str
    stage_receipts: List[PerStageReceipt]
    orchestrator_signature_b64: str
    version: str = "v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "partition_hash": self.partition_hash,
            "stage_receipts": [
                s.to_dict() for s in self.stage_receipts
            ],
            "orchestrator_signature_b64": (
                self.orchestrator_signature_b64
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "PipelineInferenceReceipt":
        return cls(
            prompt_hash=d["prompt_hash"],
            output_hash=d["output_hash"],
            partition_hash=d["partition_hash"],
            stage_receipts=[
                PerStageReceipt.from_dict(s)
                for s in (d.get("stage_receipts") or [])
            ],
            orchestrator_signature_b64=d.get(
                "orchestrator_signature_b64", "",
            ),
            version=d.get("version", "v1"),
        )


@dataclass
class PipelineVerificationResult:
    ok: bool
    signature_valid: bool
    chain_valid: bool
    diagnostic: str


# ── Canonical signing bytes ─────────────────────────


def _canonical_receipt_bytes(
    r: PipelineInferenceReceipt,
) -> bytes:
    """Stable signing payload. Excludes the signature
    field (sig can't bind itself)."""
    payload = {
        "version": r.version,
        "prompt_hash": r.prompt_hash,
        "output_hash": r.output_hash,
        "partition_hash": r.partition_hash,
        "stage_receipts": [
            s.to_dict() for s in r.stage_receipts
        ],
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


# ── Sign + verify ───────────────────────────────────


def sign_pipeline_receipt(
    receipt: PipelineInferenceReceipt,
    *,
    orchestrator_privkey_b64: str,
) -> PipelineInferenceReceipt:
    """Sign a receipt with the orchestrator's Ed25519
    privkey. Mutates and returns the receipt with
    orchestrator_signature_b64 populated."""
    priv = _load_ed25519_priv(orchestrator_privkey_b64)
    sig = priv.sign(_canonical_receipt_bytes(receipt))
    receipt.orchestrator_signature_b64 = _ed25519_b64e(sig)
    return receipt


def _verify_chain(
    receipt: PipelineInferenceReceipt,
) -> tuple[bool, str]:
    """Return (chain_valid, diagnostic). The chain
    invariants:
      1. Receipt has >= 1 stage
      2. Stage 0's input_activation_hash == prompt_hash
      3. For each adjacent pair: stage K's
         output_activation_hash == stage K+1's
         input_activation_hash
      4. Last stage's output_activation_hash ==
         receipt.output_hash
    """
    stages = receipt.stage_receipts
    if not stages:
        return False, "receipt has no stages"
    if (
        stages[0].input_activation_hash
        != receipt.prompt_hash
    ):
        return False, (
            f"stage 0 input_activation_hash "
            f"({stages[0].input_activation_hash[:12]}...) "
            f"!= receipt.prompt_hash "
            f"({receipt.prompt_hash[:12]}...) — chain "
            f"broken at prompt entry"
        )
    for k in range(len(stages) - 1):
        if (
            stages[k].output_activation_hash
            != stages[k + 1].input_activation_hash
        ):
            return False, (
                f"stage {k} output_activation_hash != "
                f"stage {k + 1} input_activation_hash — "
                f"chain broken between stages"
            )
    if (
        stages[-1].output_activation_hash
        != receipt.output_hash
    ):
        return False, (
            f"last stage output_activation_hash "
            f"({stages[-1].output_activation_hash[:12]}...) "
            f"!= receipt.output_hash "
            f"({receipt.output_hash[:12]}...) — chain "
            f"broken at output exit"
        )
    return True, "activation hash chain intact"


def _verify_attestation_tiers(
    receipt: PipelineInferenceReceipt,
    required_tier: AttestationTier,
) -> tuple[bool, str]:
    required_rank = tier_rank(required_tier)
    for stage in receipt.stage_receipts:
        effective = effective_tier_from_result(
            stage.attestation,
        )
        if tier_rank(effective) < required_rank:
            return False, (
                f"stage {stage.stage_id} attestation tier "
                f"{effective.value!r} below required "
                f"{required_tier.value!r}"
            )
    return True, "all stages meet attestation tier"


def verify_pipeline_receipt(
    receipt: PipelineInferenceReceipt,
    *,
    orchestrator_pubkey_b64: str,
    require_min_attestation_tier: Optional[str] = None,
) -> PipelineVerificationResult:
    """Verify the receipt end-to-end:
      1. Orchestrator Ed25519 signature
      2. Activation hash chain (load-bearing)
      3. Optional minimum attestation tier per stage
    """
    # Resolve attestation-tier requirement upfront so an
    # invalid value fails LOUD before doing any crypto.
    tier_obj: Optional[AttestationTier] = None
    if require_min_attestation_tier is not None:
        try:
            tier_obj = AttestationTier(
                require_min_attestation_tier.lower(),
            )
        except ValueError:
            raise ValueError(
                f"unknown attestation tier "
                f"{require_min_attestation_tier!r}"
            )

    # Signature
    try:
        pub = _load_ed25519_pub(orchestrator_pubkey_b64)
        sig = _ed25519_b64d(
            receipt.orchestrator_signature_b64,
        )
    except ValueError as e:
        return PipelineVerificationResult(
            ok=False, signature_valid=False,
            chain_valid=False,
            diagnostic=f"signature parse error: {e}",
        )
    try:
        pub.verify(sig, _canonical_receipt_bytes(receipt))
        sig_ok = True
    except InvalidSignature:
        sig_ok = False

    # Chain
    chain_ok, chain_diag = _verify_chain(receipt)

    # Attestation tier (optional)
    if tier_obj is not None:
        att_ok, att_diag = _verify_attestation_tiers(
            receipt, tier_obj,
        )
    else:
        att_ok, att_diag = True, "no tier requirement"

    overall = sig_ok and chain_ok and att_ok
    if not sig_ok:
        diag = "orchestrator signature did not verify"
    elif not chain_ok:
        diag = chain_diag
    elif not att_ok:
        diag = att_diag
    else:
        diag = "ok: signature + chain + attestation tier"

    return PipelineVerificationResult(
        ok=overall,
        signature_valid=sig_ok,
        chain_valid=chain_ok,
        diagnostic=diag,
    )
