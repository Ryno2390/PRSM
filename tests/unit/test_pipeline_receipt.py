"""Sprint 312 — pipeline inference receipt.

The receipt is what makes multi-stage inference VERIFIABLE
end-to-end. Anyone holding the receipt + the orchestrator's
pubkey can confirm:
  - The orchestrator signed it (Ed25519)
  - The activation hash chain is intact: stage K's output
    hash == stage K+1's input hash (no MITM substitution
    of intermediate activations)
  - The partition hash matches what they expected (no
    substitution of model partitioning)
  - Each stage's TEE attestation satisfies the caller's
    minimum tier requirement (sprint 305 composition)
"""
from __future__ import annotations

import base64
import hashlib

import pytest

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
)
from prsm.compute.inference.pipeline_receipt import (
    PerStageReceipt,
    PipelineInferenceReceipt,
    sign_pipeline_receipt,
    verify_pipeline_receipt,
)
from prsm.enterprise.federated_learning import (
    generate_worker_keypair,
)


def _hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _per_stage(
    stage_id: int, input_hash: str, output_hash: str,
    layers=None,
    attestation=None,
) -> PerStageReceipt:
    return PerStageReceipt(
        stage_id=stage_id,
        layer_indices=list(layers or [stage_id]),
        input_activation_hash=input_hash,
        output_activation_hash=output_hash,
        attestation=(
            attestation
            if attestation is not None
            else AttestationVerificationResult(
                vendor="software-fallback",
                structural_parse_ok=True,
            )
        ),
    )


def _build_chained_receipt(
    *,
    prompt=b"a prompt",
    activations=None,
):
    """Build a receipt whose stages form a valid hash chain.
    `activations[0]` IS the prompt (so stage 0's input_hash
    == prompt_hash by construction); subsequent entries
    are intermediate activations; `activations[-1]` is the
    final output."""
    activations = activations or [
        prompt, b"act-1", b"act-2", b"final",
    ]
    prompt_hash = _hash(activations[0])
    output_hash = _hash(activations[-1])
    stage_receipts = []
    # Stage K: input = hash(activations[K]); output =
    # hash(activations[K+1])
    for k in range(len(activations) - 1):
        in_h = _hash(activations[k])
        out_h = _hash(activations[k + 1])
        stage_receipts.append(_per_stage(
            stage_id=k,
            input_hash=in_h,
            output_hash=out_h,
        ))
    return PipelineInferenceReceipt(
        prompt_hash=prompt_hash,
        output_hash=output_hash,
        partition_hash="fake-partition-hash",
        stage_receipts=stage_receipts,
        orchestrator_signature_b64="",
    )


# ── Serialization round-trip ───────────────────────


def test_per_stage_receipt_round_trip():
    s = _per_stage(0, "in", "out")
    restored = PerStageReceipt.from_dict(s.to_dict())
    assert restored.stage_id == s.stage_id
    assert restored.input_activation_hash == s.input_activation_hash
    assert restored.output_activation_hash == s.output_activation_hash
    assert (
        restored.attestation.vendor == s.attestation.vendor
    )


def test_pipeline_receipt_round_trip():
    r = _build_chained_receipt()
    restored = PipelineInferenceReceipt.from_dict(
        r.to_dict(),
    )
    assert restored.prompt_hash == r.prompt_hash
    assert restored.output_hash == r.output_hash
    assert restored.partition_hash == r.partition_hash
    assert len(restored.stage_receipts) == len(
        r.stage_receipts,
    )


# ── Signature + verification happy path ────────────


def test_sign_and_verify_happy_path():
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    assert signed.orchestrator_signature_b64
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert result.ok
    assert result.signature_valid
    assert result.chain_valid


# ── Tamper detection: intermediate activation ──────


def test_tampered_intermediate_activation_breaks_chain():
    """The point of the hash chain: a MITM that swaps an
    intermediate stage's output cannot escape detection
    because the next stage's recorded input hash won't
    match anymore."""
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    # Tamper stage 1's output_activation_hash — now stage
    # 2's input_activation_hash no longer matches
    r.stage_receipts[1].output_activation_hash = (
        _hash(b"FORGED-INTERMEDIATE")
    )
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert not result.ok
    assert not result.chain_valid
    assert "chain" in result.diagnostic.lower()


def test_signature_tamper_detected():
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    # Tamper the output_hash AFTER signing
    signed.output_hash = _hash(b"FORGED-OUTPUT")
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert not result.ok
    assert not result.signature_valid


def test_wrong_orchestrator_pubkey_rejected():
    priv, _ = generate_worker_keypair()
    _, other_pub = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=other_pub,
    )
    assert not result.ok
    assert not result.signature_valid


def test_partition_hash_tamper_breaks_signature():
    """Substituting the partition_hash after signing
    breaks verification (the partition_hash is part of
    the signed payload)."""
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    signed.partition_hash = "different-partition"
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert not result.ok


# ── Hash chain edge cases ───────────────────────────


def test_single_stage_pipeline():
    """A 1-stage pipeline: prompt → stage 0 → final.
    Hash chain is trivial (one input, one output) but the
    receipt must still verify."""
    priv, pub = generate_worker_keypair()
    prompt = b"single-stage test"
    output = b"final output"
    receipt = PipelineInferenceReceipt(
        prompt_hash=_hash(prompt),
        output_hash=_hash(output),
        partition_hash="ph",
        stage_receipts=[_per_stage(
            stage_id=0,
            input_hash=_hash(prompt),
            output_hash=_hash(output),
        )],
        orchestrator_signature_b64="",
    )
    signed = sign_pipeline_receipt(
        receipt, orchestrator_privkey_b64=priv,
    )
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert result.ok


def test_chain_requires_stage_0_input_matches_prompt():
    """Stage 0's input_activation_hash must equal the
    receipt's prompt_hash (otherwise the orchestrator
    could substitute the prompt that fed into stage 0)."""
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    # Make stage 0's input hash != prompt_hash
    r.stage_receipts[0].input_activation_hash = _hash(
        b"DIFFERENT-PROMPT",
    )
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert not result.ok
    assert not result.chain_valid


def test_chain_requires_last_stage_output_matches_receipt():
    """Final stage's output_activation_hash must equal the
    receipt's output_hash."""
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    # Make the receipt's output_hash != last stage's output
    r.output_hash = _hash(b"DIFFERENT-OUTPUT")
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    result = verify_pipeline_receipt(
        signed, orchestrator_pubkey_b64=pub,
    )
    assert not result.ok
    assert not result.chain_valid


# ── Attestation tier requirement ───────────────────


def test_min_attestation_tier_gate_pass():
    """When the caller requires a minimum attestation tier,
    verify checks each stage's attestation against it. All
    stages 'software-fallback' → passes 'software' but
    fails 'hardware_unverified'."""
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    # Default attestation in _per_stage is software-fallback
    result = verify_pipeline_receipt(
        signed,
        orchestrator_pubkey_b64=pub,
        require_min_attestation_tier="software",
    )
    assert result.ok


def test_min_attestation_tier_gate_fail():
    priv, pub = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    result = verify_pipeline_receipt(
        signed,
        orchestrator_pubkey_b64=pub,
        require_min_attestation_tier="hardware_unverified",
    )
    assert not result.ok
    assert "attestation" in result.diagnostic.lower()


def test_min_attestation_tier_invalid_value():
    priv, _ = generate_worker_keypair()
    r = _build_chained_receipt()
    signed = sign_pipeline_receipt(
        r, orchestrator_privkey_b64=priv,
    )
    _, pub = generate_worker_keypair()
    with pytest.raises(ValueError, match="tier"):
        verify_pipeline_receipt(
            signed,
            orchestrator_pubkey_b64=pub,
            require_min_attestation_tier="not-a-tier",
        )
