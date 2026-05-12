"""Sprint 292 — privacy-claim verification public API.

Vision §7 claim audit: PRSM advertises "TEE-attested compute"
(§7 line 432, line 437) as a structural property. Today the
attestation envelope is in place (executor + receipt fields)
but every receipt produced by the local executor carries a
``DEV-ONLY-SW-TEE:`` software-stub attestation (executor.py
:592-613). There is NO public API that lets a caller detect
that — meaning end-users currently CANNOT tell whether their
prompt ran under real hardware TEE or software fallback.

This sprint exposes that truth. Three surfaces:

  is_dev_only_attestation(blob) -> bool
    Public predicate; matches the docstring intent at
    executor.py:602-606 ("verifiers MUST reject any
    attestation starting with the DEV-ONLY prefix").

  verify_receipt_privacy_claim(receipt, *,
        require_hardware_attestation=False,
        require_dp_noise=False, identity=None,
        public_key_b64=None) -> PrivacyVerification
    Composite check returning (ok, [reasons]). Validates:
      - settler signature (existing verify_receipt)
      - epsilon_spent > 0 iff privacy_tier != none (DP noise
        was actually applied)
      - tee_attestation is hardware-backed (when required)
      - multi-stage envelope verifies structurally
        (when multi-stage envelope present)

  POST /compute/receipt/verify   HTTP surface
  prsm_verify_inference_privacy  MCP surface

The HTTP+MCP surface ships in sprint 293; this sprint
ships the verification primitives + their tests + the
inline endpoint hook.
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)
from prsm.compute.inference.privacy_verification import (
    PrivacyVerification,
    is_dev_only_attestation,
    verify_receipt_privacy_claim,
)
from prsm.compute.inference.receipt import (
    InferenceReceipt, sign_receipt,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, generate_node_identity


# ── is_dev_only_attestation pure predicate ───────────────


def test_dev_only_prefix_blob_detected():
    # 16 prefix bytes + 48 sha384 = 64 byte software stub
    blob = SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 48
    assert is_dev_only_attestation(blob) is True


def test_real_hardware_attestation_not_dev_only():
    """A real hardware attestation does NOT start with the
    DEV-ONLY prefix. Use a 64-byte non-prefix blob to
    represent a hardware attestation in this test."""
    blob = b"REAL_VENDOR_HW_" + b"\x00" * 49
    assert is_dev_only_attestation(blob) is False


def test_empty_attestation_not_dev_only():
    """Empty bytes is not DEV-ONLY (it's just absent). The
    require_hardware_attestation check handles emptiness
    separately."""
    assert is_dev_only_attestation(b"") is False


def test_short_attestation_not_dev_only():
    """Anything shorter than the prefix length cannot match."""
    assert is_dev_only_attestation(b"DEV-ONLY") is False


def test_none_attestation_not_dev_only():
    """None should be handled defensively."""
    assert is_dev_only_attestation(None) is False  # type: ignore


# ── PrivacyVerification dataclass ────────────────────────


def test_privacy_verification_to_dict():
    v = PrivacyVerification(
        ok=False,
        reasons=["signature invalid", "dev-only attestation"],
        signature_valid=False,
        dp_noise_applied=True,
        hardware_attested=False,
        multi_stage_envelope_present=False,
    )
    d = v.to_dict()
    assert d["ok"] is False
    assert "signature invalid" in d["reasons"]
    assert d["signature_valid"] is False
    assert d["hardware_attested"] is False


# ── Helper: build a signed software-attested receipt ─────


def _receipt_at_tier(
    privacy_tier: PrivacyLevel,
    *,
    epsilon_spent: float | None = None,
    sign: bool = True,
    use_dev_only_attestation: bool = True,
) -> tuple[InferenceReceipt, NodeIdentity]:
    """Build a receipt with the given privacy posture. Returns
    (receipt, identity-that-signed-it)."""
    identity = generate_node_identity("verifier-test")
    if epsilon_spent is None:
        epsilon_spent = (
            0.0 if privacy_tier == PrivacyLevel.NONE else 8.0
        )
    if use_dev_only_attestation:
        attestation = (
            SOFTWARE_TEE_ATTESTATION_PREFIX
            + hashlib.sha384(b"sw-tee:test").digest()
        )
    else:
        attestation = b"HW_VENDOR_QUOTE_" + b"\x00" * 48

    receipt = InferenceReceipt(
        job_id="infer-job-test",
        request_id="req-test",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=privacy_tier,
        epsilon_spent=epsilon_spent,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=attestation,
        output_hash=hashlib.sha256(b"out").digest(),
        duration_seconds=0.1,
        cost_ftns="0.01",
        settler_signature=b"\x00" * 64,
        settler_node_id="",
    )
    if sign:
        receipt = sign_receipt(receipt, identity)
    return receipt, identity


# ── verify_receipt_privacy_claim happy path ──────────────


def test_verify_signed_dev_only_software_attested_default():
    """Default verification (require_hardware_attestation=
    False): a signed receipt with dev-only attestation should
    pass overall — but hardware_attested should be False
    so callers see the truth."""
    receipt, identity = _receipt_at_tier(PrivacyLevel.STANDARD)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.signature_valid is True
    assert result.dp_noise_applied is True
    assert result.hardware_attested is False
    # Default: no hardware requirement → ok
    assert result.ok is True


def test_verify_signed_hardware_attested():
    """Receipt with non-dev-only attestation flips hardware_
    attested to True."""
    receipt, identity = _receipt_at_tier(
        PrivacyLevel.STANDARD,
        use_dev_only_attestation=False,
    )
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.hardware_attested is True
    assert result.ok is True


def test_verify_require_hardware_fails_on_dev_only():
    """When caller requires hardware attestation, dev-only
    receipts fail with a clear reason."""
    receipt, identity = _receipt_at_tier(PrivacyLevel.STANDARD)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
        require_hardware_attestation=True,
    )
    assert result.ok is False
    assert result.hardware_attested is False
    assert any(
        "dev-only" in r.lower() or "software" in r.lower()
        or "hardware" in r.lower()
        for r in result.reasons
    )


def test_verify_require_hardware_passes_on_real_attestation():
    receipt, identity = _receipt_at_tier(
        PrivacyLevel.STANDARD,
        use_dev_only_attestation=False,
    )
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
        require_hardware_attestation=True,
    )
    assert result.ok is True
    assert result.hardware_attested is True


# ── DP noise applied gate ────────────────────────────────


def test_dp_noise_not_applied_when_tier_none():
    """privacy_tier=none with epsilon=0 → dp_noise_applied
    is False, but that's expected (no privacy promise
    made)."""
    receipt, identity = _receipt_at_tier(PrivacyLevel.NONE)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.dp_noise_applied is False
    # ok regardless — caller didn't promise privacy
    assert result.ok is True


def test_dp_noise_required_but_missing_fails():
    """If caller requires DP noise (the receipt's privacy_
    tier claims privacy but epsilon_spent==0), fail."""
    # Build a malformed receipt: tier=STANDARD but ε=0 (the
    # executor should never produce this, but it's worth
    # defending against)
    receipt, identity = _receipt_at_tier(
        PrivacyLevel.STANDARD, epsilon_spent=0.0,
    )
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
        require_dp_noise=True,
    )
    assert result.ok is False
    assert result.dp_noise_applied is False
    assert any(
        "noise" in r.lower() or "epsilon" in r.lower()
        for r in result.reasons
    )


def test_dp_noise_required_passes_when_epsilon_spent():
    receipt, identity = _receipt_at_tier(PrivacyLevel.STANDARD)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
        require_dp_noise=True,
    )
    assert result.dp_noise_applied is True
    assert result.ok is True


# ── Signature checks ─────────────────────────────────────


def test_unsigned_receipt_fails():
    receipt, identity = _receipt_at_tier(
        PrivacyLevel.STANDARD, sign=False,
    )
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.signature_valid is False
    assert result.ok is False


def test_signature_with_wrong_key_fails():
    receipt, _ = _receipt_at_tier(PrivacyLevel.STANDARD)
    # Use a different identity to verify
    other = generate_node_identity("other-node")
    result = verify_receipt_privacy_claim(
        receipt, identity=other,
    )
    assert result.signature_valid is False
    assert result.ok is False


def test_signature_via_public_key_b64():
    """Verifier can accept public_key_b64 as an alternative
    to identity — important for MCP-side verification where
    we have the b64 from /node/identity/pubkey."""
    import base64
    receipt, identity = _receipt_at_tier(PrivacyLevel.STANDARD)
    pubkey_b64 = base64.b64encode(
        identity.public_key_bytes
    ).decode("ascii")
    result = verify_receipt_privacy_claim(
        receipt, public_key_b64=pubkey_b64,
    )
    assert result.signature_valid is True
    assert result.ok is True


# ── Combined: privacy-tier-honored predicate ─────────────


def test_high_tier_requires_smaller_epsilon():
    """A receipt claiming privacy_tier=HIGH (ε=4) but with
    epsilon_spent=8 isn't honoring the claimed posture.
    Verifier surfaces the mismatch."""
    receipt, identity = _receipt_at_tier(
        PrivacyLevel.HIGH, epsilon_spent=8.0,
    )
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
        require_dp_noise=True,
    )
    # epsilon_spent > 0 means DP noise was applied,
    # but the value doesn't match HIGH's ε=4. Surface as a
    # reason; ok stays True if signature + noise check pass
    # but expected_epsilon mismatch is captured for callers.
    assert result.dp_noise_applied is True
    assert result.expected_epsilon == 4.0
    assert result.epsilon_spent == 8.0
    assert any(
        "mismatch" in r.lower() or "expected" in r.lower()
        for r in result.reasons
    )


def test_to_dict_includes_all_fields():
    receipt, identity = _receipt_at_tier(PrivacyLevel.STANDARD)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    d = result.to_dict()
    for k in [
        "ok", "reasons", "signature_valid",
        "dp_noise_applied", "hardware_attested",
        "multi_stage_envelope_present",
        "privacy_tier", "epsilon_spent",
        "expected_epsilon",
    ]:
        assert k in d
