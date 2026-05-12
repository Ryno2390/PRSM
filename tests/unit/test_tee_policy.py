"""Sprint 305 — TEE-only execution policy primitive.

Vision §7 Enterprise Confidentiality Mode layer 3:
extends the §7 attestation backend registry with a
declarative policy so an enterprise can express "this
job must run on a real TEE, not software fallback" and
have the policy enforced — or at minimum verified before
dispatch.

The policy operates on the existing AttestationVerification-
Result shape (sprint 293-297 substrate). Effective-tier
derivation:
  - unknown / errored      → NONE
  - software-fallback      → SOFTWARE
  - real vendor (intel/amd
    /apple) + verified=F   → HARDWARE_UNVERIFIED
  - real vendor + verified=T → HARDWARE_VERIFIED

A policy is satisfied when the effective tier >= the
policy's min_attestation_tier AND vendor allowlist
(if any) admits the result's vendor.
"""
from __future__ import annotations

import struct

import pytest

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
)
from prsm.enterprise.tee_policy import (
    AttestationTier,
    PolicyResult,
    PolicyStatus,
    TEEPolicy,
    effective_tier_from_result,
    evaluate_attestation_blob,
    evaluate_attestation_result,
    tier_rank,
)


# ── Enum + ordering ──────────────────────────────────


def test_tier_values():
    assert AttestationTier.NONE.value == "none"
    assert AttestationTier.SOFTWARE.value == "software"
    assert (
        AttestationTier.HARDWARE_UNVERIFIED.value
        == "hardware_unverified"
    )
    assert (
        AttestationTier.HARDWARE_VERIFIED.value
        == "hardware_verified"
    )


def test_tier_rank_strictly_increasing():
    assert (
        tier_rank(AttestationTier.NONE)
        < tier_rank(AttestationTier.SOFTWARE)
        < tier_rank(AttestationTier.HARDWARE_UNVERIFIED)
        < tier_rank(AttestationTier.HARDWARE_VERIFIED)
    )


def test_status_values():
    assert PolicyStatus.PASS.value == "pass"
    assert PolicyStatus.FAIL.value == "fail"
    assert PolicyStatus.SKIPPED.value == "skipped"


# ── effective_tier_from_result ───────────────────────


def _r(**kw) -> AttestationVerificationResult:
    return AttestationVerificationResult(**kw)


def test_effective_tier_unknown_is_none():
    assert effective_tier_from_result(
        _r(vendor="unknown"),
    ) == AttestationTier.NONE


def test_effective_tier_error_is_none():
    assert effective_tier_from_result(
        _r(vendor="intel-sgx", error="parse failed"),
    ) == AttestationTier.NONE


def test_effective_tier_software_fallback():
    assert effective_tier_from_result(
        _r(vendor="software-fallback",
           structural_parse_ok=True),
    ) == AttestationTier.SOFTWARE


def test_effective_tier_real_vendor_unverified():
    for v in (
        "intel-sgx", "intel-tdx", "amd-sev-snp", "apple-sep",
    ):
        assert effective_tier_from_result(
            _r(vendor=v, structural_parse_ok=True,
               vendor_verified=False),
        ) == AttestationTier.HARDWARE_UNVERIFIED


def test_effective_tier_real_vendor_verified():
    for v in (
        "intel-sgx", "intel-tdx", "amd-sev-snp", "apple-sep",
    ):
        assert effective_tier_from_result(
            _r(vendor=v, structural_parse_ok=True,
               vendor_verified=True,
               signature_chain_ok=True),
        ) == AttestationTier.HARDWARE_VERIFIED


# ── TEEPolicy basic satisfaction ────────────────────


def test_policy_none_accepts_everything():
    policy = TEEPolicy(
        min_attestation_tier=AttestationTier.NONE,
    )
    # Even an unknown blob passes
    res = evaluate_attestation_result(
        _r(vendor="unknown"), policy,
    )
    assert res.status == PolicyStatus.PASS


def test_policy_software_rejects_unknown():
    policy = TEEPolicy(
        min_attestation_tier=AttestationTier.SOFTWARE,
    )
    res = evaluate_attestation_result(
        _r(vendor="unknown"), policy,
    )
    assert res.status == PolicyStatus.FAIL


def test_policy_software_accepts_fallback():
    policy = TEEPolicy(
        min_attestation_tier=AttestationTier.SOFTWARE,
    )
    res = evaluate_attestation_result(
        _r(vendor="software-fallback",
           structural_parse_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.PASS


def test_policy_hardware_rejects_fallback():
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_UNVERIFIED
        ),
    )
    res = evaluate_attestation_result(
        _r(vendor="software-fallback",
           structural_parse_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.FAIL
    assert "tier" in (res.diagnostic or "").lower()


def test_policy_hardware_accepts_real_vendor():
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_UNVERIFIED
        ),
    )
    res = evaluate_attestation_result(
        _r(vendor="intel-sgx", structural_parse_ok=True,
           vendor_verified=False),
        policy,
    )
    assert res.status == PolicyStatus.PASS
    assert res.effective_tier == (
        AttestationTier.HARDWARE_UNVERIFIED
    )


def test_policy_hardware_verified_rejects_unverified():
    """If the policy mandates real cryptographic
    verification (DCAP / equivalent), a structurally-
    parsed-but-not-crypto-verified quote must be denied."""
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_VERIFIED
        ),
    )
    res = evaluate_attestation_result(
        _r(vendor="intel-sgx", structural_parse_ok=True,
           vendor_verified=False),
        policy,
    )
    assert res.status == PolicyStatus.FAIL


def test_policy_hardware_verified_accepts_verified():
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_VERIFIED
        ),
    )
    res = evaluate_attestation_result(
        _r(vendor="amd-sev-snp", structural_parse_ok=True,
           vendor_verified=True,
           signature_chain_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.PASS


# ── Vendor allowlist ─────────────────────────────────


def test_vendor_allowlist_admits_listed():
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_UNVERIFIED
        ),
        allowed_vendors={"intel-sgx", "amd-sev-snp"},
    )
    res = evaluate_attestation_result(
        _r(vendor="intel-sgx", structural_parse_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.PASS


def test_vendor_allowlist_rejects_unlisted():
    """Even if tier is high enough, an unlisted vendor
    must be denied — covers the 'we only trust Intel for
    this workload' case."""
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_UNVERIFIED
        ),
        allowed_vendors={"intel-sgx", "intel-tdx"},
    )
    res = evaluate_attestation_result(
        _r(vendor="amd-sev-snp", structural_parse_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.FAIL
    assert "vendor" in (res.diagnostic or "").lower()


def test_empty_allowlist_admits_nothing():
    """An explicit empty allowlist (vs unset/None) is
    operator confusion — fail every check so the misconfig
    is loud."""
    policy = TEEPolicy(
        min_attestation_tier=AttestationTier.SOFTWARE,
        allowed_vendors=set(),
    )
    res = evaluate_attestation_result(
        _r(vendor="intel-sgx", structural_parse_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.FAIL


def test_none_allowlist_admits_any():
    """`allowed_vendors=None` means no allowlist gate;
    only the tier check applies."""
    policy = TEEPolicy(
        min_attestation_tier=AttestationTier.SOFTWARE,
        allowed_vendors=None,
    )
    res = evaluate_attestation_result(
        _r(vendor="apple-sep", structural_parse_ok=True),
        policy,
    )
    assert res.status == PolicyStatus.PASS


# ── evaluate_attestation_blob (registry-routed) ─────


# A small valid software-fallback blob. Format pinned by
# `SOFTWARE_TEE_ATTESTATION_PREFIX` in attestation_backends.
def _software_blob() -> bytes:
    from prsm.compute.inference.attestation_backends import (
        SOFTWARE_TEE_ATTESTATION_PREFIX,
    )
    return SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 32


def test_blob_software_passes_software_policy():
    res = evaluate_attestation_blob(
        _software_blob(),
        TEEPolicy(
            min_attestation_tier=AttestationTier.SOFTWARE,
        ),
    )
    assert res.status == PolicyStatus.PASS


def test_blob_software_fails_hardware_policy():
    res = evaluate_attestation_blob(
        _software_blob(),
        TEEPolicy(
            min_attestation_tier=(
                AttestationTier.HARDWARE_UNVERIFIED
            ),
        ),
    )
    assert res.status == PolicyStatus.FAIL


def test_blob_none_fails_software_policy():
    res = evaluate_attestation_blob(
        None,
        TEEPolicy(
            min_attestation_tier=AttestationTier.SOFTWARE,
        ),
    )
    assert res.status == PolicyStatus.FAIL


def test_blob_garbage_fails_loud():
    res = evaluate_attestation_blob(
        b"definitely-not-an-attestation",
        TEEPolicy(
            min_attestation_tier=AttestationTier.SOFTWARE,
        ),
    )
    assert res.status == PolicyStatus.FAIL


# ── Serialization ───────────────────────────────────


def test_policy_to_dict_round_trip():
    policy = TEEPolicy(
        min_attestation_tier=(
            AttestationTier.HARDWARE_UNVERIFIED
        ),
        allowed_vendors={"intel-sgx", "amd-sev-snp"},
        require_signature_chain=False,
    )
    d = policy.to_dict()
    assert d["min_attestation_tier"] == (
        "hardware_unverified"
    )
    assert sorted(d["allowed_vendors"]) == [
        "amd-sev-snp", "intel-sgx",
    ]
    restored = TEEPolicy.from_dict(d)
    assert restored == policy


def test_policy_from_dict_unknown_tier_rejected():
    with pytest.raises(ValueError, match="tier"):
        TEEPolicy.from_dict({
            "min_attestation_tier": "made-up",
            "allowed_vendors": None,
        })


def test_result_to_dict():
    policy = TEEPolicy(
        min_attestation_tier=AttestationTier.SOFTWARE,
    )
    res = evaluate_attestation_result(
        _r(vendor="software-fallback",
           structural_parse_ok=True),
        policy,
    )
    d = res.to_dict()
    assert d["status"] == "pass"
    assert d["effective_tier"] == "software"
    assert d["min_required_tier"] == "software"
