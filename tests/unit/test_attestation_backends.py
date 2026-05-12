"""Sprint 293 — hardware attestation backend interface +
Intel ASP stub.

Sprint 292 surfaced the truth that every local-executor
receipt today carries a `DEV-ONLY-SW-TEE:` stub attestation.
This sprint builds the BACKEND INTERFACE that real hardware
attestation (Intel SGX/TDX via ASP, AMD SEV-SNP via KDS,
Apple SEP) plugs behind.

Scope this sprint:
  - AttestationBackend Protocol + AttestationVerificationResult
  - AttestationBackendRegistry — vendor detection by quote
    magic bytes
  - IntelASPBackend — structural parsing of SGX (v3) + TDX
    (v4) quote headers. v1 returns structural-only result;
    real cryptographic verification (Intel DCAP library +
    Intel signing key chain) ships behind the same interface
    when commissioned.
  - DevOnlyBackend — handles the sprint-292 software-fallback
    prefix as a first-class backend (returns
    vendor="software-fallback", vendor_verified=False).

Sprint 294 will add AMD KDS backend; sprint 297 will wire
real cryptographic verification behind these structural
parsers.
"""
from __future__ import annotations

import hashlib
import struct

import pytest

from prsm.compute.inference.attestation_backends import (
    AttestationBackend,
    AttestationBackendRegistry,
    AttestationVerificationResult,
    DevOnlyBackend,
    IntelASPBackend,
    detect_vendor,
    verify_attestation,
)
from prsm.compute.inference.executor import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)


# ── Helpers: build synthetic quote bytes ─────────────────


def _sgx_v3_quote(
    mrenclave: bytes = b"\x11" * 32,
    mrsigner: bytes = b"\x22" * 32,
) -> bytes:
    """Build a structurally-valid SGX v3 quote header.

    Real SGX quote format (simplified):
      bytes 0-1:  version (uint16 LE) = 3
      bytes 2-3:  att_key_type (uint16 LE)
      bytes 4-7:  reserved
      bytes 8-23: qe_svn + pce_svn + vendor id
      bytes 48-79:  MRENCLAVE
      bytes 176-207: MRSIGNER

    For sprint-293 structural parsing we only need the
    version + the two measurement fields at known offsets.
    Pad the rest with deterministic bytes.
    """
    header = struct.pack("<H", 3)        # version=3 (SGX)
    header += struct.pack("<H", 2)       # att_key_type=ECDSA-P256
    header += b"\x00" * 44               # reserved + vendor block
    assert len(header) == 48
    # MRENCLAVE at offset 48 (32 bytes)
    body = mrenclave
    # Filler to MRSIGNER offset (176)
    body += b"\x00" * (176 - 48 - 32)
    body += mrsigner
    body += b"\x00" * 64                 # report_data + tail
    return header + body


def _tdx_v4_quote(
    mrtd: bytes = b"\x33" * 48,
    rtmr0: bytes = b"\x44" * 48,
) -> bytes:
    """Build a structurally-valid TDX v4 quote header. Version=4."""
    header = struct.pack("<H", 4)        # version=4 (TDX)
    header += struct.pack("<H", 2)
    header += b"\x00" * 44
    body = mrtd
    body += b"\x00" * (176 - 48 - 48)
    body += rtmr0
    body += b"\x00" * 64
    return header + body


def _dev_only_blob() -> bytes:
    return SOFTWARE_TEE_ATTESTATION_PREFIX + hashlib.sha384(
        b"sw-tee:test",
    ).digest()


# ── AttestationVerificationResult dataclass ──────────────


def test_result_to_dict():
    r = AttestationVerificationResult(
        vendor="intel-sgx",
        vendor_verified=False,
        vendor_data={"mrenclave_hex": "ab"},
        signature_chain_ok=False,
        error=None,
        structural_parse_ok=True,
    )
    d = r.to_dict()
    assert d["vendor"] == "intel-sgx"
    assert d["vendor_verified"] is False
    assert d["vendor_data"]["mrenclave_hex"] == "ab"
    assert d["structural_parse_ok"] is True


# ── detect_vendor pure function ──────────────────────────


def test_detect_vendor_sgx_v3():
    assert detect_vendor(_sgx_v3_quote()) == "intel-sgx"


def test_detect_vendor_tdx_v4():
    assert detect_vendor(_tdx_v4_quote()) == "intel-tdx"


def test_detect_vendor_dev_only():
    assert detect_vendor(_dev_only_blob()) == "software-fallback"


def test_detect_vendor_empty():
    assert detect_vendor(b"") == "unknown"


def test_detect_vendor_short():
    assert detect_vendor(b"\x01") == "unknown"


def test_detect_vendor_unrecognized():
    # Version=99 (not SGX or TDX); not the dev-only prefix
    blob = struct.pack("<H", 99) + b"\x00" * 300
    assert detect_vendor(blob) == "unknown"


# ── IntelASPBackend.verify ───────────────────────────────


def test_intel_asp_parses_sgx_quote():
    backend = IntelASPBackend()
    quote = _sgx_v3_quote(
        mrenclave=b"\x11" * 32,
        mrsigner=b"\x22" * 32,
    )
    result = backend.verify(quote)
    assert result.vendor == "intel-sgx"
    assert result.structural_parse_ok is True
    # v1 returns False until DCAP cryptographic verification
    # is wired (sprint 297). The vendor_data carries the
    # parsed measurements so callers can pin against expected
    # values out-of-band.
    assert result.vendor_verified is False
    assert (
        result.vendor_data["mrenclave_hex"] == "11" * 32
    )
    assert (
        result.vendor_data["mrsigner_hex"] == "22" * 32
    )
    assert result.vendor_data["version"] == 3


def test_intel_asp_parses_tdx_quote():
    backend = IntelASPBackend()
    quote = _tdx_v4_quote(
        mrtd=b"\xab" * 48,
        rtmr0=b"\xcd" * 48,
    )
    result = backend.verify(quote)
    assert result.vendor == "intel-tdx"
    assert result.structural_parse_ok is True
    assert result.vendor_data["mrtd_hex"] == "ab" * 48
    assert result.vendor_data["rtmr0_hex"] == "cd" * 48
    assert result.vendor_data["version"] == 4


def test_intel_asp_rejects_dev_only_blob():
    """Intel backend doesn't claim ownership of the dev-only
    blob — vendor mismatch returns unparseable."""
    backend = IntelASPBackend()
    result = backend.verify(_dev_only_blob())
    assert result.structural_parse_ok is False
    assert result.vendor != "intel-sgx"
    assert result.vendor != "intel-tdx"
    assert "intel" in (result.error or "").lower()


def test_intel_asp_rejects_short_blob():
    backend = IntelASPBackend()
    result = backend.verify(b"\x03\x00")  # version OK but truncated
    assert result.structural_parse_ok is False
    assert (
        "short" in (result.error or "").lower()
        or "truncated" in (result.error or "").lower()
        or "too short" in (result.error or "").lower()
    )


def test_intel_asp_rejects_unrecognized_version():
    backend = IntelASPBackend()
    blob = struct.pack("<H", 99) + b"\x00" * 300
    result = backend.verify(blob)
    assert result.structural_parse_ok is False


def test_intel_asp_handles_none_input():
    backend = IntelASPBackend()
    result = backend.verify(None)  # type: ignore
    assert result.structural_parse_ok is False
    assert result.vendor_verified is False


# ── DevOnlyBackend ───────────────────────────────────────


def test_dev_only_backend_recognizes_software_fallback():
    backend = DevOnlyBackend()
    result = backend.verify(_dev_only_blob())
    assert result.vendor == "software-fallback"
    assert result.vendor_verified is False
    assert result.structural_parse_ok is True
    assert "dev-only" in (
        result.vendor_data.get("note", "")
        + (result.error or "")
    ).lower()


def test_dev_only_backend_rejects_real_quote():
    backend = DevOnlyBackend()
    result = backend.verify(_sgx_v3_quote())
    # Not a dev-only blob → backend doesn't claim it
    assert result.vendor != "software-fallback"
    assert result.structural_parse_ok is False


# ── Registry / verify_attestation dispatcher ─────────────


def test_registry_default_includes_intel_and_dev_only():
    reg = AttestationBackendRegistry()
    vendors = {
        b.handles_vendor for b in reg.backends
    }
    assert "intel" in vendors  # IntelASPBackend handles both sgx + tdx
    assert "software-fallback" in vendors


def test_registry_dispatches_sgx_to_intel():
    reg = AttestationBackendRegistry()
    result = reg.verify(_sgx_v3_quote())
    assert result.vendor == "intel-sgx"
    assert result.structural_parse_ok is True


def test_registry_dispatches_tdx_to_intel():
    reg = AttestationBackendRegistry()
    result = reg.verify(_tdx_v4_quote())
    assert result.vendor == "intel-tdx"


def test_registry_dispatches_dev_only():
    reg = AttestationBackendRegistry()
    result = reg.verify(_dev_only_blob())
    assert result.vendor == "software-fallback"
    assert result.vendor_verified is False


def test_registry_unknown_vendor():
    reg = AttestationBackendRegistry()
    blob = struct.pack("<H", 99) + b"\x00" * 300
    result = reg.verify(blob)
    assert result.vendor == "unknown"
    assert result.structural_parse_ok is False


def test_registry_empty_input():
    reg = AttestationBackendRegistry()
    result = reg.verify(b"")
    assert result.vendor == "unknown"
    assert result.structural_parse_ok is False


def test_verify_attestation_top_level_helper():
    """Module-level helper backed by a default registry —
    convenience for callers that don't need to customize."""
    result = verify_attestation(_sgx_v3_quote())
    assert result.vendor == "intel-sgx"


# ── Custom backend registration ──────────────────────────


def test_custom_backend_registration():
    """Operators can register their own backend (e.g., for a
    proprietary TEE flavor)."""
    class CustomBackend:
        handles_vendor = "custom-tee"

        def verify(self, blob):
            if blob and blob.startswith(b"CUSTOM:"):
                return AttestationVerificationResult(
                    vendor="custom-tee",
                    vendor_verified=True,
                    vendor_data={"note": "custom backend hit"},
                    signature_chain_ok=True,
                    structural_parse_ok=True,
                )
            return AttestationVerificationResult(
                vendor="unknown",
                vendor_verified=False,
                vendor_data={},
                signature_chain_ok=False,
                error="not a custom blob",
                structural_parse_ok=False,
            )

    reg = AttestationBackendRegistry()
    reg.register(CustomBackend())
    result = reg.verify(b"CUSTOM:my-quote-bytes")
    assert result.vendor == "custom-tee"
    assert result.vendor_verified is True


# ── verify_receipt_privacy_claim integration ─────────────


def test_sprint292_verifier_populates_vendor_for_sgx():
    """Sprint 292's verify_receipt_privacy_claim should now
    populate vendor + vendor_verified fields on
    PrivacyVerification when sprint 293 backends parse the
    attestation."""
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt,
    )
    from prsm.compute.inference.privacy_verification import (
        verify_receipt_privacy_claim,
    )
    from prsm.compute.inference.receipt import sign_receipt
    from prsm.compute.tee.models import (
        PrivacyLevel, TEEType,
    )
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("sgx-test")
    receipt = InferenceReceipt(
        job_id="j", request_id="r",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SGX,
        tee_attestation=_sgx_v3_quote(),
        output_hash=hashlib.sha256(b"out").digest(),
        duration_seconds=0.1,
        cost_ftns="0.01",
        settler_signature=b"\x00" * 64,
        settler_node_id="",
    )
    receipt = sign_receipt(receipt, identity)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.hardware_attested is True
    # New sprint-293 fields surfaced
    assert result.attestation_vendor == "intel-sgx"
    assert (
        result.attestation_vendor_data.get(
            "mrenclave_hex"
        )
        is not None
    )


def test_sprint292_verifier_marks_dev_only_correctly():
    from prsm.compute.inference.models import (
        ContentTier, InferenceReceipt,
    )
    from prsm.compute.inference.privacy_verification import (
        verify_receipt_privacy_claim,
    )
    from prsm.compute.inference.receipt import sign_receipt
    from prsm.compute.tee.models import (
        PrivacyLevel, TEEType,
    )
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("sw-test")
    receipt = InferenceReceipt(
        job_id="j", request_id="r",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=_dev_only_blob(),
        output_hash=hashlib.sha256(b"out").digest(),
        duration_seconds=0.1,
        cost_ftns="0.01",
        settler_signature=b"\x00" * 64,
        settler_node_id="",
    )
    receipt = sign_receipt(receipt, identity)
    result = verify_receipt_privacy_claim(
        receipt, identity=identity,
    )
    assert result.hardware_attested is False
    assert result.attestation_vendor == "software-fallback"
