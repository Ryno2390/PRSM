"""Sprint 294 — AMD KDS backend for SEV-SNP attestation reports.

Sprint 293 shipped the AttestationBackend interface + Intel
ASP stub for SGX (v3) + TDX (v4). This sprint adds AMD SEV-SNP
support via the AMDKDSBackend (mirrors KDS = Key Distribution
Service that signs SEV-SNP attestation reports).

SEV-SNP attestation report layout (subset, per AMD's "SEV
Secure Nested Paging Firmware ABI Specification"):
  bytes 0-3:    version (uint32 LE) = 2 (current spec)
  bytes 4-7:    guest_svn (uint32)
  bytes 8-15:   policy (uint64)
  bytes 16-31:  family_id (16 bytes)
  bytes 32-47:  image_id (16 bytes)
  bytes 48-51:  vmpl
  bytes 144-191: MEASUREMENT (48 bytes; sha384 of guest)
  bytes 320-383: REPORT_DATA (64 bytes; caller-supplied nonce)
  bytes 416-479: chip_id (64 bytes)

Vendor detection disambiguator:
  bytes [0:4] = b"\\x02\\x00\\x00\\x00" → AMD SEV-SNP (uint32
                                          version=2). Intel
                                          SGX/TDX use uint16
                                          version=3/4, so
                                          bytes [2:4] ≠ \\x00\\x00.

v1 ships structural parsing only — cryptographic verification
(SEV-SNP signing-chain to AMD KDS endorsement key + TCB
versioning) wires behind the same interface in sprint 297.
"""
from __future__ import annotations

import struct

import pytest

from prsm.compute.inference.attestation_backends import (
    AMDKDSBackend,
    AttestationBackendRegistry,
    detect_vendor,
    verify_attestation,
)


# ── Helpers: build synthetic SEV-SNP report ──────────────


def _sev_snp_report(
    measurement: bytes = b"\x55" * 48,
    report_data: bytes = b"\x66" * 64,
    chip_id: bytes = b"\x77" * 64,
    version: int = 2,
) -> bytes:
    """Build a structurally-valid SEV-SNP report.

    Total length is at least 480 bytes (CHIP_ID extends to
    offset 480 in our v1 parser). Real SEV-SNP reports are
    ~1184 bytes with a trailing 512-byte ECDSA-P384 signature;
    we don't need the signature bytes for structural parsing.
    """
    buf = bytearray(b"\x00" * 480)
    # version (uint32 LE)
    struct.pack_into("<I", buf, 0, version)
    # guest_svn = 5 (arbitrary; field exists for callers)
    struct.pack_into("<I", buf, 4, 5)
    # policy = arbitrary uint64
    struct.pack_into("<Q", buf, 8, 0x00000000_00000001)
    # measurement at offset 144 (48 bytes)
    buf[144:144 + 48] = measurement
    # report_data at offset 320 (64 bytes)
    buf[320:320 + 64] = report_data
    # chip_id at offset 416 (64 bytes)
    buf[416:416 + 64] = chip_id
    return bytes(buf)


# ── detect_vendor recognizes SEV-SNP ─────────────────────


def test_detect_vendor_sev_snp_v2():
    assert detect_vendor(_sev_snp_report()) == "amd-sev-snp"


def test_detect_vendor_distinguishes_sev_snp_from_intel():
    """SEV-SNP version=2 uint32 looks like SGX uint16
    version=2 in the first 2 bytes. The byte [2:4] zero
    check disambiguates."""
    # Intel SGX is uint16 version=3; this fake intel-shaped
    # blob has version_uint16=2 + a non-zero att_key_type
    # at bytes 2-3. Should NOT be SEV-SNP.
    fake_intel = struct.pack("<H", 2) + struct.pack("<H", 5)
    fake_intel += b"\x00" * 500
    # Not version=3/4 so it's not Intel either; not
    # SOFTWARE_TEE prefix; so it should be "unknown" (NOT
    # SEV-SNP).
    assert detect_vendor(fake_intel) != "amd-sev-snp"


def test_detect_vendor_sgx_unchanged():
    """Existing sprint-293 detection paths still work."""
    sgx_v3 = struct.pack("<H", 3) + b"\x00" * 300
    assert detect_vendor(sgx_v3) == "intel-sgx"


def test_detect_vendor_unsupported_sev_snp_version():
    """SEV-SNP version != 2 should not be claimed.
    Defends against accidental claim of forward-incompatible
    versions."""
    # version=99 uint32
    blob = struct.pack("<I", 99) + b"\x00" * 500
    assert detect_vendor(blob) != "amd-sev-snp"


# ── AMDKDSBackend.verify ─────────────────────────────────


def test_amd_kds_parses_sev_snp_report():
    backend = AMDKDSBackend()
    measurement = b"\xab" * 48
    report_data = b"\xcd" * 64
    chip_id = b"\xef" * 64
    report = _sev_snp_report(
        measurement=measurement,
        report_data=report_data,
        chip_id=chip_id,
    )
    result = backend.verify(report)
    assert result.vendor == "amd-sev-snp"
    assert result.structural_parse_ok is True
    # v1 stub: vendor_verified False until real KDS signing-
    # chain verification ships
    assert result.vendor_verified is False
    assert result.vendor_data["measurement_hex"] == "ab" * 48
    assert result.vendor_data["report_data_hex"] == "cd" * 64
    assert result.vendor_data["chip_id_hex"] == "ef" * 64
    assert result.vendor_data["version"] == 2
    assert result.vendor_data["guest_svn"] == 5


def test_amd_kds_rejects_intel_quote():
    """AMD backend doesn't claim Intel SGX/TDX quotes."""
    backend = AMDKDSBackend()
    sgx_blob = struct.pack("<H", 3) + b"\x00" * 500
    result = backend.verify(sgx_blob)
    assert result.structural_parse_ok is False
    assert result.vendor != "amd-sev-snp"


def test_amd_kds_rejects_short_blob():
    backend = AMDKDSBackend()
    # Has version=2 prefix but truncated
    blob = struct.pack("<I", 2) + b"\x00" * 50
    result = backend.verify(blob)
    assert result.structural_parse_ok is False
    assert (
        "short" in (result.error or "").lower()
        or "truncated" in (result.error or "").lower()
    )


def test_amd_kds_rejects_wrong_version():
    backend = AMDKDSBackend()
    blob = struct.pack("<I", 99) + b"\x00" * 500
    result = backend.verify(blob)
    assert result.structural_parse_ok is False


def test_amd_kds_handles_none_input():
    backend = AMDKDSBackend()
    result = backend.verify(None)  # type: ignore
    assert result.structural_parse_ok is False


def test_amd_kds_handles_empty_input():
    backend = AMDKDSBackend()
    result = backend.verify(b"")
    assert result.structural_parse_ok is False


# ── Registry integration ─────────────────────────────────


def test_registry_default_includes_amd():
    reg = AttestationBackendRegistry()
    vendors = {b.handles_vendor for b in reg.backends}
    assert "amd" in vendors


def test_registry_dispatches_sev_snp_to_amd():
    reg = AttestationBackendRegistry()
    result = reg.verify(_sev_snp_report())
    assert result.vendor == "amd-sev-snp"
    assert result.structural_parse_ok is True


def test_registry_intel_still_works():
    """Sprint-293 routing must not regress."""
    reg = AttestationBackendRegistry()
    sgx_blob = struct.pack("<H", 3) + b"\x00" * 300
    result = reg.verify(sgx_blob)
    # IntelASPBackend reports SGX truncated since 300 bytes
    # < _SGX_MIN_LEN. Important: AMDKDSBackend should NOT
    # have claimed it.
    assert result.vendor in ("intel-sgx", "unknown")
    assert result.vendor != "amd-sev-snp"


def test_module_level_verify_attestation_handles_sev_snp():
    result = verify_attestation(_sev_snp_report())
    assert result.vendor == "amd-sev-snp"


# ── PrivacyVerification integration ──────────────────────


def test_verify_receipt_with_sev_snp_attestation():
    """The sprint-292 verify_receipt_privacy_claim should
    now populate amd-sev-snp vendor + parsed measurement
    when the attestation is a SEV-SNP report."""
    import hashlib

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

    identity = generate_node_identity("sev-test")
    measurement = b"\x12" * 48
    receipt = InferenceReceipt(
        job_id="j", request_id="r",
        model_id="mock-llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SEV,
        tee_attestation=_sev_snp_report(
            measurement=measurement,
        ),
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
    assert result.attestation_vendor == "amd-sev-snp"
    assert (
        result.attestation_vendor_data["measurement_hex"]
        == "12" * 48
    )
