"""Tests for TEE data models (Ring 7)."""

import math
import pytest

from prsm.compute.tee.models import (
    TEEType,
    TEECapability,
    DPConfig,
    ConfidentialResult,
    PrivacyLevel,
    HARDWARE_TEE_TYPES,
)


# ── TEEType ──────────────────────────────────────────────────────────

class TestTEEType:
    def test_enum_values(self):
        assert TEEType.NONE.value == "none"
        assert TEEType.SGX.value == "sgx"
        assert TEEType.SOFTWARE.value == "software"

    def test_str_enum(self):
        assert TEEType.TDX == "tdx"
        assert isinstance(TEEType.SEV, str)

    def test_all_hardware_types_in_frozenset(self):
        for t in (TEEType.SGX, TEEType.TDX, TEEType.SEV, TEEType.TRUSTZONE, TEEType.SECURE_ENCLAVE):
            assert t.value in HARDWARE_TEE_TYPES


# ── TEECapability ────────────────────────────────────────────────────

class TestTEECapability:
    def test_defaults(self):
        cap = TEECapability()
        assert cap.tee_type == TEEType.NONE
        assert cap.max_enclave_memory_mb == 0
        assert not cap.available
        assert not cap.is_hardware_backed

    def test_available_when_not_none(self):
        cap = TEECapability(tee_type=TEEType.SOFTWARE)
        assert cap.available

    def test_hardware_backed(self):
        cap = TEECapability(tee_type=TEEType.SGX)
        assert cap.is_hardware_backed

    def test_software_not_hardware_backed(self):
        cap = TEECapability(tee_type=TEEType.SOFTWARE)
        assert not cap.is_hardware_backed

    def test_to_dict_from_dict_roundtrip(self):
        original = TEECapability(
            tee_type=TEEType.TDX,
            max_enclave_memory_mb=256,
            max_threads=4,
            attestation_supported=True,
        )
        d = original.to_dict()
        assert d["tee_type"] == "tdx"
        restored = TEECapability.from_dict(d)
        assert restored == original


# ── DPConfig ─────────────────────────────────────────────────────────

class TestDPConfig:
    def test_defaults(self):
        cfg = DPConfig()
        assert cfg.epsilon == 8.0
        assert cfg.delta == 1e-5
        assert cfg.clip_norm == 1.0
        assert cfg.noise_mechanism == "gaussian"

    def test_noise_scale_formula(self):
        cfg = DPConfig(epsilon=8.0, delta=1e-5, clip_norm=1.0)
        expected = 1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 8.0
        assert abs(cfg.noise_scale - expected) < 1e-10

    def test_noise_scale_zero_epsilon(self):
        cfg = DPConfig(epsilon=0.0)
        assert cfg.noise_scale == float("inf")

    def test_noise_scale_negative_epsilon(self):
        cfg = DPConfig(epsilon=-1.0)
        assert cfg.noise_scale == float("inf")


# ── PrivacyLevel ─────────────────────────────────────────────────────

class TestPrivacyLevel:
    def test_config_for_level_none(self):
        cfg = PrivacyLevel.config_for_level(PrivacyLevel.NONE)
        assert cfg.epsilon == float("inf")

    def test_config_for_level_standard(self):
        cfg = PrivacyLevel.config_for_level(PrivacyLevel.STANDARD)
        assert cfg.epsilon == 8.0

    def test_config_for_level_maximum(self):
        cfg = PrivacyLevel.config_for_level(PrivacyLevel.MAXIMUM)
        assert cfg.epsilon == 1.0


# ── ConfidentialResult ───────────────────────────────────────────────

class TestConfidentialResult:
    def test_defaults(self):
        r = ConfidentialResult()
        assert r.tee_type == TEEType.SOFTWARE
        assert not r.is_hardware_attested

    def test_hardware_attested(self):
        r = ConfidentialResult(
            tee_type=TEEType.SGX,
            attestation_proof="abc123",
        )
        assert r.is_hardware_attested

    def test_not_attested_without_proof(self):
        r = ConfidentialResult(tee_type=TEEType.SGX, attestation_proof=None)
        assert not r.is_hardware_attested

    def test_not_attested_with_empty_proof(self):
        r = ConfidentialResult(tee_type=TEEType.SGX, attestation_proof="")
        assert not r.is_hardware_attested
