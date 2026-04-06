"""Tests for TEE runtime and confidential execution pipeline."""

import pytest
from unittest.mock import MagicMock
from prsm.compute.tee.runtime import TEERuntime, SoftwareTEERuntime
from prsm.compute.tee.confidential_executor import ConfidentialExecutor
from prsm.compute.tee.models import TEEType, PrivacyLevel
from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestTEERuntime:
    def test_software_implements_interface(self):
        assert isinstance(SoftwareTEERuntime(), TEERuntime)

    def test_software_name(self):
        assert SoftwareTEERuntime().name == "software"

    def test_software_tee_type(self):
        assert SoftwareTEERuntime().tee_type == TEEType.SOFTWARE

    def test_software_available(self):
        assert isinstance(SoftwareTEERuntime().available, bool)


class TestSoftwareExecution:
    @pytest.mark.skipif(not SoftwareTEERuntime().available, reason="wasmtime not installed")
    def test_execute(self):
        runtime = SoftwareTEERuntime()
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(module, b"test", ResourceLimits(max_memory_bytes=64*1024*1024))
        assert result.status == ExecutionStatus.SUCCESS


class TestConfidentialExecutor:
    def test_creation(self):
        executor = ConfidentialExecutor(privacy_level=PrivacyLevel.STANDARD)
        assert executor.privacy_level == PrivacyLevel.STANDARD

    @pytest.mark.skipif(not SoftwareTEERuntime().available, reason="wasmtime not installed")
    def test_confidential_execution(self):
        executor = ConfidentialExecutor(privacy_level=PrivacyLevel.STANDARD)
        result = executor.execute_confidential(MINIMAL_WASM, b"test", ResourceLimits(max_memory_bytes=64*1024*1024))
        assert result.tee_type == TEEType.SOFTWARE
        assert result.dp_applied is True
        assert result.epsilon_spent == 8.0

    def test_no_privacy_skips_dp(self):
        executor = ConfidentialExecutor(privacy_level=PrivacyLevel.NONE)
        assert executor.dp_config.epsilon == float("inf")


class TestHardwareProfileTEE:
    def test_profile_has_tee_fields(self):
        from prsm.compute.wasm.profiler_models import HardwareProfile
        profile = HardwareProfile()
        assert hasattr(profile, "tee_available")
        assert hasattr(profile, "tee_type")
        assert profile.tee_available is False

    def test_profile_to_dict_includes_tee(self):
        from prsm.compute.wasm.profiler_models import HardwareProfile
        profile = HardwareProfile(tee_available=True, tee_type="sgx")
        d = profile.to_dict()
        assert d["tee_available"] is True
        assert d["tee_type"] == "sgx"


class TestGossipTEE:
    def test_constant_exists(self):
        from prsm.node.gossip import GOSSIP_TEE_CAPABILITY
        assert GOSSIP_TEE_CAPABILITY == "tee_capability"

    def test_retention(self):
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS
        assert GOSSIP_RETENTION_SECONDS.get("tee_capability") == 86400
