"""Ring 7 Smoke Test -- confidential compute verification."""

import pytest
import numpy as np
from prsm.compute.tee import TEEType, TEECapability, DPConfig, PrivacyLevel, ConfidentialResult, DPNoiseInjector
from prsm.compute.tee.runtime import SoftwareTEERuntime
from prsm.compute.tee.confidential_executor import ConfidentialExecutor
from prsm.compute.wasm.models import ResourceLimits

MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing7Smoke:
    def test_dp_noise_protects_activations(self):
        injector = DPNoiseInjector(DPConfig(epsilon=4.0))
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy = injector.inject(original)
        assert not np.array_equal(noisy, original)
        assert injector.epsilon_spent == 4.0

    def test_privacy_levels_produce_different_noise(self):
        standard = DPNoiseInjector.from_privacy_level(PrivacyLevel.STANDARD)
        maximum = DPNoiseInjector.from_privacy_level(PrivacyLevel.MAXIMUM)
        assert standard.config.noise_scale < maximum.config.noise_scale

    @pytest.mark.skipif(not SoftwareTEERuntime().available, reason="wasmtime not installed")
    def test_confidential_execution_pipeline(self):
        executor = ConfidentialExecutor(privacy_level=PrivacyLevel.STANDARD)
        result = executor.execute_confidential(MINIMAL_WASM, b"test", ResourceLimits(max_memory_bytes=64*1024*1024))
        assert result.dp_applied is True
        assert result.tee_type == TEEType.SOFTWARE
        assert result.execution_time_seconds >= 0

    def test_tee_capability_detection(self):
        from prsm.compute.wasm.profiler import HardwareProfiler
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert isinstance(profile.tee_available, bool)
        assert isinstance(profile.tee_type, str)

    def test_all_ring_1_through_7_imports(self):
        from prsm.compute.wasm import WASMRuntime, HardwareProfiler
        from prsm.compute.agents import AgentDispatcher, AgentExecutor
        from prsm.compute.swarm import SwarmCoordinator
        from prsm.economy.pricing import PricingEngine
        from prsm.economy.prosumer import ProsumerManager
        # Ring 5 AgentForge removed in v1.6.0 (legacy NWTN AGI framework pruned)
        from prsm.compute.tee import TEEType, DPConfig, ConfidentialResult, DPNoiseInjector
        from prsm.compute.tee.confidential_executor import ConfidentialExecutor
        assert all(x is not None for x in [WASMRuntime, AgentDispatcher, SwarmCoordinator, PricingEngine, ConfidentialExecutor])
