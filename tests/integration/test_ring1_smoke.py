"""
Ring 1 Smoke Test
=================

End-to-end test: detect hardware profile, load a WASM module,
execute in sandbox, verify resource metering.
"""

import pytest
from prsm.compute.wasm import (
    WasmtimeRuntime,
    ResourceLimits,
    ExecutionStatus,
    HardwareProfiler,
    ComputeTier,
)


# Minimal WASM: (module (func (export "run") (result i32) (i32.const 42)))
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing1Smoke:
    def test_hardware_profiler_produces_valid_profile(self):
        """Profiler should detect real hardware and classify a tier."""
        profiler = HardwareProfiler()
        profile = profiler.detect()

        assert profile.cpu_cores >= 1
        assert profile.ram_total_gb > 0
        assert profile.compute_tier in [
            ComputeTier.T1, ComputeTier.T2, ComputeTier.T3, ComputeTier.T4,
        ]
        assert profile.tflops_fp32 > 0

        # Serialization roundtrip
        d = profile.to_dict()
        assert isinstance(d["compute_tier"], str)
        assert isinstance(d["thermal_class"], str)

    @pytest.mark.skipif(
        not WasmtimeRuntime().available,
        reason="wasmtime not installed",
    )
    def test_wasm_sandbox_execution(self):
        """WASM module should execute in sandbox and return metered result."""
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)

        result = runtime.execute(
            module=module,
            input_data=b'{"test": true}',
            resource_limits=ResourceLimits(
                max_memory_bytes=64 * 1024 * 1024,
                max_execution_seconds=5,
            ),
        )

        assert result.status == ExecutionStatus.SUCCESS
        assert result.execution_time_seconds >= 0
        assert result.execution_time_seconds < 5.0
        assert result.pcu() >= 0

    @pytest.mark.skipif(
        not WasmtimeRuntime().available,
        reason="wasmtime not installed",
    )
    def test_full_pipeline_profile_then_execute(self):
        """Full Ring 1 flow: profile hardware, then execute WASM."""
        # Step 1: Profile
        profiler = HardwareProfiler()
        profile = profiler.detect()

        # Step 2: Configure limits based on profile
        limits = ResourceLimits(
            max_memory_bytes=min(
                256 * 1024 * 1024,
                int(profile.ram_available_gb * 0.1 * 1024 ** 3) if profile.ram_available_gb > 0 else 256 * 1024 * 1024,
            ),
            max_execution_seconds=30,
        )

        # Step 3: Execute
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(module, b"", limits)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.pcu() >= 0
