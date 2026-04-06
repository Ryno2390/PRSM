"""Tests for WASM runtime data models and sandbox execution."""

import pytest
from decimal import Decimal

from prsm.compute.wasm.models import (
    ResourceLimits,
    ExecutionResult,
    ExecutionStatus,
    WASMModule,
)


class TestResourceLimits:
    def test_default_limits(self):
        limits = ResourceLimits()
        assert limits.max_memory_bytes == 256 * 1024 * 1024  # 256 MB
        assert limits.max_execution_seconds == 30
        assert limits.max_output_bytes == 10 * 1024 * 1024  # 10 MB

    def test_custom_limits(self):
        limits = ResourceLimits(
            max_memory_bytes=512 * 1024 * 1024,
            max_execution_seconds=60,
            max_output_bytes=20 * 1024 * 1024,
        )
        assert limits.max_memory_bytes == 512 * 1024 * 1024
        assert limits.max_execution_seconds == 60

    def test_limits_reject_zero_memory(self):
        with pytest.raises(ValueError, match="max_memory_bytes must be positive"):
            ResourceLimits(max_memory_bytes=0)

    def test_limits_reject_zero_time(self):
        with pytest.raises(ValueError, match="max_execution_seconds must be positive"):
            ResourceLimits(max_execution_seconds=0)


class TestExecutionResult:
    def test_successful_result(self):
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=b'{"answer": 42}',
            execution_time_seconds=1.5,
            memory_used_bytes=1024 * 1024,
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output == b'{"answer": 42}'
        assert result.error is None

    def test_failed_result(self):
        result = ExecutionResult(
            status=ExecutionStatus.ERROR,
            output=b"",
            execution_time_seconds=0.1,
            memory_used_bytes=0,
            error="Division by zero in WASM module",
        )
        assert result.status == ExecutionStatus.ERROR
        assert result.error is not None

    def test_timeout_result(self):
        result = ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            output=b"",
            execution_time_seconds=30.0,
            memory_used_bytes=256 * 1024 * 1024,
        )
        assert result.status == ExecutionStatus.TIMEOUT

    def test_pcu_calculation(self):
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=b"ok",
            execution_time_seconds=10.0,
            memory_used_bytes=1024 * 1024 * 1024,  # 1 GB
            tflops_used=5.0,
        )
        # PCU = tflops * seconds + memory_gb_seconds + egress_mb
        # = 5.0 * 10.0 + 1.0 * 10.0 + 0.0 = 60.0
        assert result.pcu() > 0


class TestWASMModule:
    def test_module_creation(self):
        wasm_bytes = b"\x00asm\x01\x00\x00\x00"  # Minimal WASM magic bytes
        module = WASMModule(
            module_id="test-module-123",
            wasm_bytes=wasm_bytes,
            entry_point="main",
        )
        assert module.module_id == "test-module-123"
        assert module.wasm_bytes == wasm_bytes
        assert module.size_bytes == len(wasm_bytes)

    def test_module_rejects_non_wasm(self):
        with pytest.raises(ValueError, match="Invalid WASM binary"):
            WASMModule(
                module_id="bad-module",
                wasm_bytes=b"this is not wasm",
                entry_point="main",
            )

    def test_module_rejects_oversized(self):
        # Default max is 5 MB
        big_bytes = b"\x00asm\x01\x00\x00\x00" + b"\x00" * (6 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds maximum"):
            WASMModule(
                module_id="big-module",
                wasm_bytes=big_bytes,
                entry_point="main",
            )
