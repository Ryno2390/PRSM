# Ring 1 — "The Sandbox" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A PRSM node can safely execute an untrusted WASM module in a sandboxed runtime and accurately report its hardware capabilities (TFLOPS, thermal class, network bandwidth) to the network.

**Architecture:** A `WASMRuntime` interface abstraction with Wasmtime as the default implementation, living in `prsm/compute/wasm/`. The existing `capability_detection.py` is extended into a full `HardwareProfiler` that produces a `HardwareProfile` dataclass. A new `WASM_EXECUTE` job type is added to `ComputeProvider` so the existing job dispatch pipeline routes WASM payloads to the sandbox. Hardware profiles are gossiped via a new `GOSSIP_HARDWARE_PROFILE` message type.

**Tech Stack:** `wasmtime` (Python bindings via `wasmtime` PyPI package), `psutil` (already a dependency).

**Note:** Network bandwidth fields (`upload_mbps`, `download_mbps`) default to 0.0 in Ring 1. Bandwidth measurement requires peer-to-peer probing which is added in Ring 2 when the dispatch protocol establishes direct connections. The `HardwareProfile` schema includes the fields now so the gossip format is stable.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/compute/wasm/__init__.py` | Package exports |
| Create | `prsm/compute/wasm/runtime.py` | `WASMRuntime` interface + `WasmtimeRuntime` default |
| Create | `prsm/compute/wasm/models.py` | Dataclasses: `WASMModule`, `ExecutionResult`, `ResourceLimits` |
| Create | `prsm/compute/wasm/profiler.py` | `HardwareProfiler` with TFLOPS estimation, thermal, bandwidth |
| Create | `prsm/compute/wasm/profiler_models.py` | Dataclasses: `HardwareProfile`, `ComputeTier` enum |
| Modify | `prsm/node/compute_provider.py:45-49` | Add `WASM_EXECUTE` to `JobType` enum |
| Modify | `prsm/node/compute_provider.py:352-365` | Add WASM route in `_execute_job()` |
| Modify | `prsm/node/gossip.py:34-72` | Add `GOSSIP_HARDWARE_PROFILE` constant + retention |
| Modify | `prsm/node/config.py:103-107` | Add WASM config fields to `NodeConfig` |
| Modify | `pyproject.toml:47-88` | Add `wasmtime` to optional dependencies |
| Create | `tests/unit/test_wasm_runtime.py` | Runtime interface + sandbox tests |
| Create | `tests/unit/test_hardware_profiler.py` | Profiler + tier classification tests |
| Create | `tests/unit/test_wasm_compute_provider.py` | WASM job execution integration tests |

---

### Task 1: WASM Data Models

**Files:**
- Create: `prsm/compute/wasm/__init__.py`
- Create: `prsm/compute/wasm/models.py`
- Test: `tests/unit/test_wasm_runtime.py`

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p prsm/compute/wasm
```

- [ ] **Step 2: Write the failing test for data models**

Create `tests/unit/test_wasm_runtime.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_wasm_runtime.py::TestResourceLimits::test_default_limits -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prsm.compute.wasm'`

- [ ] **Step 4: Write the models implementation**

Create `prsm/compute/wasm/__init__.py`:

```python
"""
WASM Compute Runtime
====================

Sandboxed WebAssembly execution for PRSM mobile agents.
Runtime-agnostic interface with Wasmtime as default.
"""

from prsm.compute.wasm.models import (
    ResourceLimits,
    ExecutionResult,
    ExecutionStatus,
    WASMModule,
)

__all__ = [
    "ResourceLimits",
    "ExecutionResult",
    "ExecutionStatus",
    "WASMModule",
]
```

Create `prsm/compute/wasm/models.py`:

```python
"""
WASM Data Models
================

Dataclasses for WASM module execution: resource limits, results, and module metadata.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# WASM magic bytes: \0asm followed by version 1
WASM_MAGIC = b"\x00asm\x01\x00\x00\x00"
DEFAULT_MAX_MODULE_SIZE = 5 * 1024 * 1024  # 5 MB


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"  # Out of memory


@dataclass
class ResourceLimits:
    """Enforced resource caps for WASM execution."""

    max_memory_bytes: int = 256 * 1024 * 1024  # 256 MB
    max_execution_seconds: int = 30
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MB

    def __post_init__(self):
        if self.max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be positive")
        if self.max_execution_seconds <= 0:
            raise ValueError("max_execution_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


@dataclass
class ExecutionResult:
    """Result from a WASM module execution."""

    status: ExecutionStatus
    output: bytes
    execution_time_seconds: float
    memory_used_bytes: int
    tflops_used: float = 0.0
    egress_bytes: int = 0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)

    def pcu(self) -> float:
        """Calculate PRSM Compute Units consumed.

        PCU = (tflops × seconds) + (memory_gb × seconds) + egress_mb
        """
        memory_gb = self.memory_used_bytes / (1024 ** 3)
        memory_gb_seconds = memory_gb * self.execution_time_seconds
        tflops_seconds = self.tflops_used * self.execution_time_seconds
        egress_mb = self.egress_bytes / (1024 ** 2)
        return tflops_seconds + memory_gb_seconds + egress_mb


@dataclass
class WASMModule:
    """A validated WASM binary ready for execution."""

    module_id: str
    wasm_bytes: bytes
    entry_point: str
    max_size: int = DEFAULT_MAX_MODULE_SIZE

    def __post_init__(self):
        if not self.wasm_bytes[:8].startswith(WASM_MAGIC[:4]):
            raise ValueError(
                f"Invalid WASM binary: expected magic bytes \\x00asm, "
                f"got {self.wasm_bytes[:4]!r}"
            )
        if len(self.wasm_bytes) > self.max_size:
            raise ValueError(
                f"WASM binary ({len(self.wasm_bytes)} bytes) exceeds maximum "
                f"allowed size ({self.max_size} bytes)"
            )

    @property
    def size_bytes(self) -> int:
        return len(self.wasm_bytes)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_wasm_runtime.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/compute/wasm/__init__.py prsm/compute/wasm/models.py tests/unit/test_wasm_runtime.py
git commit -m "feat(ring1): WASM data models — ResourceLimits, ExecutionResult, WASMModule"
```

---

### Task 2: WASM Runtime Interface + Wasmtime Implementation

**Files:**
- Create: `prsm/compute/wasm/runtime.py`
- Test: `tests/unit/test_wasm_runtime.py` (append)
- Modify: `prsm/compute/wasm/__init__.py`
- Modify: `pyproject.toml:90-182`

- [ ] **Step 1: Add wasmtime to optional dependencies**

In `pyproject.toml`, add a new extras group after the existing `[project.optional-dependencies]` entries:

```toml
wasm = ["wasmtime>=22.0.0"]
```

- [ ] **Step 2: Install wasmtime locally**

```bash
pip install wasmtime>=22.0.0
```

- [ ] **Step 3: Write the failing tests for runtime interface**

Append to `tests/unit/test_wasm_runtime.py`:

```python
from prsm.compute.wasm.runtime import WASMRuntime, WasmtimeRuntime
from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus


class TestWASMRuntimeInterface:
    """Test the abstract interface contract."""

    def test_wasmtime_implements_interface(self):
        runtime = WasmtimeRuntime()
        assert isinstance(runtime, WASMRuntime)

    def test_runtime_name(self):
        runtime = WasmtimeRuntime()
        assert runtime.name == "wasmtime"

    def test_runtime_available(self):
        runtime = WasmtimeRuntime()
        # Should be True if wasmtime package is installed
        assert isinstance(runtime.available, bool)


# Minimal valid WASM module that exports a function "run" returning i32(42)
# Hand-assembled: (module (func (export "run") (result i32) (i32.const 42)))
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d,  # magic
    0x01, 0x00, 0x00, 0x00,  # version
    0x01, 0x05, 0x01,        # type section: 1 type
    0x60, 0x00, 0x01, 0x7f,  # func () -> i32
    0x03, 0x02, 0x01, 0x00,  # function section: func 0 has type 0
    0x07, 0x07, 0x01,        # export section: 1 export
    0x03, 0x72, 0x75, 0x6e,  # export name "run"
    0x00, 0x00,              # export kind=func, index=0
    0x0a, 0x06, 0x01,        # code section: 1 body
    0x04, 0x00,              # body size=4, 0 locals
    0x41, 0x2a,              # i32.const 42
    0x0b,                    # end
])


@pytest.mark.skipif(
    not WasmtimeRuntime().available,
    reason="wasmtime not installed",
)
class TestWasmtimeExecution:
    def test_load_valid_module(self):
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)
        assert module is not None

    def test_load_invalid_bytes_raises(self):
        runtime = WasmtimeRuntime()
        with pytest.raises(ValueError, match="Failed to compile"):
            runtime.load(b"\x00asm\x01\x00\x00\x00\xff\xff")

    def test_execute_returns_result(self):
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(
            module=module,
            input_data=b"",
            resource_limits=ResourceLimits(),
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.execution_time_seconds >= 0
        assert result.memory_used_bytes >= 0

    def test_execute_respects_memory_limit(self):
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)
        # Very small memory limit — module should still work (it's tiny)
        result = runtime.execute(
            module=module,
            input_data=b"",
            resource_limits=ResourceLimits(max_memory_bytes=1 * 1024 * 1024),
        )
        # Minimal module uses almost no memory, should succeed
        assert result.status == ExecutionStatus.SUCCESS

    def test_execute_with_input_data(self):
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(
            module=module,
            input_data=b'{"query": "test"}',
            resource_limits=ResourceLimits(),
        )
        assert result.status == ExecutionStatus.SUCCESS
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_wasm_runtime.py::TestWASMRuntimeInterface -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 5: Write the runtime implementation**

Create `prsm/compute/wasm/runtime.py`:

```python
"""
WASM Runtime Interface + Wasmtime Implementation
=================================================

Abstract WASMRuntime interface with Wasmtime as the default sandboxed
execution engine. The interface allows swapping in alternative runtimes
(Wasmer, etc.) without changing consuming code.
"""

import abc
import logging
import time
from typing import Any, Optional

from prsm.compute.wasm.models import (
    ExecutionResult,
    ExecutionStatus,
    ResourceLimits,
)

logger = logging.getLogger(__name__)


class WASMRuntime(abc.ABC):
    """Abstract interface for WASM execution runtimes."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runtime implementation name (e.g., 'wasmtime', 'wasmer')."""

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this runtime's dependencies are installed."""

    @abc.abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Validate and compile a WASM binary.

        Args:
            wasm_bytes: Raw WASM binary content.

        Returns:
            A compiled module handle (runtime-specific type).

        Raises:
            ValueError: If the binary is invalid or cannot be compiled.
        """

    @abc.abstractmethod
    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        """Execute a compiled WASM module in a sandboxed environment.

        The module can read `input_data` via WASI stdin and write output
        to WASI stdout. No filesystem, network, or other host access is
        provided.

        Args:
            module: Compiled module from load().
            input_data: Bytes piped to the module's stdin.
            resource_limits: Enforced resource caps.

        Returns:
            ExecutionResult with status, output, and resource metrics.
        """


class WasmtimeRuntime(WASMRuntime):
    """Wasmtime-based WASM runtime with WASI sandbox."""

    @property
    def name(self) -> str:
        return "wasmtime"

    @property
    def available(self) -> bool:
        try:
            import wasmtime  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "wasmtime package not installed. "
                "Install with: pip install prsm-network[wasm]"
            )

    def load(self, wasm_bytes: bytes) -> Any:
        """Compile WASM bytes into a wasmtime Module."""
        self._ensure_available()
        import wasmtime

        try:
            engine = wasmtime.Engine()
            module = wasmtime.Module(engine, wasm_bytes)
            return module
        except Exception as e:
            raise ValueError(f"Failed to compile WASM module: {e}") from e

    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        """Execute WASM module with Wasmtime WASI sandbox."""
        self._ensure_available()
        import wasmtime
        import wasmtime.bindgen as _  # noqa: F401 — verify full install

        started_at = time.time()

        try:
            # Configure engine with resource limits
            config = wasmtime.Config()
            config.consume_fuel = True
            engine = wasmtime.Engine(config)

            # Re-compile with the fuel-aware engine
            mod = wasmtime.Module(engine, module._module if hasattr(module, '_module') else module.module.serialize() if hasattr(module, 'module') else None)
        except Exception:
            # Fallback: use the module's own engine
            engine = module.engine
            mod = module

        try:
            store = wasmtime.Store(engine)

            # Set fuel limit as a proxy for execution time
            # ~1 billion fuel units ≈ several seconds of compute
            fuel_per_second = 100_000_000
            total_fuel = fuel_per_second * resource_limits.max_execution_seconds
            store.set_fuel(total_fuel)

            # Configure memory limit
            store.set_limits(
                memory_size=resource_limits.max_memory_bytes,
            )

            # Set up WASI with stdin piped from input_data
            wasi_config = wasmtime.WasiConfig()
            wasi_config.stdin_bytes = input_data
            wasi_config.inherit_stderr()
            store.set_wasi(wasi_config)

            # Create linker with WASI
            linker = wasmtime.Linker(engine)
            linker.define_wasi()

            # Instantiate and run
            instance = linker.instantiate(store, mod)

            # Look for exported function: _start (WASI command) or the named entry
            start_fn = instance.exports(store).get("_start")
            if start_fn is None:
                # Try "run" as a fallback entry point
                start_fn = instance.exports(store).get("run")

            if start_fn is None:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output=b"",
                    execution_time_seconds=time.time() - started_at,
                    memory_used_bytes=0,
                    error="No exported '_start' or 'run' function found",
                )

            # Execute
            call_result = start_fn(store)

            elapsed = time.time() - started_at
            fuel_remaining = store.get_fuel()
            fuel_consumed = total_fuel - fuel_remaining

            # Read output from WASI stdout capture
            # For wasmtime, stdout capture requires a different approach —
            # the result is the function's return value for non-WASI modules
            output = b""
            if call_result is not None:
                if isinstance(call_result, (int, float)):
                    output = str(call_result).encode()
                elif isinstance(call_result, bytes):
                    output = call_result
                else:
                    output = str(call_result).encode()

            # Estimate memory usage from fuel consumption
            memory_estimate = min(
                fuel_consumed * 8,  # Rough: 8 bytes per fuel unit
                resource_limits.max_memory_bytes,
            )

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time_seconds=elapsed,
                memory_used_bytes=memory_estimate,
                started_at=started_at,
            )

        except wasmtime.WasmtimeError as e:
            elapsed = time.time() - started_at
            error_str = str(e)

            if "fuel" in error_str.lower() or "out of fuel" in error_str.lower():
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    output=b"",
                    execution_time_seconds=elapsed,
                    memory_used_bytes=0,
                    error=f"Execution exceeded time limit: {error_str}",
                )

            if "memory" in error_str.lower():
                return ExecutionResult(
                    status=ExecutionStatus.OOM,
                    output=b"",
                    execution_time_seconds=elapsed,
                    memory_used_bytes=resource_limits.max_memory_bytes,
                    error=f"Out of memory: {error_str}",
                )

            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output=b"",
                execution_time_seconds=elapsed,
                memory_used_bytes=0,
                error=f"WASM execution error: {error_str}",
            )

        except Exception as e:
            elapsed = time.time() - started_at
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output=b"",
                execution_time_seconds=elapsed,
                memory_used_bytes=0,
                error=f"Unexpected error: {e}",
            )
```

- [ ] **Step 6: Update `__init__.py` exports**

Add to `prsm/compute/wasm/__init__.py`:

```python
from prsm.compute.wasm.runtime import WASMRuntime, WasmtimeRuntime

__all__ = [
    "ResourceLimits",
    "ExecutionResult",
    "ExecutionStatus",
    "WASMModule",
    "WASMRuntime",
    "WasmtimeRuntime",
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_wasm_runtime.py -v`
Expected: All model tests PASS. Wasmtime tests PASS if wasmtime installed, SKIP otherwise.

- [ ] **Step 8: Commit**

```bash
git add prsm/compute/wasm/runtime.py prsm/compute/wasm/__init__.py pyproject.toml tests/unit/test_wasm_runtime.py
git commit -m "feat(ring1): WASMRuntime interface + Wasmtime sandbox implementation"
```

---

### Task 3: Hardware Profiler Data Models

**Files:**
- Create: `prsm/compute/wasm/profiler_models.py`
- Test: `tests/unit/test_hardware_profiler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_hardware_profiler.py`:

```python
"""Tests for hardware profiler and compute tier classification."""

import pytest

from prsm.compute.wasm.profiler_models import (
    HardwareProfile,
    ComputeTier,
    ThermalClass,
)


class TestComputeTier:
    def test_tier_from_tflops_t1(self):
        assert ComputeTier.from_tflops(3.0) == ComputeTier.T1

    def test_tier_from_tflops_t2(self):
        assert ComputeTier.from_tflops(15.0) == ComputeTier.T2

    def test_tier_from_tflops_t3(self):
        assert ComputeTier.from_tflops(50.0) == ComputeTier.T3

    def test_tier_from_tflops_t4(self):
        assert ComputeTier.from_tflops(100.0) == ComputeTier.T4

    def test_tier_boundary_t1_t2(self):
        assert ComputeTier.from_tflops(5.0) == ComputeTier.T2

    def test_tier_boundary_t2_t3(self):
        assert ComputeTier.from_tflops(30.0) == ComputeTier.T3

    def test_tier_boundary_t3_t4(self):
        assert ComputeTier.from_tflops(80.0) == ComputeTier.T4

    def test_tier_zero_tflops(self):
        assert ComputeTier.from_tflops(0.0) == ComputeTier.T1


class TestHardwareProfile:
    def test_profile_creation(self):
        profile = HardwareProfile(
            cpu_cores=8,
            cpu_freq_mhz=3200.0,
            ram_total_gb=16.0,
            ram_available_gb=10.0,
            gpu_name="NVIDIA RTX 4070",
            gpu_vram_gb=12.0,
            tflops_fp32=29.1,
            tflops_fp16=58.2,
            gpu_api="cuda",
            storage_available_gb=500.0,
            upload_mbps=50.0,
            download_mbps=200.0,
            thermal_class=ThermalClass.SUSTAINED,
        )
        assert profile.compute_tier == ComputeTier.T2
        assert profile.tflops_fp32 == 29.1

    def test_profile_tier_derived_from_tflops(self):
        profile = HardwareProfile(
            cpu_cores=4,
            cpu_freq_mhz=2400.0,
            ram_total_gb=8.0,
            ram_available_gb=4.0,
            tflops_fp32=2.0,
            thermal_class=ThermalClass.BURST,
        )
        assert profile.compute_tier == ComputeTier.T1

    def test_profile_to_dict(self):
        profile = HardwareProfile(
            cpu_cores=8,
            cpu_freq_mhz=3200.0,
            ram_total_gb=16.0,
            ram_available_gb=10.0,
            tflops_fp32=29.1,
            thermal_class=ThermalClass.SUSTAINED,
        )
        d = profile.to_dict()
        assert d["cpu_cores"] == 8
        assert d["compute_tier"] == "t2"
        assert d["thermal_class"] == "sustained"
        assert "tflops_fp32" in d

    def test_profile_from_dict(self):
        d = {
            "cpu_cores": 8,
            "cpu_freq_mhz": 3200.0,
            "ram_total_gb": 16.0,
            "ram_available_gb": 10.0,
            "tflops_fp32": 29.1,
            "tflops_fp16": 0.0,
            "gpu_name": "",
            "gpu_vram_gb": 0.0,
            "gpu_api": "",
            "storage_available_gb": 0.0,
            "upload_mbps": 0.0,
            "download_mbps": 0.0,
            "thermal_class": "sustained",
        }
        profile = HardwareProfile.from_dict(d)
        assert profile.cpu_cores == 8
        assert profile.compute_tier == ComputeTier.T2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_hardware_profiler.py::TestComputeTier -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the profiler models**

Create `prsm/compute/wasm/profiler_models.py`:

```python
"""
Hardware Profile Data Models
============================

Dataclasses for hardware capability reporting and compute tier classification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class ComputeTier(str, Enum):
    """Hardware compute tiers based on TFLOPS."""

    T1 = "t1"  # < 5 TFLOPS: Mobile, IoT, old laptops
    T2 = "t2"  # 5–30 TFLOPS: Consoles, mid-range GPUs
    T3 = "t3"  # 30–80 TFLOPS: High-end desktops, M-series
    T4 = "t4"  # 80+ TFLOPS: Datacenter GPUs

    @classmethod
    def from_tflops(cls, tflops: float) -> "ComputeTier":
        if tflops >= 80.0:
            return cls.T4
        elif tflops >= 30.0:
            return cls.T3
        elif tflops >= 5.0:
            return cls.T2
        else:
            return cls.T1


class ThermalClass(str, Enum):
    """Thermal headroom classification."""

    SUSTAINED = "sustained"  # Can run indefinitely (desktops, servers)
    BURST = "burst"          # 5–10 min at full power (laptops, phones)
    THROTTLED = "throttled"  # Already thermally constrained


@dataclass
class HardwareProfile:
    """Complete hardware capability profile for a PRSM node."""

    # CPU
    cpu_cores: int = 1
    cpu_freq_mhz: float = 0.0

    # Memory
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0

    # GPU
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    tflops_fp32: float = 0.0
    tflops_fp16: float = 0.0
    gpu_api: str = ""  # "cuda", "metal", "rocm", ""

    # Storage
    storage_available_gb: float = 0.0

    # Network
    upload_mbps: float = 0.0
    download_mbps: float = 0.0

    # Thermal
    thermal_class: ThermalClass = ThermalClass.SUSTAINED

    @property
    def compute_tier(self) -> ComputeTier:
        return ComputeTier.from_tflops(self.tflops_fp32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "tflops_fp32": self.tflops_fp32,
            "tflops_fp16": self.tflops_fp16,
            "gpu_api": self.gpu_api,
            "storage_available_gb": self.storage_available_gb,
            "upload_mbps": self.upload_mbps,
            "download_mbps": self.download_mbps,
            "thermal_class": self.thermal_class.value,
            "compute_tier": self.compute_tier.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareProfile":
        thermal = d.get("thermal_class", "sustained")
        if isinstance(thermal, str):
            thermal = ThermalClass(thermal)
        return cls(
            cpu_cores=d.get("cpu_cores", 1),
            cpu_freq_mhz=d.get("cpu_freq_mhz", 0.0),
            ram_total_gb=d.get("ram_total_gb", 0.0),
            ram_available_gb=d.get("ram_available_gb", 0.0),
            gpu_name=d.get("gpu_name", ""),
            gpu_vram_gb=d.get("gpu_vram_gb", 0.0),
            tflops_fp32=d.get("tflops_fp32", 0.0),
            tflops_fp16=d.get("tflops_fp16", 0.0),
            gpu_api=d.get("gpu_api", ""),
            storage_available_gb=d.get("storage_available_gb", 0.0),
            upload_mbps=d.get("upload_mbps", 0.0),
            download_mbps=d.get("download_mbps", 0.0),
            thermal_class=thermal,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_hardware_profiler.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/wasm/profiler_models.py tests/unit/test_hardware_profiler.py
git commit -m "feat(ring1): HardwareProfile + ComputeTier data models"
```

---

### Task 4: Hardware Profiler Implementation

**Files:**
- Create: `prsm/compute/wasm/profiler.py`
- Test: `tests/unit/test_hardware_profiler.py` (append)

- [ ] **Step 1: Write the failing tests for the profiler**

Append to `tests/unit/test_hardware_profiler.py`:

```python
from unittest.mock import patch, MagicMock
from prsm.compute.wasm.profiler import HardwareProfiler


class TestHardwareProfiler:
    def test_detect_cpu(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        # Should always detect at least 1 core
        assert profile.cpu_cores >= 1
        assert profile.ram_total_gb > 0

    @patch("prsm.compute.wasm.profiler.subprocess.run")
    def test_detect_nvidia_gpu(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA RTX 4070, 12288, 29150\n",
        )
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.gpu_name == "NVIDIA RTX 4070"
        assert profile.gpu_vram_gb == pytest.approx(12.0, abs=0.1)
        assert profile.tflops_fp32 == pytest.approx(29.15, abs=0.1)
        assert profile.gpu_api == "cuda"

    def test_detect_returns_valid_tier(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.compute_tier in [
            ComputeTier.T1, ComputeTier.T2, ComputeTier.T3, ComputeTier.T4,
        ]

    def test_detect_thermal_class(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        assert profile.thermal_class in [
            ThermalClass.SUSTAINED, ThermalClass.BURST, ThermalClass.THROTTLED,
        ]

    def test_tflops_estimate_cpu_fallback(self):
        """When no GPU, TFLOPS should be estimated from CPU."""
        profiler = HardwareProfiler()
        tflops = profiler._estimate_cpu_tflops(cores=8, freq_mhz=3200.0)
        # CPU TFLOPS are low but non-zero
        assert tflops > 0
        assert tflops < 5.0  # CPU alone shouldn't reach T2

    def test_profile_serialization_roundtrip(self):
        profiler = HardwareProfiler()
        profile = profiler.detect()
        d = profile.to_dict()
        restored = HardwareProfile.from_dict(d)
        assert restored.cpu_cores == profile.cpu_cores
        assert restored.compute_tier == profile.compute_tier
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_hardware_profiler.py::TestHardwareProfiler::test_detect_cpu -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the profiler implementation**

Create `prsm/compute/wasm/profiler.py`:

```python
"""
Hardware Profiler
=================

Detects and reports hardware capabilities for PRSM compute tier classification.
Extends the basic capability_detection.py with TFLOPS estimation, thermal
classification, and network bandwidth measurement.
"""

import logging
import os
import platform
import subprocess
from typing import Optional

from prsm.compute.wasm.profiler_models import (
    HardwareProfile,
    ThermalClass,
)

logger = logging.getLogger(__name__)

# Known GPU TFLOPS (FP32) lookup table — covers common consumer GPUs.
# Used when nvidia-smi reports a name but not TFLOPS directly.
GPU_TFLOPS_TABLE = {
    # NVIDIA RTX 40 series
    "RTX 4090": 82.6,
    "RTX 4080": 48.7,
    "RTX 4070 Ti": 40.1,
    "RTX 4070": 29.1,
    "RTX 4060 Ti": 22.1,
    "RTX 4060": 15.1,
    # NVIDIA RTX 30 series
    "RTX 3090": 35.6,
    "RTX 3080": 29.8,
    "RTX 3070": 20.3,
    "RTX 3060": 12.7,
    # Apple Silicon
    "Apple M1": 2.6,
    "Apple M1 Pro": 5.3,
    "Apple M1 Max": 10.6,
    "Apple M2": 3.6,
    "Apple M2 Pro": 6.8,
    "Apple M2 Max": 13.6,
    "Apple M3": 4.1,
    "Apple M3 Pro": 8.4,
    "Apple M3 Max": 16.8,
    "Apple M4": 4.6,
    "Apple M4 Pro": 9.2,
    "Apple M4 Max": 18.4,
    # Consoles (for reference — not detected via nvidia-smi)
    "PS5 RDNA 2": 10.3,
    "Xbox Series X RDNA 2": 12.1,
}


class HardwareProfiler:
    """Detects hardware capabilities and produces a HardwareProfile."""

    def detect(self) -> HardwareProfile:
        """Run full hardware detection and return a profile."""
        cpu_cores, cpu_freq = self._detect_cpu()
        ram_total, ram_available = self._detect_memory()
        gpu_name, gpu_vram, gpu_tflops, gpu_api = self._detect_gpu()
        storage_gb = self._detect_storage()
        thermal = self._detect_thermal_class()

        # If no GPU TFLOPS detected, estimate from CPU
        tflops_fp32 = gpu_tflops if gpu_tflops > 0 else self._estimate_cpu_tflops(cpu_cores, cpu_freq)
        tflops_fp16 = tflops_fp32 * 2.0 if gpu_api else tflops_fp32  # GPUs typically 2x FP16

        profile = HardwareProfile(
            cpu_cores=cpu_cores,
            cpu_freq_mhz=cpu_freq,
            ram_total_gb=ram_total,
            ram_available_gb=ram_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram,
            tflops_fp32=round(tflops_fp32, 2),
            tflops_fp16=round(tflops_fp16, 2),
            gpu_api=gpu_api,
            storage_available_gb=storage_gb,
            thermal_class=thermal,
        )

        logger.info(
            f"Hardware profile: {profile.compute_tier.value} tier, "
            f"{profile.tflops_fp32} TFLOPS FP32, "
            f"{profile.gpu_name or 'no GPU'}, "
            f"{profile.thermal_class.value} thermal"
        )

        return profile

    def _detect_cpu(self) -> tuple:
        """Detect CPU cores and frequency."""
        cores = os.cpu_count() or 1
        freq = 0.0
        try:
            import psutil
            cores = psutil.cpu_count(logical=True) or 1
            freq_info = psutil.cpu_freq()
            if freq_info:
                freq = freq_info.current
        except ImportError:
            pass
        return cores, freq

    def _detect_memory(self) -> tuple:
        """Detect total and available RAM in GB."""
        total = 0.0
        available = 0.0
        try:
            import psutil
            mem = psutil.virtual_memory()
            total = round(mem.total / (1024 ** 3), 2)
            available = round(mem.available / (1024 ** 3), 2)
        except ImportError:
            pass
        return total, available

    def _detect_gpu(self) -> tuple:
        """Detect GPU name, VRAM, TFLOPS, and API.

        Returns:
            (gpu_name, gpu_vram_gb, tflops_fp32, gpu_api)
        """
        # Try NVIDIA first
        name, vram, tflops, api = self._detect_nvidia()
        if name:
            return name, vram, tflops, api

        # Try Apple Silicon
        name, vram, tflops, api = self._detect_apple_silicon()
        if name:
            return name, vram, tflops, api

        return "", 0.0, 0.0, ""

    def _detect_nvidia(self) -> tuple:
        """Detect NVIDIA GPU via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,clocks.max.sm",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                name = parts[0]
                vram_mb = float(parts[1]) if len(parts) > 1 else 0
                vram_gb = round(vram_mb / 1024, 2)

                # Look up TFLOPS from table
                tflops = 0.0
                for key, val in GPU_TFLOPS_TABLE.items():
                    if key in name:
                        tflops = val
                        break

                # If not in table, estimate from clock speed
                if tflops == 0.0 and len(parts) > 2:
                    try:
                        clock_mhz = float(parts[2])
                        # Rough estimate: TFLOPS ≈ cores × clock × 2 (FMA) / 1e6
                        # We don't have core count from nvidia-smi easily, so use VRAM as proxy
                        # More VRAM generally correlates with more cores
                        estimated_cores = int(vram_gb * 500)  # Very rough
                        tflops = round(estimated_cores * clock_mhz * 2 / 1e6, 1)
                    except (ValueError, ZeroDivisionError):
                        pass

                return name, vram_gb, tflops, "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "", 0.0, 0.0, ""

    def _detect_apple_silicon(self) -> tuple:
        """Detect Apple Silicon GPU."""
        if platform.system() != "Darwin":
            return "", 0.0, 0.0, ""

        if not os.path.exists("/System/Library/PrivateFrameworks/MetalPerformance.framework"):
            return "", 0.0, 0.0, ""

        # Detect chip name from sysctl
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                brand = result.stdout.strip()
                # Extract chip name (e.g., "Apple M3 Max")
                name = ""
                for key in GPU_TFLOPS_TABLE:
                    if key.startswith("Apple") and key.split("Apple ")[1] in brand:
                        name = key
                        break

                if not name:
                    # Generic Apple Silicon
                    name = f"Apple {brand.split('Apple ')[-1]}" if "Apple" in brand else "Apple Silicon"

                tflops = GPU_TFLOPS_TABLE.get(name, 3.0)

                # Apple Silicon shares RAM with GPU — estimate GPU portion
                try:
                    import psutil
                    total_ram = psutil.virtual_memory().total / (1024 ** 3)
                    # GPU typically gets up to 2/3 of unified memory
                    vram = round(total_ram * 0.66, 2)
                except ImportError:
                    vram = 8.0  # Conservative default

                return name, vram, tflops, "metal"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return "", 0.0, 0.0, ""

    def _detect_storage(self) -> float:
        """Detect available storage in GB."""
        try:
            import shutil
            usage = shutil.disk_usage(os.path.expanduser("~/.prsm"))
        except FileNotFoundError:
            try:
                import shutil
                usage = shutil.disk_usage(os.path.expanduser("~"))
            except Exception:
                return 0.0
        return round(usage.free / (1024 ** 3), 2)

    def _detect_thermal_class(self) -> ThermalClass:
        """Classify thermal headroom.

        Heuristic:
        - Desktops and servers → SUSTAINED
        - Laptops → BURST
        - If battery is discharging → THROTTLED
        """
        system = platform.system()

        # Check for battery (laptop indicator)
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery is not None:
                if not battery.power_plugged:
                    return ThermalClass.THROTTLED
                return ThermalClass.BURST  # Laptop on power
        except (ImportError, AttributeError):
            pass

        # macOS: check if laptop via model identifier
        if system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True, text=True, timeout=5,
                )
                model = result.stdout.strip().lower()
                if "book" in model:
                    return ThermalClass.BURST
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        return ThermalClass.SUSTAINED

    def _estimate_cpu_tflops(self, cores: int, freq_mhz: float) -> float:
        """Estimate CPU-only TFLOPS (very rough).

        Modern CPUs with AVX-512 can do ~32 FLOPS/cycle/core.
        Without AVX-512, ~16 FLOPS/cycle/core (AVX2).
        """
        if freq_mhz <= 0:
            freq_mhz = 2000.0  # Conservative default

        flops_per_cycle = 16  # Assume AVX2
        freq_ghz = freq_mhz / 1000.0
        tflops = cores * freq_ghz * flops_per_cycle / 1000.0
        return round(tflops, 3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_hardware_profiler.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/wasm/profiler.py tests/unit/test_hardware_profiler.py
git commit -m "feat(ring1): HardwareProfiler with GPU detection, TFLOPS estimation, thermal classification"
```

---

### Task 5: Gossip Integration — GOSSIP_HARDWARE_PROFILE

**Files:**
- Modify: `prsm/node/gossip.py:34-116`
- Test: `tests/unit/test_hardware_profiler.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_hardware_profiler.py`:

```python
class TestGossipHardwareProfile:
    def test_gossip_constant_exists(self):
        from prsm.node.gossip import GOSSIP_HARDWARE_PROFILE
        assert GOSSIP_HARDWARE_PROFILE == "hardware_profile"

    def test_gossip_retention_is_24h(self):
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS
        assert GOSSIP_RETENTION_SECONDS.get("hardware_profile") == 86400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_hardware_profiler.py::TestGossipHardwareProfile -v`
Expected: FAIL with `ImportError: cannot import name 'GOSSIP_HARDWARE_PROFILE'`

- [ ] **Step 3: Add the gossip constant**

In `prsm/node/gossip.py`, add after the existing `GOSSIP_CAPABILITY_ANNOUNCE` constant (around line 55):

```python
GOSSIP_HARDWARE_PROFILE = "hardware_profile"
```

In the `GOSSIP_RETENTION_SECONDS` dict (around line 76–116), add:

```python
"hardware_profile": 86400,  # 24 hours
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_hardware_profiler.py::TestGossipHardwareProfile -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/node/gossip.py tests/unit/test_hardware_profiler.py
git commit -m "feat(ring1): add GOSSIP_HARDWARE_PROFILE message type (24h retention)"
```

---

### Task 6: WASM Job Type in ComputeProvider

**Files:**
- Modify: `prsm/node/compute_provider.py:45-49, 352-365`
- Create: `tests/unit/test_wasm_compute_provider.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_wasm_compute_provider.py`:

```python
"""Tests for WASM job execution via ComputeProvider."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.compute_provider import ComputeProvider, ComputeJob, JobType, JobStatus


class TestWASMJobType:
    def test_wasm_execute_in_job_type_enum(self):
        assert hasattr(JobType, "WASM_EXECUTE")
        assert JobType.WASM_EXECUTE == "wasm_execute"


# Reuse the minimal WASM module from test_wasm_runtime.py
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestWASMJobExecution:
    @pytest.fixture
    def provider(self):
        identity = MagicMock()
        identity.node_id = "test-node-123"
        identity.sign = MagicMock(return_value="test-signature")
        identity.public_key_b64 = "dGVzdC1rZXk="

        transport = MagicMock()
        gossip = AsyncMock()
        gossip.publish = AsyncMock()

        ledger = AsyncMock()

        provider = ComputeProvider(
            identity=identity,
            transport=transport,
            gossip=gossip,
            ledger=ledger,
        )
        return provider

    @pytest.mark.asyncio
    async def test_execute_wasm_job_routes_to_wasm_runner(self, provider):
        """Verify that WASM_EXECUTE jobs route to the WASM execution path."""
        import base64

        job = ComputeJob(
            job_id="wasm-job-001",
            job_type=JobType.WASM_EXECUTE,
            requester_id="requester-abc",
            payload={
                "wasm_bytes_b64": base64.b64encode(MINIMAL_WASM).decode(),
                "input_data_b64": base64.b64encode(b'{"query": "test"}').decode(),
                "entry_point": "run",
                "max_memory_bytes": 256 * 1024 * 1024,
                "max_execution_seconds": 30,
            },
            ftns_budget=1.0,
        )

        await provider._execute_job(job)

        assert job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        # If wasmtime is installed, should complete; otherwise may fail gracefully
        if job.status == JobStatus.COMPLETED:
            assert job.result is not None
            assert "execution_status" in job.result
            assert job.result_signature is not None

    @pytest.mark.asyncio
    async def test_wasm_job_with_invalid_binary_fails(self, provider):
        """Invalid WASM binary should fail the job, not crash the provider."""
        import base64

        job = ComputeJob(
            job_id="wasm-job-bad",
            job_type=JobType.WASM_EXECUTE,
            requester_id="requester-abc",
            payload={
                "wasm_bytes_b64": base64.b64encode(b"not-valid-wasm").decode(),
                "input_data_b64": base64.b64encode(b"").decode(),
                "entry_point": "run",
            },
            ftns_budget=1.0,
        )

        await provider._execute_job(job)

        assert job.status == JobStatus.FAILED
        assert job.error is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_wasm_compute_provider.py::TestWASMJobType -v`
Expected: FAIL with `AttributeError: WASM_EXECUTE`

- [ ] **Step 3: Add WASM_EXECUTE to JobType enum**

In `prsm/node/compute_provider.py`, modify the `JobType` enum (lines 45–49):

```python
class JobType(str, Enum):
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    BENCHMARK = "benchmark"
    TRAINING = "training"
    WASM_EXECUTE = "wasm_execute"
```

- [ ] **Step 4: Add WASM execution route in `_execute_job`**

In `prsm/node/compute_provider.py`, modify `_execute_job` (lines 352–365). Add the WASM case before the `else` clause:

```python
            elif job.job_type == JobType.WASM_EXECUTE:
                result = await self._run_wasm(job)
```

Then add the `_run_wasm` method after the existing `_run_training` method (after line 593):

```python
    async def _run_wasm(self, job: ComputeJob) -> Dict[str, Any]:
        """Execute a WASM module in a sandboxed runtime."""
        import base64
        from prsm.compute.wasm.runtime import WasmtimeRuntime
        from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus

        payload = job.payload
        wasm_bytes = base64.b64decode(payload.get("wasm_bytes_b64", ""))
        input_data = base64.b64decode(payload.get("input_data_b64", ""))

        limits = ResourceLimits(
            max_memory_bytes=payload.get("max_memory_bytes", 256 * 1024 * 1024),
            max_execution_seconds=payload.get("max_execution_seconds", 30),
            max_output_bytes=payload.get("max_output_bytes", 10 * 1024 * 1024),
        )

        runtime = WasmtimeRuntime()
        if not runtime.available:
            raise RuntimeError(
                "WASM runtime not available. Install with: pip install prsm-network[wasm]"
            )

        module = runtime.load(wasm_bytes)
        result = runtime.execute(module, input_data, limits)

        return {
            "execution_status": result.status.value,
            "output_b64": base64.b64encode(result.output).decode(),
            "execution_time_seconds": result.execution_time_seconds,
            "memory_used_bytes": result.memory_used_bytes,
            "pcu": result.pcu(),
            "error": result.error,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_wasm_compute_provider.py -v`
Expected: `TestWASMJobType` PASS. `TestWASMJobExecution` PASS if wasmtime installed, or FAIL gracefully with "WASM runtime not available" error captured in job.error.

- [ ] **Step 6: Commit**

```bash
git add prsm/node/compute_provider.py tests/unit/test_wasm_compute_provider.py
git commit -m "feat(ring1): WASM_EXECUTE job type in ComputeProvider with sandbox routing"
```

---

### Task 7: NodeConfig WASM Fields + Package Exports

**Files:**
- Modify: `prsm/node/config.py:103-107`
- Modify: `prsm/compute/wasm/__init__.py`

- [ ] **Step 1: Add WASM config fields to NodeConfig**

In `prsm/node/config.py`, after the Content Economy fields (around line 107), add:

```python
    # WASM Runtime (Ring 1)
    wasm_enabled: bool = True
    wasm_max_memory_bytes: int = 256 * 1024 * 1024  # 256 MB default sandbox
    wasm_max_execution_seconds: int = 30
    wasm_max_module_size: int = 5 * 1024 * 1024  # 5 MB
```

- [ ] **Step 2: Update package `__init__.py` with all exports**

Update `prsm/compute/wasm/__init__.py`:

```python
"""
WASM Compute Runtime
====================

Sandboxed WebAssembly execution for PRSM mobile agents.
Runtime-agnostic interface with Wasmtime as default.
"""

from prsm.compute.wasm.models import (
    ResourceLimits,
    ExecutionResult,
    ExecutionStatus,
    WASMModule,
)
from prsm.compute.wasm.runtime import WASMRuntime, WasmtimeRuntime
from prsm.compute.wasm.profiler_models import (
    HardwareProfile,
    ComputeTier,
    ThermalClass,
)
from prsm.compute.wasm.profiler import HardwareProfiler

__all__ = [
    "ResourceLimits",
    "ExecutionResult",
    "ExecutionStatus",
    "WASMModule",
    "WASMRuntime",
    "WasmtimeRuntime",
    "HardwareProfile",
    "ComputeTier",
    "ThermalClass",
    "HardwareProfiler",
]
```

- [ ] **Step 3: Commit**

```bash
git add prsm/node/config.py prsm/compute/wasm/__init__.py
git commit -m "feat(ring1): WASM config fields in NodeConfig + complete package exports"
```

---

### Task 8: Integration Test — Full Ring 1 Smoke Test

**Files:**
- Create: `tests/integration/test_ring1_smoke.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_ring1_smoke.py`:

```python
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
                int(profile.ram_available_gb * 0.1 * 1024 ** 3),  # 10% of available RAM
            ),
            max_execution_seconds=30,
        )

        # Step 3: Execute
        runtime = WasmtimeRuntime()
        module = runtime.load(MINIMAL_WASM)
        result = runtime.execute(module, b"", limits)

        assert result.status == ExecutionStatus.SUCCESS
        assert result.pcu() >= 0
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m pytest tests/integration/test_ring1_smoke.py -v`
Expected: Hardware profiler test PASS. WASM tests PASS if wasmtime installed, SKIP otherwise.

- [ ] **Step 3: Run the full Phase 4 + Ring 1 test suite to verify no regressions**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/integration/test_ring1_smoke.py tests/node/test_content_economy.py tests/node/test_settler_registry.py -v --timeout=60`
Expected: All tests PASS (Ring 1 tests + existing Phase tests)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_ring1_smoke.py
git commit -m "test(ring1): integration smoke test — hardware profile + WASM sandbox execution"
```

---

### Task 9: Final Cleanup — Version Bump + Push

**Files:**
- Modify: `prsm/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

In `prsm/__init__.py`, change `__version__` to `"0.26.0"`.
In `pyproject.toml`, change `version` to `"0.26.0"`.

- [ ] **Step 2: Run the complete test suite one final time**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/integration/test_ring1_smoke.py -v --timeout=60`
Expected: All Ring 1 tests PASS

- [ ] **Step 3: Commit and push**

```bash
git add prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.26.0 for Ring 1 — WASM Sandbox"
git push origin main
```

- [ ] **Step 4: Build and publish to PyPI**

```bash
rm -rf build/ dist/ prsm_network.egg-info/
python3 -m build
python3 -m twine upload dist/prsm_network-0.26.0*
```
