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
