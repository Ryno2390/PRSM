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
