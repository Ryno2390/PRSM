"""Bundled WASM executor binaries.

These are reference / smoke-test binaries that ship with PRSM so a
fresh node can construct ``SwarmDispatcherAdapter`` without requiring
the operator to first build a custom executor.

Current binaries:

- ``minimal_executor.wasm`` (36 bytes) — exports ``run`` returning
  i32(42). Loads + executes under the Wasmtime runtime but does NOT
  interpret an ``InstructionManifest``. Suitable for:
  - Smoke-testing the dispatch pipeline end-to-end
  - QueryOrchestrator wiring verification (sprint 173 unblocked)
  - Demonstrations
  Not suitable for:
  - Production query execution (no instruction interpretation)
  - Any operator workflow that expects real shard processing

Production deployments should build a real executor and supply it
via ``PRSM_WASM_EXECUTOR_PATH``. See ``prsm/compute/wasm/runtime.py``
for the WASM-module ABI (``_start`` for WASI command modules,
``run`` for bare-function modules).
"""
from __future__ import annotations

from pathlib import Path


_BINARIES_DIR = Path(__file__).resolve().parent

#: Path to the minimal stub binary. Operators wiring without a custom
#: executor get this as the canonical fallback.
MINIMAL_EXECUTOR_PATH = _BINARIES_DIR / "minimal_executor.wasm"


def load_minimal_executor() -> bytes:
    """Read the bundled minimal executor binary as bytes.

    Raises ``RuntimeError`` if the binary file is missing or empty
    (indicates a corrupted install / packaging bug, not a normal
    operator-config problem).
    """
    if not MINIMAL_EXECUTOR_PATH.is_file():
        raise RuntimeError(
            f"Bundled minimal_executor.wasm missing at "
            f"{MINIMAL_EXECUTOR_PATH!s}. Reinstall the prsm-network "
            f"package or rebuild from source."
        )
    data = MINIMAL_EXECUTOR_PATH.read_bytes()
    if not data:
        raise RuntimeError(
            f"Bundled minimal_executor.wasm is empty at "
            f"{MINIMAL_EXECUTOR_PATH!s}. Indicates a packaging bug."
        )
    return data


__all__ = ["MINIMAL_EXECUTOR_PATH", "load_minimal_executor"]
