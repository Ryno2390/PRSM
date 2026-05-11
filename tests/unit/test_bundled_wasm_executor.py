"""Sprint 174 — bundled minimal WASM executor binary ships with PRSM.

QueryOrchestrator wiring (sprint 173) found a missing kwarg
``wasm_executor_binary`` on SwarmDispatcherAdapter. Sprint 173
added ``PRSM_WASM_EXECUTOR_PATH`` env. Sprint 174 bundles a
known-loadable minimal stub so a fresh node can wire QO
end-to-end without first building a custom executor.

The stub is the same 36-byte module used by test_wasm_runtime.py +
test_swarm_coordinator.py — it exports ``run`` returning i32(42).
Loads cleanly under Wasmtime, satisfies the binary-bytes contract
of SwarmDispatcherAdapter, and provides a smoke-test surface for
the dispatch pipeline. It does NOT interpret InstructionManifest;
production deployments must supply a real executor.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_minimal_executor_path_exists():
    """Sprint 174 — the bundled binary must ship with the package."""
    from prsm.compute.wasm.binaries import MINIMAL_EXECUTOR_PATH
    assert MINIMAL_EXECUTOR_PATH.is_file()


def test_minimal_executor_starts_with_wasm_magic():
    """Sprint 174 — binary file is a valid WASM module (magic bytes
    + version). Cheap structural integrity check."""
    from prsm.compute.wasm.binaries import load_minimal_executor
    data = load_minimal_executor()
    assert data[:4] == b"\x00asm"  # magic
    assert data[4:8] == b"\x01\x00\x00\x00"  # version 1


def test_minimal_executor_loadable_by_wasmtime():
    """Sprint 174 — the bundled binary must compile under the same
    Wasmtime runtime SwarmDispatcherAdapter uses. Without this the
    bundled fallback is useless."""
    wasmtime = pytest.importorskip("wasmtime")
    from prsm.compute.wasm.binaries import load_minimal_executor

    engine = wasmtime.Engine()
    module = wasmtime.Module(engine, load_minimal_executor())
    export_names = {e.name for e in module.exports}
    # Sprint 174 invariant — the stub exports `run` (bare-function
    # ABI per prsm/compute/wasm/runtime.py:127). Operators building
    # a real WASI executor would export `_start` instead.
    assert "run" in export_names


def test_minimal_executor_size_under_kilobyte():
    """Sprint 174 — the stub is tiny by design (smoke-test only).
    A bundled binary that grew to MB-scale would indicate
    packaging drift — likely someone replaced the stub with a real
    executor that should have used PRSM_WASM_EXECUTOR_PATH instead."""
    from prsm.compute.wasm.binaries import load_minimal_executor
    assert len(load_minimal_executor()) < 1024
