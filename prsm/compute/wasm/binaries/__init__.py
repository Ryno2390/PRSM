"""Bundled WASM executor binaries.

These are reference binaries that ship with PRSM so a fresh node can
construct ``SwarmDispatcherAdapter`` without requiring the operator
to first build a custom executor.

Current binaries:

- ``prsm_executor.wasm`` (~188 KB) — Sprint 177: real instruction-
  interpreting executor written in Rust + serde_json, compiled to
  wasm32-wasip1. Interprets ``InstructionManifest`` against tabular
  data (CSV / JSON / JSONL) inside a Wasmtime sandbox. Implements
  all 11 AgentOps mirrored from the Python ``DataProcessor``
  reference: COUNT, SUM, AVERAGE, FILTER, LIMIT, SELECT, SORT,
  GROUP_BY, AGGREGATE, COMPARE, TIME_SERIES. WASI command-module
  ABI: reads JSON from stdin, writes JSON to stdout.

  Wire contract::

    stdin:  {"manifest": <InstructionManifest>, "data": "<csv|json|jsonl>"}
    stdout: {"status": "success"|"error", "records": [...], "count": N, ...}

  Source at ``prsm/compute/wasm/source/executor/``. Built with
  ``cargo build --release --target wasm32-wasip1``.

- ``minimal_executor.wasm`` (36 bytes) — Sprint 174 smoke-test
  stub. Exports ``run`` returning i32(42). Retained for tests that
  exercise the bare-function ABI path in ``runtime.py``. NOT used
  by default node wiring anymore.

Production deployments can still override the bundled executor by
setting ``PRSM_WASM_EXECUTOR_PATH`` to a custom binary path. Use
cases: operators with non-tabular data formats, custom op support,
or stricter audit requirements.
"""
from __future__ import annotations

from pathlib import Path


_BINARIES_DIR = Path(__file__).resolve().parent

#: Sprint 177 — real instruction-interpreting executor. Default
#: binary returned by ``load_bundled_executor()``.
PRSM_EXECUTOR_PATH = _BINARIES_DIR / "prsm_executor.wasm"

#: Sprint 174 — 36-byte stub. Retained for tests; not used by
#: production wiring.
MINIMAL_EXECUTOR_PATH = _BINARIES_DIR / "minimal_executor.wasm"


def load_bundled_executor() -> bytes:
    """Return the bundled real executor binary as bytes.

    Sprint 177 — returns the Rust-built ``prsm_executor.wasm``.
    Raises ``RuntimeError`` on missing / empty file (indicates a
    corrupted install / packaging bug).
    """
    if not PRSM_EXECUTOR_PATH.is_file():
        raise RuntimeError(
            f"Bundled prsm_executor.wasm missing at "
            f"{PRSM_EXECUTOR_PATH!s}. Rebuild from source via "
            f"`cargo build --release --target wasm32-wasip1` in "
            f"prsm/compute/wasm/source/executor/, or reinstall "
            f"the prsm-network package."
        )
    data = PRSM_EXECUTOR_PATH.read_bytes()
    if not data:
        raise RuntimeError(
            f"Bundled prsm_executor.wasm is empty at "
            f"{PRSM_EXECUTOR_PATH!s}."
        )
    return data


def load_minimal_executor() -> bytes:
    """Return the bundled minimal stub binary (sprint 174).

    Retained for tests that exercise the bare-function ABI; not
    used by production wiring.
    """
    if not MINIMAL_EXECUTOR_PATH.is_file():
        raise RuntimeError(
            f"Bundled minimal_executor.wasm missing at "
            f"{MINIMAL_EXECUTOR_PATH!s}."
        )
    data = MINIMAL_EXECUTOR_PATH.read_bytes()
    if not data:
        raise RuntimeError(
            f"Bundled minimal_executor.wasm is empty at "
            f"{MINIMAL_EXECUTOR_PATH!s}."
        )
    return data


__all__ = [
    "PRSM_EXECUTOR_PATH",
    "MINIMAL_EXECUTOR_PATH",
    "load_bundled_executor",
    "load_minimal_executor",
]
