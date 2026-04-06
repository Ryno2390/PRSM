"""
WASM Runtime Interface
======================

Abstract runtime interface and Wasmtime concrete implementation
for sandboxed WebAssembly execution of untrusted mobile agents.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Tuple

from prsm.compute.wasm.models import ExecutionResult, ExecutionStatus, ResourceLimits


class WASMRuntime(ABC):
    """Abstract interface for a WASM execution runtime."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable runtime name."""
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the runtime backend is installed and usable."""
        ...

    @abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Compile WASM bytes into a runnable module.

        Returns a runtime-specific handle that can be passed to ``execute``.
        """
        ...

    @abstractmethod
    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        """Execute a compiled module inside a sandbox.

        Parameters
        ----------
        module:
            The handle returned by :meth:`load`.
        input_data:
            Opaque bytes made available to the module via WASI stdin.
        resource_limits:
            Memory, time, and output caps enforced during execution.

        Returns
        -------
        ExecutionResult
        """
        ...


class WasmtimeRuntime(WASMRuntime):
    """Concrete runtime backed by the *wasmtime* Python package."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "wasmtime"

    @property
    def available(self) -> bool:
        try:
            import wasmtime as _wt  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, wasm_bytes: bytes) -> Tuple[Any, Any]:
        """Compile *wasm_bytes* into a ``wasmtime.Module``.

        Returns ``(engine, module)`` so that :meth:`execute` has access to
        the engine that produced the module.
        """
        import wasmtime

        try:
            config = wasmtime.Config()
            config.consume_fuel = True
            engine = wasmtime.Engine(config)
            module = wasmtime.Module(engine, wasm_bytes)
            return (engine, module)
        except Exception as exc:
            raise ValueError(f"Failed to compile WASM module: {exc}") from exc

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        module: Any,
        input_data: bytes,
        resource_limits: ResourceLimits,
    ) -> ExecutionResult:
        import wasmtime

        engine, wasm_module = module

        # ---- Store with fuel + memory limits ----
        store = wasmtime.Store(engine)
        fuel_budget = resource_limits.max_execution_seconds * 1_000_000_000
        store.set_fuel(fuel_budget)
        store.set_limits(memory_size=resource_limits.max_memory_bytes)

        # ---- Determine whether the module needs WASI ----
        export_names = [e.name for e in wasm_module.exports]
        uses_wasi = "_start" in export_names
        bare_run = "run" in export_names and not uses_wasi

        t0 = time.monotonic()

        try:
            if uses_wasi:
                # WASI command module — wire up stdin/stdout
                wasi_config = wasmtime.WasiConfig()
                wasi_config.stdin_bytes = input_data
                store.set_wasi(wasi_config)

                linker = wasmtime.Linker(engine)
                linker.define_wasi()
                instance = linker.instantiate(store, wasm_module)

                exports = instance.exports(store)
                start_fn = exports["_start"]
                start_fn(store)
                output = b""
            elif bare_run:
                # Bare module with a "run" export — no WASI needed
                instance = wasmtime.Instance(store, wasm_module, [])
                exports = instance.exports(store)
                run_fn = exports["run"]
                raw = run_fn(store)
                output = str(raw).encode() if raw is not None else b""
            else:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    output=b"",
                    execution_time_seconds=0.0,
                    memory_used_bytes=0,
                    error="Module has no '_start' or 'run' export",
                )

            elapsed = time.monotonic() - t0
            remaining_fuel = store.get_fuel()
            fuel_used = fuel_budget - remaining_fuel

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time_seconds=elapsed,
                memory_used_bytes=fuel_used // 1000,  # rough proxy
            )

        except wasmtime.WasmtimeError as exc:
            elapsed = time.monotonic() - t0
            err_msg = str(exc).lower()

            if "fuel" in err_msg:
                status = ExecutionStatus.TIMEOUT
            elif "memory" in err_msg or "grow" in err_msg:
                status = ExecutionStatus.OOM
            else:
                status = ExecutionStatus.ERROR

            return ExecutionResult(
                status=status,
                output=b"",
                execution_time_seconds=elapsed,
                memory_used_bytes=0,
                error=str(exc),
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                output=b"",
                execution_time_seconds=elapsed,
                memory_used_bytes=0,
                error=str(exc),
            )
