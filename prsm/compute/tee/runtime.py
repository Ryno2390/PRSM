"""
TEE Runtime Interface
=====================

Abstract interface for trusted execution environments.
SoftwareTEERuntime wraps the Ring 1 WASM sandbox as a fallback.
"""

import abc
import logging
from typing import Any

from prsm.compute.tee.models import TEEType
from prsm.compute.wasm.models import ExecutionResult, ResourceLimits

logger = logging.getLogger(__name__)


class TEERuntime(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Runtime name."""

    @property
    @abc.abstractmethod
    def tee_type(self) -> TEEType:
        """Type of TEE."""

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Whether available on current hardware."""

    @abc.abstractmethod
    def load(self, wasm_bytes: bytes) -> Any:
        """Load a WASM module."""

    @abc.abstractmethod
    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        """Execute in TEE sandbox."""


class SoftwareTEERuntime(TEERuntime):
    """Software-only TEE using Ring 1 WASM sandbox as fallback."""

    def __init__(self):
        self._wasm_runtime = None

    def _get_wasm_runtime(self):
        if self._wasm_runtime is None:
            from prsm.compute.wasm.runtime import WasmtimeRuntime
            self._wasm_runtime = WasmtimeRuntime()
        return self._wasm_runtime

    @property
    def name(self) -> str:
        return "software"

    @property
    def tee_type(self) -> TEEType:
        return TEEType.SOFTWARE

    @property
    def available(self) -> bool:
        return self._get_wasm_runtime().available

    def load(self, wasm_bytes: bytes) -> Any:
        return self._get_wasm_runtime().load(wasm_bytes)

    def execute(self, module: Any, input_data: bytes, resource_limits: ResourceLimits) -> ExecutionResult:
        return self._get_wasm_runtime().execute(module, input_data, resource_limits)
