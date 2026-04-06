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

        PCU = (tflops x seconds) + (memory_gb x seconds) + egress_mb
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
