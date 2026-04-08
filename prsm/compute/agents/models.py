"""
Mobile Agent Data Models
========================

Core data structures for WASM-based mobile agents that travel to data
instead of moving data to compute. Part of Ring 2 (Sovereign-Edge AI).

Classes:
    AgentManifest  - Declares what an agent needs (content IDs, hardware, limits)
    MobileAgent    - The agent itself (WASM binary + manifest + metadata)
    DispatchStatus - Lifecycle states for agent dispatch
    DispatchRecord - Tracks an agent's journey from origin to target node
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Constants ───────────────────────────────────────────────────────────

WASM_MAGIC = b"\x00asm"
DEFAULT_MAX_AGENT_SIZE = 5 * 1024 * 1024  # 5 MiB


# ── AgentManifest ───────────────────────────────────────────────────────

@dataclass
class AgentManifest:
    """Declares what a mobile agent needs to execute.

    The manifest travels with the agent and is checked by the target
    node before accepting the workload.
    """

    required_content_ids: List[str] = field(default_factory=list)
    min_hardware_tier: str = "t1"
    output_schema: Optional[Dict[str, Any]] = None
    max_memory_bytes: int = 256 * 1024 * 1024  # 256 MiB
    max_execution_seconds: int = 60
    max_output_bytes: int = 10 * 1024 * 1024  # 10 MiB
    required_capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "required_content_ids": list(self.required_content_ids),
            "min_hardware_tier": self.min_hardware_tier,
            "output_schema": self.output_schema,
            "max_memory_bytes": self.max_memory_bytes,
            "max_execution_seconds": self.max_execution_seconds,
            "max_output_bytes": self.max_output_bytes,
            "required_capabilities": list(self.required_capabilities),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AgentManifest:
        """Reconstruct from a dictionary."""
        return cls(
            required_content_ids=d.get("required_content_ids", []),
            min_hardware_tier=d.get("min_hardware_tier", "t1"),
            output_schema=d.get("output_schema"),
            max_memory_bytes=d.get("max_memory_bytes", 256 * 1024 * 1024),
            max_execution_seconds=d.get("max_execution_seconds", 60),
            max_output_bytes=d.get("max_output_bytes", 10 * 1024 * 1024),
            required_capabilities=d.get("required_capabilities", []),
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON representation."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# ── DispatchStatus ──────────────────────────────────────────────────────

class DispatchStatus(str, Enum):
    """Lifecycle states for a mobile agent dispatch."""

    PENDING = "pending"
    BIDDING = "bidding"
    TRANSFERRING = "transferring"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    REFUNDED = "refunded"


# ── MobileAgent ─────────────────────────────────────────────────────────

@dataclass
class MobileAgent:
    """A WASM-based mobile agent that travels to data.

    Validates the WASM binary on creation and enforces size limits.
    """

    agent_id: str
    wasm_binary: bytes
    manifest: AgentManifest
    origin_node: str
    signature: str
    ftns_budget: float
    ttl: int  # seconds
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.wasm_binary.startswith(WASM_MAGIC):
            raise ValueError(
                f"Invalid WASM binary for agent {self.agent_id}: "
                "missing magic bytes (\\x00asm)"
            )
        if len(self.wasm_binary) > DEFAULT_MAX_AGENT_SIZE:
            raise ValueError(
                f"WASM binary for agent {self.agent_id} "
                f"({len(self.wasm_binary)} bytes) exceeds maximum "
                f"allowed size ({DEFAULT_MAX_AGENT_SIZE} bytes)"
            )

    @property
    def size_bytes(self) -> int:
        """Size of the WASM binary in bytes."""
        return len(self.wasm_binary)

    def is_expired(self) -> bool:
        """Check whether the agent's TTL has elapsed."""
        return time.time() >= self.created_at + self.ttl

    def binary_hash(self) -> str:
        """SHA-256 hex digest of the WASM binary."""
        return hashlib.sha256(self.wasm_binary).hexdigest()


# ── DispatchRecord ──────────────────────────────────────────────────────

@dataclass
class DispatchRecord:
    """Tracks a mobile agent's dispatch from origin to target node."""

    agent_id: str
    origin_node: str
    target_node: str
    ftns_budget: float
    status: DispatchStatus = DispatchStatus.PENDING
    escrow_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    bids: List[Dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    dispatch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    _done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def done_event(self) -> asyncio.Event:
        """Event that is set when the dispatch reaches a terminal state."""
        return self._done_event
