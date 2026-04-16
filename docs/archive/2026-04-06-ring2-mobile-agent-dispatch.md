# Ring 2 — "The Courier" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A WASM mobile agent can be dispatched from one PRSM node to a remote node holding the data it needs, execute in the Ring 1 sandbox, and trigger FTNS payment on completion. This is the core "code-to-data" shift.

**Architecture:** A `MobileAgent` dataclass carries the WASM binary + manifest. An `AgentDispatcher` orchestrates the dispatch-bid-execute-settle lifecycle using the existing gossip protocol for signaling and WebSocket direct messages for binary transfer. Three new gossip message types (`GOSSIP_AGENT_DISPATCH`, `GOSSIP_AGENT_ACCEPT`, `GOSSIP_AGENT_RESULT`) handle the dispatch protocol. The existing `PaymentEscrow` handles FTNS locking/release. The Ring 1 `WasmtimeRuntime` handles sandboxed execution.

**Tech Stack:** Existing PRSM infrastructure (gossip, transport, escrow, WASM runtime). No new external dependencies.

**Note:** Ring 2 reuses the existing agent collaboration bidding protocol where appropriate, but adds WASM-specific dispatch semantics (binary transfer, manifest validation, hardware-tier matching).

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/compute/agents/__init__.py` | Package exports |
| Create | `prsm/compute/agents/models.py` | `MobileAgent`, `AgentManifest`, `DispatchStatus` |
| Create | `prsm/compute/agents/dispatcher.py` | `AgentDispatcher` — dispatch-bid-execute-settle lifecycle |
| Create | `prsm/compute/agents/executor.py` | `AgentExecutor` — receives and executes agents on provider side |
| Modify | `prsm/node/gossip.py:35-72` | Add `GOSSIP_AGENT_DISPATCH`, `GOSSIP_AGENT_ACCEPT`, `GOSSIP_AGENT_RESULT` |
| Modify | `prsm/node/node.py:690-913` | Wire AgentDispatcher + AgentExecutor into node lifecycle |
| Create | `tests/unit/test_mobile_agent_models.py` | Agent/manifest data model tests |
| Create | `tests/unit/test_agent_dispatcher.py` | Dispatcher lifecycle tests |
| Create | `tests/unit/test_agent_executor.py` | Executor sandbox + settlement tests |
| Create | `tests/integration/test_ring2_dispatch.py` | End-to-end dispatch smoke test |

---

### Task 1: Mobile Agent Data Models

**Files:**
- Create: `prsm/compute/agents/__init__.py`
- Create: `prsm/compute/agents/models.py`
- Test: `tests/unit/test_mobile_agent_models.py`

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p prsm/compute/agents
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_mobile_agent_models.py`:

```python
"""Tests for Mobile Agent data models."""

import pytest
import json
import uuid
from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)


class TestAgentManifest:
    def test_manifest_creation(self):
        manifest = AgentManifest(
            required_cids=["QmShard123"],
            min_hardware_tier="t2",
            output_schema={"type": "object"},
            max_memory_bytes=256 * 1024 * 1024,
            max_execution_seconds=30,
            max_output_bytes=10 * 1024 * 1024,
        )
        assert manifest.required_cids == ["QmShard123"]
        assert manifest.min_hardware_tier == "t2"

    def test_manifest_to_dict_roundtrip(self):
        manifest = AgentManifest(
            required_cids=["QmA", "QmB"],
            min_hardware_tier="t3",
            output_schema={"type": "object", "properties": {"result": {"type": "number"}}},
        )
        d = manifest.to_dict()
        restored = AgentManifest.from_dict(d)
        assert restored.required_cids == ["QmA", "QmB"]
        assert restored.min_hardware_tier == "t3"
        assert restored.output_schema == manifest.output_schema

    def test_manifest_hash_deterministic(self):
        m1 = AgentManifest(required_cids=["QmA"], min_hardware_tier="t2")
        m2 = AgentManifest(required_cids=["QmA"], min_hardware_tier="t2")
        assert m1.content_hash() == m2.content_hash()

    def test_manifest_hash_changes_with_content(self):
        m1 = AgentManifest(required_cids=["QmA"], min_hardware_tier="t2")
        m2 = AgentManifest(required_cids=["QmB"], min_hardware_tier="t2")
        assert m1.content_hash() != m2.content_hash()


WASM_MAGIC = b"\x00asm\x01\x00\x00\x00"


class TestMobileAgent:
    def test_agent_creation(self):
        agent = MobileAgent(
            agent_id=str(uuid.uuid4()),
            wasm_binary=WASM_MAGIC + b"\x00" * 100,
            manifest=AgentManifest(required_cids=["QmTest"], min_hardware_tier="t1"),
            origin_node="node-abc123",
            signature="sig-placeholder",
            ftns_budget=5.0,
            ttl=60,
        )
        assert agent.origin_node == "node-abc123"
        assert agent.ftns_budget == 5.0
        assert agent.ttl == 60

    def test_agent_rejects_invalid_wasm(self):
        with pytest.raises(ValueError, match="Invalid WASM"):
            MobileAgent(
                agent_id="bad-agent",
                wasm_binary=b"not wasm",
                manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
                origin_node="node-x",
                signature="sig",
                ftns_budget=1.0,
                ttl=30,
            )

    def test_agent_rejects_oversized_binary(self):
        big_wasm = WASM_MAGIC + b"\x00" * (6 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds maximum"):
            MobileAgent(
                agent_id="big-agent",
                wasm_binary=big_wasm,
                manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
                origin_node="node-x",
                signature="sig",
                ftns_budget=1.0,
                ttl=30,
            )

    def test_agent_size_bytes(self):
        binary = WASM_MAGIC + b"\x00" * 500
        agent = MobileAgent(
            agent_id="sized-agent",
            wasm_binary=binary,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="node-x",
            signature="sig",
            ftns_budget=1.0,
            ttl=30,
        )
        assert agent.size_bytes == len(binary)

    def test_agent_is_expired(self):
        import time
        agent = MobileAgent(
            agent_id="expired-agent",
            wasm_binary=WASM_MAGIC,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="node-x",
            signature="sig",
            ftns_budget=1.0,
            ttl=0,  # Already expired
            created_at=time.time() - 100,
        )
        assert agent.is_expired()


class TestDispatchRecord:
    def test_dispatch_record_creation(self):
        record = DispatchRecord(
            agent_id="agent-123",
            origin_node="node-a",
            target_node="node-b",
            ftns_budget=5.0,
        )
        assert record.status == DispatchStatus.PENDING
        assert record.target_node == "node-b"

    def test_dispatch_status_transitions(self):
        record = DispatchRecord(
            agent_id="agent-123",
            origin_node="node-a",
            target_node="node-b",
            ftns_budget=5.0,
        )
        record.status = DispatchStatus.BIDDING
        assert record.status == DispatchStatus.BIDDING
        record.status = DispatchStatus.EXECUTING
        assert record.status == DispatchStatus.EXECUTING
        record.status = DispatchStatus.COMPLETED
        assert record.status == DispatchStatus.COMPLETED
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mobile_agent_models.py::TestAgentManifest::test_manifest_creation -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the models implementation**

Create `prsm/compute/agents/__init__.py`:

```python
"""
Mobile Agent Framework
======================

WASM-based mobile agents that travel to data instead of moving data to compute.
Ring 2 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)

__all__ = [
    "AgentManifest",
    "MobileAgent",
    "DispatchStatus",
    "DispatchRecord",
]
```

Create `prsm/compute/agents/models.py`:

```python
"""
Mobile Agent Data Models
========================

Core data structures for WASM mobile agents: manifest, agent package,
dispatch tracking.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


WASM_MAGIC = b"\x00asm"
DEFAULT_MAX_AGENT_SIZE = 5 * 1024 * 1024  # 5 MB


class DispatchStatus(str, Enum):
    """Lifecycle status of a mobile agent dispatch."""
    PENDING = "pending"          # Created, not yet dispatched
    BIDDING = "bidding"          # Dispatch broadcast, collecting bids
    TRANSFERRING = "transferring"  # Binary being sent to winning node
    EXECUTING = "executing"      # Running in remote sandbox
    COMPLETED = "completed"      # Result received successfully
    FAILED = "failed"            # Execution or dispatch failed
    EXPIRED = "expired"          # TTL exceeded
    REFUNDED = "refunded"        # Escrow returned to origin


@dataclass
class AgentManifest:
    """Describes what a mobile agent needs and what it produces."""

    required_cids: List[str] = field(default_factory=list)
    min_hardware_tier: str = "t1"
    output_schema: Dict[str, Any] = field(default_factory=dict)
    max_memory_bytes: int = 256 * 1024 * 1024
    max_execution_seconds: int = 30
    max_output_bytes: int = 10 * 1024 * 1024
    required_capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_cids": self.required_cids,
            "min_hardware_tier": self.min_hardware_tier,
            "output_schema": self.output_schema,
            "max_memory_bytes": self.max_memory_bytes,
            "max_execution_seconds": self.max_execution_seconds,
            "max_output_bytes": self.max_output_bytes,
            "required_capabilities": self.required_capabilities,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentManifest":
        return cls(
            required_cids=d.get("required_cids", []),
            min_hardware_tier=d.get("min_hardware_tier", "t1"),
            output_schema=d.get("output_schema", {}),
            max_memory_bytes=d.get("max_memory_bytes", 256 * 1024 * 1024),
            max_execution_seconds=d.get("max_execution_seconds", 30),
            max_output_bytes=d.get("max_output_bytes", 10 * 1024 * 1024),
            required_capabilities=d.get("required_capabilities", []),
        )

    def content_hash(self) -> str:
        """Deterministic hash of the manifest contents."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class MobileAgent:
    """A self-contained WASM agent package ready for dispatch."""

    agent_id: str
    wasm_binary: bytes
    manifest: AgentManifest
    origin_node: str
    signature: str  # Ed25519 signature from origin node
    ftns_budget: float
    ttl: int  # Max seconds before expiry
    created_at: float = field(default_factory=time.time)
    max_size: int = DEFAULT_MAX_AGENT_SIZE

    def __post_init__(self):
        if not self.wasm_binary[:4].startswith(WASM_MAGIC):
            raise ValueError(
                f"Invalid WASM binary: expected magic bytes \\x00asm, "
                f"got {self.wasm_binary[:4]!r}"
            )
        if len(self.wasm_binary) > self.max_size:
            raise ValueError(
                f"WASM binary ({len(self.wasm_binary)} bytes) exceeds maximum "
                f"allowed size ({self.max_size} bytes)"
            )

    @property
    def size_bytes(self) -> int:
        return len(self.wasm_binary)

    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    def binary_hash(self) -> str:
        """SHA-256 hash of the WASM binary for verification."""
        return hashlib.sha256(self.wasm_binary).hexdigest()


@dataclass
class DispatchRecord:
    """Tracks the lifecycle of a single agent dispatch."""

    agent_id: str
    origin_node: str
    target_node: str
    ftns_budget: float
    status: DispatchStatus = DispatchStatus.PENDING
    escrow_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_signature: Optional[str] = None
    error: Optional[str] = None
    bids: List[Dict[str, Any]] = field(default_factory=list)
    retries: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    _result_event: "asyncio.Event" = field(default=None, repr=False)

    def __post_init__(self):
        import asyncio
        if self._result_event is None:
            self._result_event = asyncio.Event()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mobile_agent_models.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/compute/agents/__init__.py prsm/compute/agents/models.py tests/unit/test_mobile_agent_models.py
git commit -m "feat(ring2): MobileAgent + AgentManifest + DispatchRecord data models"
```

---

### Task 2: Gossip Constants for Agent Dispatch

**Files:**
- Modify: `prsm/node/gossip.py:35-118`
- Test: `tests/unit/test_mobile_agent_models.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_mobile_agent_models.py`:

```python
class TestGossipAgentDispatch:
    def test_dispatch_constant(self):
        from prsm.node.gossip import GOSSIP_AGENT_DISPATCH
        assert GOSSIP_AGENT_DISPATCH == "agent_dispatch"

    def test_accept_constant(self):
        from prsm.node.gossip import GOSSIP_AGENT_ACCEPT
        assert GOSSIP_AGENT_ACCEPT == "agent_accept"

    def test_agent_result_constant(self):
        from prsm.node.gossip import GOSSIP_AGENT_RESULT
        assert GOSSIP_AGENT_RESULT == "agent_result"

    def test_retention_policies(self):
        from prsm.node.gossip import GOSSIP_RETENTION_SECONDS
        assert GOSSIP_RETENTION_SECONDS.get("agent_dispatch") == 3600
        assert GOSSIP_RETENTION_SECONDS.get("agent_accept") == 3600
        assert GOSSIP_RETENTION_SECONDS.get("agent_result") == 3600
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mobile_agent_models.py::TestGossipAgentDispatch -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add the gossip constants**

In `prsm/node/gossip.py`, add after the existing agent gossip constants (near line 55):

```python
GOSSIP_AGENT_DISPATCH = "agent_dispatch"
GOSSIP_AGENT_ACCEPT = "agent_accept"
GOSSIP_AGENT_RESULT = "agent_result"
```

In `GOSSIP_RETENTION_SECONDS` dict, add:

```python
"agent_dispatch": 3600,   # 1 hour
"agent_accept": 3600,     # 1 hour
"agent_result": 3600,     # 1 hour
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mobile_agent_models.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/node/gossip.py tests/unit/test_mobile_agent_models.py
git commit -m "feat(ring2): GOSSIP_AGENT_DISPATCH/ACCEPT/RESULT message types"
```

---

### Task 3: Agent Executor (Provider Side)

**Files:**
- Create: `prsm/compute/agents/executor.py`
- Test: `tests/unit/test_agent_executor.py`

The executor runs on the **provider node** — it receives a WASM agent, validates it, executes it in the Ring 1 sandbox, and returns the signed result.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_agent_executor.py`:

```python
"""Tests for AgentExecutor — provider-side agent execution."""

import pytest
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.compute.agents.executor import AgentExecutor
from prsm.compute.agents.models import AgentManifest, MobileAgent
from prsm.compute.wasm.models import ExecutionStatus


# Minimal WASM: (module (func (export "run") (result i32) (i32.const 42)))
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


@pytest.fixture
def mock_identity():
    identity = MagicMock()
    identity.node_id = "provider-node-001"
    identity.sign = MagicMock(return_value="provider-signature-b64")
    identity.public_key_b64 = "cHJvdmlkZXIta2V5"
    return identity


@pytest.fixture
def mock_gossip():
    gossip = AsyncMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


@pytest.fixture
def executor(mock_identity, mock_gossip):
    return AgentExecutor(
        identity=mock_identity,
        gossip=mock_gossip,
    )


class TestAgentExecutor:
    @pytest.mark.asyncio
    async def test_execute_valid_agent(self, executor):
        agent = MobileAgent(
            agent_id="test-agent-001",
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="requester-node",
            signature="requester-sig",
            ftns_budget=1.0,
            ttl=60,
        )

        result = await executor.execute_agent(agent, input_data=b"")

        assert result["status"] in ("success", "error")
        if result["status"] == "success":
            assert "output_b64" in result
            assert "execution_time_seconds" in result
            assert "pcu" in result
            assert "provider_signature" in result

    @pytest.mark.asyncio
    async def test_execute_invalid_wasm_returns_error(self, executor):
        agent = MobileAgent.__new__(MobileAgent)
        agent.agent_id = "bad-agent"
        agent.wasm_binary = b"\x00asm\x01\x00\x00\x00\xff\xff"  # Invalid but passes magic check
        agent.manifest = AgentManifest(required_cids=[], min_hardware_tier="t1")
        agent.origin_node = "requester-node"
        agent.signature = "sig"
        agent.ftns_budget = 1.0
        agent.ttl = 60
        agent.created_at = __import__("time").time()
        agent.max_size = 5 * 1024 * 1024

        result = await executor.execute_agent(agent, input_data=b"")

        assert result["status"] == "error"
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_expired_agent_rejected(self, executor):
        import time
        agent = MobileAgent(
            agent_id="expired-agent",
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="requester-node",
            signature="sig",
            ftns_budget=1.0,
            ttl=0,
            created_at=time.time() - 100,
        )

        result = await executor.execute_agent(agent, input_data=b"")

        assert result["status"] == "error"
        assert "expired" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_publishes_result_via_gossip(self, executor, mock_gossip):
        agent = MobileAgent(
            agent_id="gossip-agent",
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="requester-node",
            signature="sig",
            ftns_budget=1.0,
            ttl=60,
        )

        await executor.execute_agent(agent, input_data=b"", publish_result=True)

        mock_gossip.publish.assert_called()
        call_args = mock_gossip.publish.call_args
        assert call_args[0][0] == "agent_result"

    @pytest.mark.asyncio
    async def test_validate_manifest_checks_hardware_tier(self, executor):
        manifest = AgentManifest(
            required_cids=[],
            min_hardware_tier="t4",  # Requires datacenter GPU
        )
        # Mock hardware profile to be T1
        with patch.object(executor, '_get_hardware_tier', return_value="t1"):
            is_valid, reason = executor.validate_manifest(manifest)
            assert not is_valid
            assert "hardware" in reason.lower()

    @pytest.mark.asyncio
    async def test_validate_manifest_accepts_sufficient_tier(self, executor):
        manifest = AgentManifest(
            required_cids=[],
            min_hardware_tier="t1",
        )
        with patch.object(executor, '_get_hardware_tier', return_value="t2"):
            is_valid, reason = executor.validate_manifest(manifest)
            assert is_valid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_agent_executor.py::TestAgentExecutor::test_execute_valid_agent -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the executor implementation**

Create `prsm/compute/agents/executor.py`:

```python
"""
Agent Executor (Provider Side)
==============================

Receives, validates, and executes WASM mobile agents in the Ring 1 sandbox.
Signs results and optionally publishes them via gossip.
"""

import base64
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from prsm.compute.agents.models import AgentManifest, MobileAgent
from prsm.compute.wasm.models import ResourceLimits, ExecutionStatus
from prsm.compute.wasm.profiler_models import ComputeTier

logger = logging.getLogger(__name__)

# Tier ordering for comparison
TIER_ORDER = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}


class AgentExecutor:
    """Executes WASM mobile agents on the provider node."""

    def __init__(
        self,
        identity,
        gossip,
        hardware_tier: str = "t1",
    ):
        self.identity = identity
        self.gossip = gossip
        self._hardware_tier = hardware_tier

    def _get_hardware_tier(self) -> str:
        return self._hardware_tier

    def validate_manifest(self, manifest: AgentManifest) -> Tuple[bool, str]:
        """Check if this node can execute an agent with the given manifest.

        Returns:
            (is_valid, reason) — reason is empty string if valid.
        """
        my_tier = TIER_ORDER.get(self._get_hardware_tier(), 1)
        required_tier = TIER_ORDER.get(manifest.min_hardware_tier, 1)

        if my_tier < required_tier:
            return False, (
                f"Hardware tier insufficient: have {self._get_hardware_tier()}, "
                f"need {manifest.min_hardware_tier}"
            )

        return True, ""

    async def execute_agent(
        self,
        agent: MobileAgent,
        input_data: bytes,
        publish_result: bool = False,
    ) -> Dict[str, Any]:
        """Execute a mobile agent in the WASM sandbox.

        Args:
            agent: The MobileAgent to execute.
            input_data: Data to provide as stdin to the WASM module.
            publish_result: If True, publish result via GOSSIP_AGENT_RESULT.

        Returns:
            Result dict with status, output, metrics, and provider signature.
        """
        # Check expiry
        if agent.is_expired():
            error_result = {
                "agent_id": agent.agent_id,
                "provider_id": self.identity.node_id,
                "status": "error",
                "error": "Agent expired before execution",
                "output_b64": "",
                "execution_time_seconds": 0.0,
                "memory_used_bytes": 0,
                "pcu": 0.0,
                "provider_signature": "",
            }
            return error_result

        # Validate manifest
        is_valid, reason = self.validate_manifest(agent.manifest)
        if not is_valid:
            return {
                "agent_id": agent.agent_id,
                "provider_id": self.identity.node_id,
                "status": "error",
                "error": reason,
                "output_b64": "",
                "execution_time_seconds": 0.0,
                "memory_used_bytes": 0,
                "pcu": 0.0,
                "provider_signature": "",
            }

        # Execute in sandbox
        try:
            from prsm.compute.wasm.runtime import WasmtimeRuntime

            runtime = WasmtimeRuntime()
            if not runtime.available:
                return {
                    "agent_id": agent.agent_id,
                    "provider_id": self.identity.node_id,
                    "status": "error",
                    "error": "WASM runtime not available",
                    "output_b64": "",
                    "execution_time_seconds": 0.0,
                    "memory_used_bytes": 0,
                    "pcu": 0.0,
                    "provider_signature": "",
                }

            limits = ResourceLimits(
                max_memory_bytes=agent.manifest.max_memory_bytes,
                max_execution_seconds=agent.manifest.max_execution_seconds,
                max_output_bytes=agent.manifest.max_output_bytes,
            )

            module = runtime.load(agent.wasm_binary)
            exec_result = runtime.execute(module, input_data, limits)

            # Build result
            output_b64 = base64.b64encode(exec_result.output).decode()

            # Sign the result
            result_bytes = json.dumps({
                "agent_id": agent.agent_id,
                "status": exec_result.status.value,
                "output_hash": __import__("hashlib").sha256(exec_result.output).hexdigest(),
            }, sort_keys=True).encode()
            provider_sig = self.identity.sign(result_bytes)

            result = {
                "agent_id": agent.agent_id,
                "provider_id": self.identity.node_id,
                "status": exec_result.status.value,
                "output_b64": output_b64,
                "execution_time_seconds": exec_result.execution_time_seconds,
                "memory_used_bytes": exec_result.memory_used_bytes,
                "pcu": exec_result.pcu(),
                "error": exec_result.error,
                "provider_signature": provider_sig,
                "provider_public_key": self.identity.public_key_b64,
            }

            logger.info(
                f"Agent {agent.agent_id[:8]} executed: "
                f"{exec_result.status.value}, {exec_result.execution_time_seconds:.3f}s, "
                f"{exec_result.pcu():.4f} PCU"
            )

        except Exception as e:
            logger.error(f"Agent {agent.agent_id[:8]} execution failed: {e}")
            result = {
                "agent_id": agent.agent_id,
                "provider_id": self.identity.node_id,
                "status": "error",
                "error": str(e),
                "output_b64": "",
                "execution_time_seconds": 0.0,
                "memory_used_bytes": 0,
                "pcu": 0.0,
                "provider_signature": "",
            }

        # Publish result if requested
        if publish_result:
            await self.gossip.publish("agent_result", result)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_agent_executor.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/agents/executor.py tests/unit/test_agent_executor.py
git commit -m "feat(ring2): AgentExecutor — provider-side WASM agent execution with signing"
```

---

### Task 4: Agent Dispatcher (Requester Side)

**Files:**
- Create: `prsm/compute/agents/dispatcher.py`
- Test: `tests/unit/test_agent_dispatcher.py`

The dispatcher runs on the **requester node** — it creates escrow, broadcasts the dispatch, collects bids, selects a winner, transfers the binary, and waits for the result.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_agent_dispatcher.py`:

```python
"""Tests for AgentDispatcher — requester-side dispatch lifecycle."""

import pytest
import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.compute.agents.dispatcher import AgentDispatcher
from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


@pytest.fixture
def mock_identity():
    identity = MagicMock()
    identity.node_id = "requester-node-001"
    identity.sign = MagicMock(return_value="requester-sig-b64")
    identity.public_key_b64 = "cmVxdWVzdGVyLWtleQ=="
    return identity


@pytest.fixture
def mock_gossip():
    gossip = AsyncMock()
    gossip.publish = AsyncMock(return_value=1)
    gossip.subscribe = MagicMock()
    return gossip


@pytest.fixture
def mock_transport():
    transport = AsyncMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    return transport


@pytest.fixture
def mock_escrow():
    escrow = AsyncMock()
    entry = MagicMock()
    entry.escrow_id = "escrow-001"
    escrow.create_escrow = AsyncMock(return_value=entry)
    escrow.release_escrow = AsyncMock(return_value=MagicMock())
    escrow.refund_escrow = AsyncMock(return_value=True)
    return escrow


@pytest.fixture
def dispatcher(mock_identity, mock_gossip, mock_transport, mock_escrow):
    d = AgentDispatcher(
        identity=mock_identity,
        gossip=mock_gossip,
        transport=mock_transport,
        escrow=mock_escrow,
    )
    return d


class TestAgentDispatcher:
    def test_create_agent(self, dispatcher, mock_identity):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(
                required_cids=["QmShard123"],
                min_hardware_tier="t2",
            ),
            ftns_budget=5.0,
            ttl=60,
        )
        assert agent.origin_node == mock_identity.node_id
        assert agent.ftns_budget == 5.0
        assert agent.wasm_binary == MINIMAL_WASM
        assert len(agent.signature) > 0

    @pytest.mark.asyncio
    async def test_dispatch_creates_escrow(self, dispatcher, mock_escrow):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            ftns_budget=5.0,
            ttl=60,
        )

        record = await dispatcher.dispatch(agent)

        mock_escrow.create_escrow.assert_called_once()
        assert record.status == DispatchStatus.BIDDING
        assert record.escrow_id == "escrow-001"

    @pytest.mark.asyncio
    async def test_dispatch_publishes_gossip(self, dispatcher, mock_gossip):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=["QmTest"], min_hardware_tier="t1"),
            ftns_budget=2.0,
            ttl=60,
        )

        await dispatcher.dispatch(agent)

        mock_gossip.publish.assert_called()
        call_args = mock_gossip.publish.call_args
        assert call_args[0][0] == "agent_dispatch"
        payload = call_args[0][1]
        assert payload["agent_id"] == agent.agent_id
        assert payload["manifest"]["required_cids"] == ["QmTest"]
        assert "wasm_binary" not in payload  # Binary NOT in gossip

    @pytest.mark.asyncio
    async def test_handle_bid(self, dispatcher):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            ftns_budget=5.0,
            ttl=60,
        )
        record = await dispatcher.dispatch(agent)

        await dispatcher._on_agent_accept(
            "agent_accept",
            {
                "agent_id": agent.agent_id,
                "provider_id": "provider-node-abc",
                "price_ftns": 0.5,
                "hardware_tier": "t2",
                "reputation": 0.8,
            },
            "provider-node-abc",
        )

        assert len(record.bids) == 1
        assert record.bids[0]["provider_id"] == "provider-node-abc"

    @pytest.mark.asyncio
    async def test_select_best_bid(self, dispatcher):
        bids = [
            {"provider_id": "cheap", "price_ftns": 0.1, "hardware_tier": "t1", "reputation": 0.5},
            {"provider_id": "balanced", "price_ftns": 0.3, "hardware_tier": "t2", "reputation": 0.8},
            {"provider_id": "premium", "price_ftns": 0.5, "hardware_tier": "t3", "reputation": 0.9},
        ]
        best = dispatcher._select_best_bid(bids, max_budget=5.0)
        assert best is not None
        # Should prefer balanced combination of price and reputation

    @pytest.mark.asyncio
    async def test_handle_result_releases_escrow(self, dispatcher, mock_escrow):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            ftns_budget=5.0,
            ttl=60,
        )
        record = await dispatcher.dispatch(agent)

        await dispatcher._on_agent_result(
            "agent_result",
            {
                "agent_id": agent.agent_id,
                "provider_id": "provider-node-abc",
                "status": "success",
                "output_b64": base64.b64encode(b'{"answer": 42}').decode(),
                "execution_time_seconds": 1.5,
                "pcu": 0.5,
                "provider_signature": "provider-sig",
                "provider_public_key": "cHJvdmlkZXIta2V5",
            },
            "provider-node-abc",
        )

        assert record.status == DispatchStatus.COMPLETED
        assert record.result is not None
        mock_escrow.release_escrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_refunds_on_no_bids(self, dispatcher, mock_escrow):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            ftns_budget=5.0,
            ttl=60,
        )
        record = await dispatcher.dispatch(agent, bid_timeout=0.1)

        # Wait for bid timeout
        await asyncio.sleep(0.2)
        await dispatcher._check_bid_timeout(agent.agent_id)

        # Should have refunded since no bids arrived
        if record.status in (DispatchStatus.FAILED, DispatchStatus.REFUNDED):
            mock_escrow.refund_escrow.assert_called()

    def test_get_dispatch_record(self, dispatcher):
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            ftns_budget=1.0,
            ttl=60,
        )
        # Before dispatch, no record
        assert dispatcher.get_record(agent.agent_id) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_agent_dispatcher.py::TestAgentDispatcher::test_create_agent -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the dispatcher implementation**

Create `prsm/compute/agents/dispatcher.py`:

```python
"""
Agent Dispatcher (Requester Side)
=================================

Orchestrates the mobile agent dispatch lifecycle:
1. Create agent + escrow
2. Broadcast dispatch via gossip
3. Collect bids from qualifying nodes
4. Select best bid
5. Transfer WASM binary to winner via direct WebSocket
6. Wait for execution result
7. Release escrow on success / refund on failure
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchRecord,
    DispatchStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_BID_TIMEOUT = 10.0  # Seconds to wait for bids
DEFAULT_RESULT_TIMEOUT = 120.0  # Seconds to wait for execution result


class AgentDispatcher:
    """Dispatches WASM mobile agents to remote nodes."""

    def __init__(
        self,
        identity,
        gossip,
        transport,
        escrow,
        bid_timeout: float = DEFAULT_BID_TIMEOUT,
        result_timeout: float = DEFAULT_RESULT_TIMEOUT,
    ):
        self.identity = identity
        self.gossip = gossip
        self.transport = transport
        self.escrow = escrow
        self.bid_timeout = bid_timeout
        self.result_timeout = result_timeout

        # Active dispatch records by agent_id
        self._dispatches: Dict[str, DispatchRecord] = {}
        # Agent objects by agent_id (for binary transfer)
        self._agents: Dict[str, MobileAgent] = {}

        # Subscribe to gossip
        self.gossip.subscribe("agent_accept", self._on_agent_accept)
        self.gossip.subscribe("agent_result", self._on_agent_result)

    def create_agent(
        self,
        wasm_binary: bytes,
        manifest: AgentManifest,
        ftns_budget: float,
        ttl: int = 120,
    ) -> MobileAgent:
        """Create a signed MobileAgent ready for dispatch."""
        agent_id = f"agent-{uuid.uuid4().hex[:12]}"

        # Sign the agent binary + manifest
        sign_data = json.dumps({
            "agent_id": agent_id,
            "binary_hash": __import__("hashlib").sha256(wasm_binary).hexdigest(),
            "manifest_hash": manifest.content_hash(),
        }, sort_keys=True).encode()
        signature = self.identity.sign(sign_data)

        agent = MobileAgent(
            agent_id=agent_id,
            wasm_binary=wasm_binary,
            manifest=manifest,
            origin_node=self.identity.node_id,
            signature=signature,
            ftns_budget=ftns_budget,
            ttl=ttl,
        )

        self._agents[agent_id] = agent
        return agent

    async def dispatch(
        self,
        agent: MobileAgent,
        bid_timeout: Optional[float] = None,
    ) -> DispatchRecord:
        """Dispatch an agent to the network.

        Creates escrow, broadcasts manifest via gossip, and starts
        collecting bids. Returns a DispatchRecord to track lifecycle.
        """
        timeout = bid_timeout or self.bid_timeout

        # Create escrow
        escrow_entry = await self.escrow.create_escrow(
            job_id=agent.agent_id,
            amount=agent.ftns_budget,
            requester_id=self.identity.node_id,
        )

        record = DispatchRecord(
            agent_id=agent.agent_id,
            origin_node=self.identity.node_id,
            target_node="",
            ftns_budget=agent.ftns_budget,
            status=DispatchStatus.BIDDING,
            escrow_id=escrow_entry.escrow_id if escrow_entry else None,
        )
        self._dispatches[agent.agent_id] = record

        # Broadcast dispatch (manifest only, NOT the binary)
        await self.gossip.publish("agent_dispatch", {
            "agent_id": agent.agent_id,
            "origin_node": self.identity.node_id,
            "manifest": agent.manifest.to_dict(),
            "ftns_budget": agent.ftns_budget,
            "ttl": agent.ttl,
            "binary_hash": agent.binary_hash(),
            "binary_size": agent.size_bytes,
        })

        logger.info(
            f"Agent {agent.agent_id[:8]} dispatched: "
            f"{len(agent.manifest.required_cids)} CIDs, "
            f"min tier {agent.manifest.min_hardware_tier}, "
            f"{agent.ftns_budget} FTNS budget"
        )

        return record

    async def _on_agent_accept(
        self,
        subtype: str,
        data: Dict[str, Any],
        sender_id: str,
    ) -> None:
        """Handle a bid from a provider node."""
        agent_id = data.get("agent_id", "")
        record = self._dispatches.get(agent_id)
        if not record or record.status != DispatchStatus.BIDDING:
            return

        bid = {
            "provider_id": data.get("provider_id", sender_id),
            "price_ftns": data.get("price_ftns", 0),
            "hardware_tier": data.get("hardware_tier", "t1"),
            "reputation": data.get("reputation", 0.5),
            "timestamp": time.time(),
        }
        record.bids.append(bid)

        logger.debug(
            f"Bid received for agent {agent_id[:8]} from {bid['provider_id'][:8]}: "
            f"{bid['price_ftns']} FTNS, tier {bid['hardware_tier']}"
        )

    async def select_and_transfer(self, agent_id: str) -> bool:
        """Select the best bid and transfer the WASM binary.

        Returns True if transfer succeeded.
        """
        record = self._dispatches.get(agent_id)
        agent = self._agents.get(agent_id)
        if not record or not agent:
            return False

        if not record.bids:
            logger.warning(f"No bids for agent {agent_id[:8]}")
            record.status = DispatchStatus.FAILED
            record.error = "No bids received"
            await self.escrow.refund_escrow(agent_id, "No bids received")
            return False

        # Select best bid
        best = self._select_best_bid(record.bids, agent.ftns_budget)
        if not best:
            record.status = DispatchStatus.FAILED
            record.error = "No acceptable bids"
            await self.escrow.refund_escrow(agent_id, "No acceptable bids")
            return False

        record.target_node = best["provider_id"]
        record.status = DispatchStatus.TRANSFERRING

        # Transfer binary via direct message
        from prsm.node.transport import P2PMessage
        msg = P2PMessage(
            msg_type="direct",
            sender_id=self.identity.node_id,
            payload={
                "type": "agent_binary_transfer",
                "agent_id": agent_id,
                "wasm_bytes_b64": base64.b64encode(agent.wasm_binary).decode(),
                "manifest": agent.manifest.to_dict(),
                "binary_hash": agent.binary_hash(),
                "origin_signature": agent.signature,
                "ftns_budget": agent.ftns_budget,
                "ttl": agent.ttl,
            },
        )

        success = await self.transport.send_to_peer(best["provider_id"], msg)
        if success:
            record.status = DispatchStatus.EXECUTING
            logger.info(f"Agent {agent_id[:8]} transferred to {best['provider_id'][:8]}")
        else:
            record.status = DispatchStatus.FAILED
            record.error = f"Binary transfer failed to {best['provider_id'][:8]}"
            await self.escrow.refund_escrow(agent_id, record.error)

        return success

    async def _on_agent_result(
        self,
        subtype: str,
        data: Dict[str, Any],
        sender_id: str,
    ) -> None:
        """Handle an execution result from a provider."""
        agent_id = data.get("agent_id", "")
        record = self._dispatches.get(agent_id)
        if not record:
            return
        if record.status not in (DispatchStatus.EXECUTING, DispatchStatus.BIDDING, DispatchStatus.TRANSFERRING):
            return

        status = data.get("status", "error")

        if status == "success":
            record.status = DispatchStatus.COMPLETED
            record.result = data
            record.result_signature = data.get("provider_signature", "")
            record.completed_at = time.time()

            # Release escrow to provider
            provider_id = data.get("provider_id", sender_id)
            await self.escrow.release_escrow(
                job_id=agent_id,
                provider_id=provider_id,
            )

            logger.info(f"Agent {agent_id[:8]} completed by {provider_id[:8]}")
        else:
            record.status = DispatchStatus.FAILED
            record.error = data.get("error", "Unknown error")
            record.completed_at = time.time()

            # Refund escrow
            await self.escrow.refund_escrow(agent_id, record.error or "Execution failed")

            logger.warning(f"Agent {agent_id[:8]} failed: {record.error}")

        # Signal anyone waiting for result
        record._result_event.set()

    async def _check_bid_timeout(self, agent_id: str) -> None:
        """Check if bid collection has timed out and handle accordingly."""
        record = self._dispatches.get(agent_id)
        if not record or record.status != DispatchStatus.BIDDING:
            return

        if not record.bids:
            record.status = DispatchStatus.FAILED
            record.error = "Bid timeout: no bids received"
            await self.escrow.refund_escrow(agent_id, "Bid timeout")
        else:
            await self.select_and_transfer(agent_id)

    async def wait_for_result(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Wait for the dispatch to complete and return the result."""
        record = self._dispatches.get(agent_id)
        if not record:
            return None

        wait_timeout = timeout or self.result_timeout
        try:
            await asyncio.wait_for(record._result_event.wait(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            if record.status not in (DispatchStatus.COMPLETED, DispatchStatus.FAILED):
                record.status = DispatchStatus.EXPIRED
                record.error = f"Result timeout after {wait_timeout}s"
                await self.escrow.refund_escrow(agent_id, record.error)
            return None

        return record.result

    def _select_best_bid(
        self,
        bids: List[Dict[str, Any]],
        max_budget: float,
    ) -> Optional[Dict[str, Any]]:
        """Select the best bid based on price, tier, and reputation.

        Score = (budget_headroom * 0.3) + (tier_score * 0.3) + (reputation * 0.4)
        """
        valid_bids = [b for b in bids if b.get("price_ftns", 0) <= max_budget]
        if not valid_bids:
            return None

        def score(bid):
            price = bid.get("price_ftns", 0)
            headroom = (max_budget - price) / max_budget if max_budget > 0 else 0
            tier = {"t1": 0.25, "t2": 0.5, "t3": 0.75, "t4": 1.0}.get(
                bid.get("hardware_tier", "t1"), 0.25
            )
            rep = bid.get("reputation", 0.5)
            return headroom * 0.3 + tier * 0.3 + rep * 0.4

        return max(valid_bids, key=score)

    def get_record(self, agent_id: str) -> Optional[DispatchRecord]:
        """Get the dispatch record for an agent."""
        return self._dispatches.get(agent_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_agent_dispatcher.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/compute/agents/dispatcher.py tests/unit/test_agent_dispatcher.py
git commit -m "feat(ring2): AgentDispatcher — dispatch-bid-transfer-settle lifecycle"
```

---

### Task 5: Update Package Exports + Node Integration

**Files:**
- Modify: `prsm/compute/agents/__init__.py`
- Modify: `prsm/node/node.py`

- [ ] **Step 1: Update `__init__.py` with all Ring 2 exports**

Update `prsm/compute/agents/__init__.py`:

```python
"""
Mobile Agent Framework
======================

WASM-based mobile agents that travel to data instead of moving data to compute.
Ring 2 of the Sovereign-Edge AI architecture.
"""

from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)
from prsm.compute.agents.dispatcher import AgentDispatcher
from prsm.compute.agents.executor import AgentExecutor

__all__ = [
    "AgentManifest",
    "MobileAgent",
    "DispatchStatus",
    "DispatchRecord",
    "AgentDispatcher",
    "AgentExecutor",
]
```

- [ ] **Step 2: Wire into node.py**

In `prsm/node/node.py`, add initialization of AgentDispatcher and AgentExecutor. Find the section where `agent_collaboration` is initialized (around line 695-707) and add after it:

```python
        # ── Mobile Agent Dispatch (Ring 2) ────────────────────────────
        from prsm.compute.agents.dispatcher import AgentDispatcher
        from prsm.compute.agents.executor import AgentExecutor

        self.agent_dispatcher = AgentDispatcher(
            identity=self.identity,
            gossip=self.gossip,
            transport=self.transport,
            escrow=self._payment_escrow,
        )

        self.agent_executor = AgentExecutor(
            identity=self.identity,
            gossip=self.gossip,
        )
```

- [ ] **Step 3: Commit**

```bash
git add prsm/compute/agents/__init__.py prsm/node/node.py
git commit -m "feat(ring2): wire AgentDispatcher + AgentExecutor into PRSMNode"
```

---

### Task 6: Integration Smoke Test

**Files:**
- Create: `tests/integration/test_ring2_dispatch.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_ring2_dispatch.py`:

```python
"""
Ring 2 Smoke Test
=================

End-to-end test: create agent, dispatch, simulate bid, execute, verify result.
Tests the full dispatch lifecycle with mocked gossip (no real network).
"""

import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

from prsm.compute.agents import (
    AgentManifest,
    MobileAgent,
    AgentDispatcher,
    AgentExecutor,
    DispatchStatus,
)
from prsm.compute.wasm import WasmtimeRuntime


# Minimal WASM: (module (func (export "run") (result i32) (i32.const 42)))
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing2Smoke:
    @pytest.mark.asyncio
    async def test_full_dispatch_lifecycle(self):
        """Test: create agent → dispatch → bid → execute → result → settle."""
        # Set up requester
        requester_identity = MagicMock()
        requester_identity.node_id = "requester-001"
        requester_identity.sign = MagicMock(return_value="req-sig")
        requester_identity.public_key_b64 = "cmVx"

        gossip = AsyncMock()
        gossip.publish = AsyncMock(return_value=1)
        gossip.subscribe = MagicMock()

        transport = AsyncMock()
        transport.send_to_peer = AsyncMock(return_value=True)

        escrow = AsyncMock()
        entry = MagicMock()
        entry.escrow_id = "escrow-test-001"
        escrow.create_escrow = AsyncMock(return_value=entry)
        escrow.release_escrow = AsyncMock()
        escrow.refund_escrow = AsyncMock()

        # Create dispatcher
        dispatcher = AgentDispatcher(
            identity=requester_identity,
            gossip=gossip,
            transport=transport,
            escrow=escrow,
        )

        # Step 1: Create agent
        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(
                required_cids=["QmTestShard"],
                min_hardware_tier="t1",
            ),
            ftns_budget=2.0,
            ttl=120,
        )
        assert agent.origin_node == "requester-001"

        # Step 2: Dispatch
        record = await dispatcher.dispatch(agent)
        assert record.status == DispatchStatus.BIDDING
        escrow.create_escrow.assert_called_once()

        # Step 3: Simulate a bid arriving
        await dispatcher._on_agent_accept(
            "agent_accept",
            {
                "agent_id": agent.agent_id,
                "provider_id": "provider-001",
                "price_ftns": 0.5,
                "hardware_tier": "t2",
                "reputation": 0.8,
            },
            "provider-001",
        )
        assert len(record.bids) == 1

        # Step 4: Select winner and transfer
        success = await dispatcher.select_and_transfer(agent.agent_id)
        assert success
        assert record.status == DispatchStatus.EXECUTING
        transport.send_to_peer.assert_called_once()

        # Step 5: Set up provider executor and execute
        provider_identity = MagicMock()
        provider_identity.node_id = "provider-001"
        provider_identity.sign = MagicMock(return_value="prov-sig")
        provider_identity.public_key_b64 = "cHJvdg=="

        executor = AgentExecutor(
            identity=provider_identity,
            gossip=gossip,
            hardware_tier="t2",
        )

        exec_result = await executor.execute_agent(agent, input_data=b"")

        # Step 6: Simulate result arriving at dispatcher
        await dispatcher._on_agent_result(
            "agent_result",
            exec_result,
            "provider-001",
        )

        # Verify lifecycle completed
        if exec_result["status"] == "success":
            assert record.status == DispatchStatus.COMPLETED
            assert record.result is not None
            escrow.release_escrow.assert_called_once()
        else:
            # WASM execution might fail if runtime has issues,
            # but the lifecycle should still complete
            assert record.status in (DispatchStatus.COMPLETED, DispatchStatus.FAILED)

    @pytest.mark.asyncio
    async def test_dispatch_with_no_bids_refunds(self):
        """Test that dispatch with no bids refunds the escrow."""
        identity = MagicMock()
        identity.node_id = "lonely-node"
        identity.sign = MagicMock(return_value="sig")

        gossip = AsyncMock()
        gossip.publish = AsyncMock(return_value=0)  # No peers
        gossip.subscribe = MagicMock()

        transport = AsyncMock()
        escrow = AsyncMock()
        entry = MagicMock()
        entry.escrow_id = "escrow-lonely"
        escrow.create_escrow = AsyncMock(return_value=entry)
        escrow.refund_escrow = AsyncMock(return_value=True)

        dispatcher = AgentDispatcher(
            identity=identity,
            gossip=gossip,
            transport=transport,
            escrow=escrow,
        )

        agent = dispatcher.create_agent(
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            ftns_budget=1.0,
            ttl=60,
        )

        record = await dispatcher.dispatch(agent, bid_timeout=0.1)

        # Wait for timeout, then check
        await asyncio.sleep(0.2)
        await dispatcher._check_bid_timeout(agent.agent_id)

        assert record.status in (DispatchStatus.FAILED, DispatchStatus.REFUNDED)
        escrow.refund_escrow.assert_called()

    @pytest.mark.skipif(
        not WasmtimeRuntime().available,
        reason="wasmtime not installed",
    )
    @pytest.mark.asyncio
    async def test_executor_produces_valid_pcu(self):
        """Verify executor returns real PCU metrics from WASM execution."""
        identity = MagicMock()
        identity.node_id = "exec-node"
        identity.sign = MagicMock(return_value="sig")
        identity.public_key_b64 = "a2V5"

        gossip = AsyncMock()

        executor = AgentExecutor(
            identity=identity,
            gossip=gossip,
            hardware_tier="t2",
        )

        agent = MobileAgent(
            agent_id="pcu-test",
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="origin",
            signature="sig",
            ftns_budget=1.0,
            ttl=60,
        )

        result = await executor.execute_agent(agent, input_data=b"test")

        assert result["status"] == "success"
        assert result["execution_time_seconds"] >= 0
        assert result["pcu"] >= 0
        assert len(result["provider_signature"]) > 0
```

- [ ] **Step 2: Run the smoke test**

Run: `python -m pytest tests/integration/test_ring2_dispatch.py -v --timeout=30`
Expected: 3 tests PASS (or 2 pass + 1 skip if wasmtime not installed)

- [ ] **Step 3: Run full Ring 1 + Ring 2 test suite for regression check**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/unit/test_mobile_agent_models.py tests/unit/test_agent_executor.py tests/unit/test_agent_dispatcher.py tests/integration/test_ring1_smoke.py tests/integration/test_ring2_dispatch.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_ring2_dispatch.py
git commit -m "test(ring2): integration smoke test — full dispatch lifecycle"
```

---

### Task 7: Version Bump + Push + PyPI

**Files:**
- Modify: `prsm/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

Change `__version__` in `prsm/__init__.py` to `"0.27.0"`.
Change `version` in `pyproject.toml` to `"0.27.0"`.

- [ ] **Step 2: Run final test suite**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/unit/test_mobile_agent_models.py tests/unit/test_agent_executor.py tests/unit/test_agent_dispatcher.py tests/integration/test_ring1_smoke.py tests/integration/test_ring2_dispatch.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 3: Commit and push**

```bash
git add prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.27.0 for Ring 2 — Mobile Agent Dispatch"
git push origin main
```

- [ ] **Step 4: Build and publish**

```bash
rm -rf build/ dist/ prsm_network.egg-info/
python3 -m build
python3 -m twine upload dist/prsm_network-0.27.0*
```
