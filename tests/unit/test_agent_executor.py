"""
Tests for AgentExecutor — provider-side WASM agent execution.
"""

import asyncio
import base64
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prsm.compute.agents.models import AgentManifest, MobileAgent
from prsm.compute.agents.executor import AgentExecutor, TIER_ORDER
from prsm.compute.wasm.models import ExecutionResult, ExecutionStatus, ResourceLimits


# ── Helpers ──────────────────────────────────────────────────────────────

# Minimal valid WASM binary (magic + version + empty module)
VALID_WASM = b"\x00asm\x01\x00\x00\x00"

def _make_identity(node_id="provider-1"):
    identity = MagicMock()
    identity.node_id = node_id
    identity.sign = MagicMock(return_value="sig-abc123")
    identity.public_key_b64 = "cHVibGlj"  # base64("public")
    return identity


def _make_gossip():
    gossip = AsyncMock()
    gossip.publish = AsyncMock(return_value=3)
    return gossip


def _make_agent(
    agent_id="agent-1",
    ttl=120,
    min_hardware_tier="t1",
    created_at=None,
):
    manifest = AgentManifest(min_hardware_tier=min_hardware_tier)
    agent = MobileAgent(
        agent_id=agent_id,
        wasm_binary=VALID_WASM,
        manifest=manifest,
        origin_node="origin-node-1",
        signature="sig-origin",
        ftns_budget=10.0,
        ttl=ttl,
        created_at=created_at or time.time(),
    )
    return agent


# ── Tests ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_valid_agent_success():
    """Execute a valid agent -> success with output, metrics, signature."""
    identity = _make_identity()
    gossip = _make_gossip()
    executor = AgentExecutor(identity, gossip, hardware_tier="t2")

    agent = _make_agent()

    mock_result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        output=b"hello world",
        execution_time_seconds=0.5,
        memory_used_bytes=1024,
    )

    with patch.object(executor._runtime, "load", return_value=("engine", "module")), \
         patch.object(executor._runtime, "execute", return_value=mock_result):
        result = await executor.execute_agent(agent, b"input-data")

    assert result["status"] == "success"
    assert result["agent_id"] == "agent-1"
    assert result["provider_id"] == "provider-1"
    assert result["output_b64"] == base64.b64encode(b"hello world").decode()
    assert result["execution_time_seconds"] == 0.5
    assert result["memory_used_bytes"] == 1024
    assert "pcu" in result
    assert result["provider_signature"] == "sig-abc123"
    assert result["provider_public_key"] == "cHVibGlj"
    assert result["error"] is None


@pytest.mark.asyncio
async def test_execute_invalid_wasm_returns_error():
    """Execute invalid WASM -> error status with error message."""
    identity = _make_identity()
    gossip = _make_gossip()
    executor = AgentExecutor(identity, gossip)

    agent = _make_agent()

    with patch.object(executor._runtime, "load", side_effect=ValueError("bad wasm")):
        result = await executor.execute_agent(agent, b"input")

    assert result["status"] == "error"
    assert "bad wasm" in result["error"]
    assert result["agent_id"] == "agent-1"


@pytest.mark.asyncio
async def test_execute_expired_agent_returns_error():
    """Execute expired agent -> error with 'expired' in message."""
    identity = _make_identity()
    gossip = _make_gossip()
    executor = AgentExecutor(identity, gossip)

    # Created 200 seconds ago with 60-second TTL -> expired
    agent = _make_agent(ttl=60, created_at=time.time() - 200)

    result = await executor.execute_agent(agent, b"input")

    assert result["status"] == "error"
    assert "expired" in result["error"].lower()


@pytest.mark.asyncio
async def test_execute_with_publish_result():
    """Execute with publish_result=True -> gossip.publish called."""
    identity = _make_identity()
    gossip = _make_gossip()
    executor = AgentExecutor(identity, gossip)

    agent = _make_agent()

    mock_result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        output=b"output",
        execution_time_seconds=0.1,
        memory_used_bytes=512,
    )

    with patch.object(executor._runtime, "load", return_value=("e", "m")), \
         patch.object(executor._runtime, "execute", return_value=mock_result):
        result = await executor.execute_agent(agent, b"data", publish_result=True)

    gossip.publish.assert_awaited_once()
    call_args = gossip.publish.call_args
    assert call_args[0][0] == "agent_result"


def test_validate_manifest_rejects_insufficient_tier():
    """Validate manifest rejects: have t1, need t4."""
    identity = _make_identity()
    gossip = _make_gossip()
    executor = AgentExecutor(identity, gossip, hardware_tier="t1")

    manifest = AgentManifest(min_hardware_tier="t4")
    ok, reason = executor.validate_manifest(manifest)

    assert ok is False
    assert "tier" in reason.lower()


def test_validate_manifest_accepts_sufficient_tier():
    """Validate manifest accepts: have t2, need t1."""
    identity = _make_identity()
    gossip = _make_gossip()
    executor = AgentExecutor(identity, gossip, hardware_tier="t2")

    manifest = AgentManifest(min_hardware_tier="t1")
    ok, reason = executor.validate_manifest(manifest)

    assert ok is True
