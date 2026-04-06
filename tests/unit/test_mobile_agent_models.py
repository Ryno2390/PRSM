"""Tests for Mobile Agent data models."""

import pytest
import json
import uuid
import time
from prsm.compute.agents.models import (
    AgentManifest,
    MobileAgent,
    DispatchStatus,
    DispatchRecord,
)


WASM_MAGIC = b"\x00asm\x01\x00\x00\x00"


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
        agent = MobileAgent(
            agent_id="expired-agent",
            wasm_binary=WASM_MAGIC,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            origin_node="node-x",
            signature="sig",
            ftns_budget=1.0,
            ttl=0,
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
        record.status = DispatchStatus.COMPLETED
        assert record.status == DispatchStatus.COMPLETED


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
