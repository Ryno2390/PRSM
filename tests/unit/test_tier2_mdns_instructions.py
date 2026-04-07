"""Tests for Tier 2: mDNS discovery and agent instructions."""

import pytest
import json

from prsm.node.mdns_discovery import MDNSDiscovery, LocalPeer
from prsm.compute.agents.instruction_set import (
    AgentOp,
    AgentInstruction,
    InstructionManifest,
    instructions_from_decomposition,
)


class TestMDNSDiscovery:
    def test_discovery_creation(self):
        disc = MDNSDiscovery(
            node_id="test-node",
            p2p_port=9001,
            display_name="TestNode",
        )
        assert disc.node_id == "test-node"
        assert disc.p2p_port == 9001
        assert len(disc.discovered_peers) == 0

    def test_handle_announcement(self):
        disc = MDNSDiscovery(node_id="node-a", p2p_port=9001)
        announcement = json.dumps({
            "type": "prsm_announce",
            "node_id": "node-b",
            "p2p_port": 9002,
            "display_name": "NodeB",
        }).encode()

        disc._handle_announcement(announcement, ("192.168.1.50", 19999))

        assert len(disc.discovered_peers) == 1
        peer = disc.discovered_peers[0]
        assert peer.node_id == "node-b"
        assert peer.address == "ws://192.168.1.50:9002"

    def test_ignores_own_announcement(self):
        disc = MDNSDiscovery(node_id="node-a", p2p_port=9001)
        announcement = json.dumps({
            "type": "prsm_announce",
            "node_id": "node-a",  # Same as self
            "p2p_port": 9001,
        }).encode()

        disc._handle_announcement(announcement, ("192.168.1.1", 19999))
        assert len(disc.discovered_peers) == 0

    def test_ignores_non_prsm_messages(self):
        disc = MDNSDiscovery(node_id="node-a", p2p_port=9001)
        disc._handle_announcement(b'{"type": "other"}', ("1.2.3.4", 5000))
        assert len(disc.discovered_peers) == 0

    def test_peer_addresses(self):
        disc = MDNSDiscovery(node_id="node-a", p2p_port=9001)
        disc._peers["b"] = LocalPeer(node_id="b", address="ws://10.0.0.2:9002")
        disc._peers["c"] = LocalPeer(node_id="c", address="ws://10.0.0.3:9003")
        addrs = disc.peer_addresses
        assert len(addrs) == 2
        assert "ws://10.0.0.2:9002" in addrs


class TestAgentInstructions:
    def test_instruction_creation(self):
        inst = AgentInstruction(op=AgentOp.FILTER, field="state", value="NC")
        assert inst.op == AgentOp.FILTER
        assert inst.field == "state"

    def test_instruction_roundtrip(self):
        inst = AgentInstruction(op=AgentOp.AGGREGATE, params={"method": "sum"})
        d = inst.to_dict()
        restored = AgentInstruction.from_dict(d)
        assert restored.op == AgentOp.AGGREGATE
        assert restored.params["method"] == "sum"

    def test_manifest_creation(self):
        manifest = InstructionManifest(
            query="EV trends in NC",
            instructions=[
                AgentInstruction(op=AgentOp.FILTER, field="state", value="NC"),
                AgentInstruction(op=AgentOp.GROUP_BY, field="vehicle_type"),
                AgentInstruction(op=AgentOp.COUNT),
            ],
        )
        assert len(manifest.instructions) == 3

    def test_manifest_json_roundtrip(self):
        manifest = InstructionManifest(
            query="test",
            instructions=[AgentInstruction(op=AgentOp.COUNT)],
        )
        json_str = manifest.to_json()
        restored = InstructionManifest.from_json(json_str)
        assert restored.query == "test"
        assert len(restored.instructions) == 1
        assert restored.instructions[0].op == AgentOp.COUNT

    def test_manifest_to_wasm_input(self):
        manifest = InstructionManifest(query="test", instructions=[])
        wasm_input = manifest.to_wasm_input()
        assert isinstance(wasm_input, bytes)
        parsed = json.loads(wasm_input)
        assert parsed["query"] == "test"

    def test_from_decomposition_maps_operations(self):
        decomp = {
            "query": "EV adoption in NC",
            "operations": ["filter", "aggregate", "time_series"],
        }
        manifest = instructions_from_decomposition(decomp)
        assert manifest.query == "EV adoption in NC"
        assert len(manifest.instructions) == 3
        ops = [i.op for i in manifest.instructions]
        assert AgentOp.FILTER in ops
        assert AgentOp.AGGREGATE in ops
        assert AgentOp.TIME_SERIES in ops

    def test_from_decomposition_handles_natural_language_ops(self):
        decomp = {
            "query": "count vehicles",
            "operations": ["Filter records for North Carolina", "Calculate EV adoption percentage"],
        }
        manifest = instructions_from_decomposition(decomp)
        assert len(manifest.instructions) >= 1
        # "Filter records..." should map to FILTER
        assert any(i.op == AgentOp.FILTER for i in manifest.instructions)

    def test_from_decomposition_empty_defaults_to_count(self):
        decomp = {"query": "hello", "operations": []}
        manifest = instructions_from_decomposition(decomp)
        assert len(manifest.instructions) == 1
        assert manifest.instructions[0].op == AgentOp.COUNT

    def test_all_agent_ops_exist(self):
        assert AgentOp.FILTER == "filter"
        assert AgentOp.AGGREGATE == "aggregate"
        assert AgentOp.GROUP_BY == "group_by"
        assert AgentOp.SUM == "sum"
        assert AgentOp.AVERAGE == "average"
        assert AgentOp.TIME_SERIES == "time_series"
