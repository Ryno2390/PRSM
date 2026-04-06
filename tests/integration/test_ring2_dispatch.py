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
        """Test: create agent -> dispatch -> bid -> execute -> result -> settle."""
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
        escrow.create = AsyncMock(return_value="escrow-test-001")
        escrow.release = AsyncMock()
        escrow.refund = AsyncMock()

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
        escrow.create.assert_called_once()

        # Step 3: Simulate bid
        await dispatcher._on_agent_accept(
            "agent_accept",
            {
                "agent_id": agent.agent_id,
                "provider_id": "provider-001",
                "bid_price": 0.5,
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

        # Step 5: Provider executes
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

        # Step 6: Result arrives at dispatcher
        await dispatcher._on_agent_result(
            "agent_result",
            exec_result,
            "provider-001",
        )

        if exec_result["status"] == "success":
            assert record.status == DispatchStatus.COMPLETED
            assert record.result is not None
            escrow.release.assert_called_once()
        else:
            assert record.status in (DispatchStatus.COMPLETED, DispatchStatus.FAILED)

    @pytest.mark.asyncio
    async def test_dispatch_with_no_bids_refunds(self):
        """Test that dispatch with no bids refunds the escrow."""
        identity = MagicMock()
        identity.node_id = "lonely-node"
        identity.sign = MagicMock(return_value="sig")

        gossip = AsyncMock()
        gossip.publish = AsyncMock(return_value=0)
        gossip.subscribe = MagicMock()

        transport = AsyncMock()
        escrow = AsyncMock()
        escrow.create = AsyncMock(return_value="escrow-lonely")
        escrow.refund = AsyncMock(return_value=True)

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
        await asyncio.sleep(0.2)
        await dispatcher._check_bid_timeout(agent.agent_id)

        assert record.status in (DispatchStatus.FAILED, DispatchStatus.REFUNDED)
        escrow.refund.assert_called()

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
