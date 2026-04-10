"""Tests for SwarmCoordinator — parallel dispatch + aggregation."""

import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

from prsm.compute.swarm.coordinator import SwarmCoordinator
from prsm.compute.swarm.models import SwarmStatus, MapReduceStrategy
from prsm.compute.agents.models import AgentManifest, DispatchStatus


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock()

    mock_record = MagicMock()
    mock_record.status = DispatchStatus.COMPLETED
    mock_record.result = {
        "status": "success",
        "output_b64": base64.b64encode(b'{"count": 42}').decode(),
        "pcu": 0.1,
        "execution_time_seconds": 1.0,
    }
    mock_record.bids = [{"provider_id": "prov-1", "price_ftns": 0.5}]
    mock_record._result_event = asyncio.Event()
    mock_record._result_event.set()

    def create_agent_side_effect(wasm_binary, manifest, ftns_budget, ttl=120):
        agent = MagicMock()
        agent.agent_id = f"agent-{id(manifest)}"
        agent.manifest = manifest
        return agent

    dispatcher.create_agent = MagicMock(side_effect=create_agent_side_effect)
    dispatcher.dispatch = AsyncMock(return_value=mock_record)
    dispatcher.select_and_transfer = AsyncMock(return_value=True)
    dispatcher.wait_for_result = AsyncMock(return_value=mock_record.result)
    dispatcher.get_record = MagicMock(return_value=mock_record)

    return dispatcher


@pytest.fixture
def coordinator(mock_dispatcher):
    return SwarmCoordinator(dispatcher=mock_dispatcher)


class TestSwarmCoordinator:
    def test_create_swarm_job(self, coordinator):
        job = coordinator.create_swarm_job(
            query="EV trends",
            shard_content_ids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
        )
        assert job.status == SwarmStatus.PENDING
        assert len(job.shard_content_ids) == 3
        assert job.budget_per_shard == 5.0

    @pytest.mark.asyncio
    async def test_execute_dispatches_per_shard(self, coordinator, mock_dispatcher):
        job = coordinator.create_swarm_job(
            query="test query",
            shard_content_ids=["QmA", "QmB"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
        )

        result = await coordinator.execute(job)

        assert mock_dispatcher.create_agent.call_count == 2
        assert mock_dispatcher.dispatch.call_count == 2
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_collects_results(self, coordinator):
        job = coordinator.create_swarm_job(
            query="test",
            shard_content_ids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
        )

        result = await coordinator.execute(job)

        assert result.shards_completed == 3
        assert result.total_pcu > 0

    @pytest.mark.asyncio
    async def test_execute_handles_partial_failure(self, coordinator, mock_dispatcher):
        call_count = {"n": 0}

        async def failing_wait(agent_id, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return None  # Simulate failure
            return {
                "status": "success",
                "output_b64": base64.b64encode(b'{"v": 1}').decode(),
                "pcu": 0.1,
                "execution_time_seconds": 0.5,
            }

        mock_dispatcher.wait_for_result = AsyncMock(side_effect=failing_wait)

        job = coordinator.create_swarm_job(
            query="test",
            shard_content_ids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
            strategy=MapReduceStrategy(quorum_pct=0.5),
        )

        result = await coordinator.execute(job)

        assert result.shards_completed >= 2
        assert job.status in (SwarmStatus.COMPLETED, SwarmStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_execute_with_aggregation(self, coordinator):
        job = coordinator.create_swarm_job(
            query="aggregate test",
            shard_content_ids=["QmA", "QmB"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
        )

        result = await coordinator.execute(job)

        assert result.aggregated_output is not None
        assert "shard_count" in result.aggregated_output

    def test_get_job(self, coordinator):
        job = coordinator.create_swarm_job(
            query="test",
            shard_content_ids=["QmA"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_content_ids=[], min_hardware_tier="t1"),
            budget_ftns=5.0,
        )
        assert coordinator.get_job(job.job_id) is not None

    def test_get_nonexistent_job(self, coordinator):
        assert coordinator.get_job("nonexistent") is None
