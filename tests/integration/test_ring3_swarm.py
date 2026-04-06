"""
Ring 3 Smoke Test
=================

End-to-end: semantic shards -> swarm dispatch -> collect + aggregate.
"""

import pytest
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

from prsm.data.shard_models import SemanticShard, SemanticShardManifest
from prsm.compute.swarm import SwarmCoordinator, SwarmStatus, MapReduceStrategy
from prsm.compute.agents.models import AgentManifest, DispatchStatus


MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestRing3Smoke:
    @pytest.mark.asyncio
    async def test_semantic_shard_to_swarm_pipeline(self):
        """Full Ring 3: find relevant shards -> dispatch swarm -> aggregate."""
        # Step 1: Create semantic shard manifest
        shards = [
            SemanticShard(
                shard_id=f"shard-{i}",
                parent_dataset="nada-nc",
                cid=f"QmShard{i:03d}",
                centroid=[float(i) * 0.2, 1.0 - float(i) * 0.2, 0.5],
                record_count=1000,
                size_bytes=1024 * 1024,
                keywords=[f"topic-{i}"],
            )
            for i in range(5)
        ]
        manifest = SemanticShardManifest(
            dataset_id="nada-nc-2025",
            total_records=5000,
            total_size_bytes=5 * 1024 * 1024,
            shards=shards,
        )

        # Step 2: Find relevant shards
        query_embedding = [0.1, 0.9, 0.5]
        relevant_pairs = manifest.find_relevant_shards(query_embedding, top_k=3)
        assert len(relevant_pairs) == 3
        relevant = [shard for shard, _score in relevant_pairs]

        # Step 3: Mock dispatcher
        mock_dispatcher = AsyncMock()

        def create_agent_fn(wasm_binary, manifest, ftns_budget, ttl=120):
            agent = MagicMock()
            agent.agent_id = f"agent-{id(manifest)}"
            agent.manifest = manifest
            return agent

        mock_dispatcher.create_agent = MagicMock(side_effect=create_agent_fn)
        mock_dispatcher.dispatch = AsyncMock(return_value=MagicMock(
            status=DispatchStatus.COMPLETED,
            bids=[{"provider_id": "p1"}],
        ))
        mock_dispatcher.select_and_transfer = AsyncMock(return_value=True)
        mock_dispatcher.wait_for_result = AsyncMock(return_value={
            "status": "success",
            "output_b64": base64.b64encode(b'{"ev_count": 142}').decode(),
            "pcu": 0.2,
            "execution_time_seconds": 1.5,
        })

        # Step 4: Execute swarm
        coordinator = SwarmCoordinator(dispatcher=mock_dispatcher)
        job = coordinator.create_swarm_job(
            query="EV adoption in NC",
            shard_cids=[s.cid for s in relevant],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
        )

        result = await coordinator.execute(job)

        assert result.shards_completed == 3
        assert result.total_pcu > 0
        assert result.aggregated_output is not None
        assert result.aggregated_output["shard_count"] == 3
        assert job.status in (SwarmStatus.COMPLETED, SwarmStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_swarm_with_quorum_failure(self):
        """Swarm job fails when quorum not met."""
        mock_dispatcher = AsyncMock()

        def create_agent_fn(wasm_binary, manifest, ftns_budget, ttl=120):
            agent = MagicMock()
            agent.agent_id = f"agent-{id(manifest)}"
            return agent

        mock_dispatcher.create_agent = MagicMock(side_effect=create_agent_fn)
        mock_dispatcher.dispatch = AsyncMock(return_value=MagicMock(status=DispatchStatus.FAILED))
        mock_dispatcher.select_and_transfer = AsyncMock(return_value=False)
        mock_dispatcher.wait_for_result = AsyncMock(return_value=None)

        coordinator = SwarmCoordinator(dispatcher=mock_dispatcher)
        job = coordinator.create_swarm_job(
            query="failing query",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=MINIMAL_WASM,
            manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=15.0,
            strategy=MapReduceStrategy(quorum_pct=0.8),
        )

        result = await coordinator.execute(job)

        assert job.status == SwarmStatus.FAILED
        assert result.shards_completed == 0
        assert "quorum" in (job.error or "").lower()
