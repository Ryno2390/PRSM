"""Ring 8 Smoke Test — model sharding + collusion resistance."""

import pytest
import json
import numpy as np
from prsm.compute.model_sharding import (
    ModelSharder,
    PipelineRandomizer,
    ShardedModel,
    PipelineConfig,
    PipelineStakeTier,
)
from prsm.compute.model_sharding.executor import TensorParallelExecutor
from prsm.compute.model_sharding.collision_detector import CollisionDetector


class TestRing8Smoke:
    def test_model_shard_roundtrip(self):
        sharder = ModelSharder()
        tensor = np.random.randn(16, 8)
        shards = sharder.shard_tensor(tensor, 4)
        assert len(shards) == 4
        reassembled = sharder.reassemble_tensor(shards)
        np.testing.assert_array_almost_equal(reassembled, tensor)

    def test_full_model_sharding(self):
        sharder = ModelSharder()
        weights = {
            "layer1": np.random.randn(32, 16),
            "layer2": np.random.randn(16, 8),
        }
        model = sharder.shard_model("model-1", "TestModel", weights, n_shards=4)
        # total_shards records the parallelism degree (n_shards)
        assert model.total_shards == 4
        # actual shard list = n_layers * n_shards = 2 * 4 = 8
        assert len(model.shards) == 8
        assert model.total_size_bytes > 0

    def test_pipeline_randomizer(self):
        randomizer = PipelineRandomizer(min_pool_size=5)
        nodes = [{"node_id": f"node-{i}", "tee_available": True} for i in range(10)]
        assignments = randomizer.assign_pipeline(4, nodes)
        assert len(assignments) == 4
        node_ids = [a["node_id"] for a in assignments]
        assert len(set(node_ids)) == 4  # All unique

    @pytest.mark.asyncio
    async def test_tensor_parallel_execution(self):
        sharder = ModelSharder()
        weights = {"w": np.random.randn(8, 4)}
        model = sharder.shard_model("m1", "Test", weights, n_shards=2)

        executor = TensorParallelExecutor()
        assignments = [
            {"node_id": "a", "shard_index": 0},
            {"node_id": "b", "shard_index": 1},
        ]
        result = await executor.execute_parallel(model, b"", assignments)
        assert result["status"] == "success"
        assert result["shards_executed"] == 2

    def test_collision_detection(self):
        detector = CollisionDetector(dp_epsilon=8.0, tolerance_multiplier=1.0)
        good = json.dumps([1.0, 2.0, 3.0]).encode()
        bad = json.dumps([999.0, 999.0, 999.0]).encode()
        report = detector.detect_collision([good, good, bad])
        assert report["match"] is False
        assert 2 in report["flagged_indices"]

    def test_all_rings_1_through_8_import(self):
        from prsm.compute.wasm import WASMRuntime, HardwareProfiler
        from prsm.compute.agents import AgentDispatcher, AgentExecutor
        from prsm.compute.swarm import SwarmCoordinator
        from prsm.economy.pricing import PricingEngine
        # Ring 5 AgentForge removed in v1.6.0 (legacy NWTN AGI framework pruned)
        from prsm.compute.tee import ConfidentialResult, DPNoiseInjector
        from prsm.compute.model_sharding import ModelSharder, PipelineRandomizer
        from prsm.compute.model_sharding.executor import TensorParallelExecutor
        from prsm.compute.model_sharding.collision_detector import CollisionDetector
        assert all(x is not None for x in [
            WASMRuntime, AgentDispatcher, SwarmCoordinator, PricingEngine,
            DPNoiseInjector, ModelSharder, TensorParallelExecutor,
            CollisionDetector,
        ])
