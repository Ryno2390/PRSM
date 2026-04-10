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
    async def test_tensor_parallel_execution_local(self):
        """Local-only tensor parallel execution.

        This intentionally uses node_id="local" because PRSM 1.6.x ships
        Ring 8 with local in-process execution and a remote-dispatch seam,
        but no first-party WASM tensor-matmul agent yet. See
        test_tensor_parallel_execution_remote for the dispatcher seam test.
        """
        sharder = ModelSharder()
        weights = {"w": np.random.randn(8, 4)}
        model = sharder.shard_model("m1", "Test", weights, n_shards=2)

        executor = TensorParallelExecutor()
        assignments = [
            {"node_id": "local", "shard_index": 0},
            {"node_id": "local", "shard_index": 1},
        ]
        result = await executor.execute_parallel(model, b"", assignments)
        assert result["status"] == "success"
        assert result["shards_executed"] == 2
        assert result["execution_modes"] == {"local": 2, "remote": 0}

    @pytest.mark.asyncio
    async def test_tensor_parallel_remote_dispatch_seam(self):
        """The remote_dispatcher hook routes non-local assignments out."""
        sharder = ModelSharder()
        weights = {"w": np.random.randn(8, 4)}
        model = sharder.shard_model("m1", "Test", weights, n_shards=2)

        called_with = []

        async def fake_dispatcher(shard, input_data, assignment):
            called_with.append((shard.shard_index, assignment.get("node_id")))
            return {"output_array": [float(shard.shard_index)] * 4}

        executor = TensorParallelExecutor(remote_dispatcher=fake_dispatcher)
        assert executor.supports_remote_execution is True

        result = await executor.execute_parallel(
            model, b"",
            [
                {"node_id": "remote-A", "shard_index": 0},
                {"node_id": "remote-B", "shard_index": 1},
            ],
        )
        assert result["status"] == "success"
        assert result["execution_modes"] == {"local": 0, "remote": 2}
        assert sorted(called_with) == [(0, "remote-A"), (1, "remote-B")]

    @pytest.mark.asyncio
    async def test_tensor_parallel_remote_without_dispatcher_errors(self):
        """Without a dispatcher, remote assignments must fail loudly."""
        sharder = ModelSharder()
        weights = {"w": np.random.randn(8, 4)}
        model = sharder.shard_model("m1", "Test", weights, n_shards=2)

        executor = TensorParallelExecutor()  # no dispatcher
        result = await executor.execute_parallel(
            model, b"", [{"node_id": "remote-X", "shard_index": 0}]
        )
        assert result["status"] == "failed"
        assert any("remote_dispatcher" in e for e in result["errors"])

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
