"""Tests for Ring 8 PipelineRandomizer — collusion-resistant node assignment."""

import pytest

from prsm.compute.model_sharding.randomizer import PipelineRandomizer


def _make_nodes(count: int, tee_enabled: bool = False):
    """Helper to create a list of node dicts."""
    return [
        {"node_id": f"node-{i}", "tee_enabled": tee_enabled}
        for i in range(count)
    ]


class TestPipelineRandomizer:
    def test_assigns_correct_shard_count(self):
        randomizer = PipelineRandomizer(min_pool_size=5)
        nodes = _make_nodes(10)
        assignments = randomizer.assign_pipeline(4, nodes)
        assert len(assignments) == 4
        indices = {a["shard_index"] for a in assignments}
        assert indices == {0, 1, 2, 3}

    def test_enforces_min_pool_size(self):
        randomizer = PipelineRandomizer(min_pool_size=20)
        nodes = _make_nodes(10)
        with pytest.raises(ValueError, match="below the minimum"):
            randomizer.assign_pipeline(4, nodes)

    def test_filters_by_tee_when_required(self):
        randomizer = PipelineRandomizer(min_pool_size=5)
        tee_nodes = _make_nodes(8, tee_enabled=True)
        non_tee = [{"node_id": f"plain-{i}", "tee_enabled": False} for i in range(10)]
        all_nodes = non_tee + tee_nodes

        assignments = randomizer.assign_pipeline(4, all_nodes, require_tee=True)
        assert len(assignments) == 4
        assigned_ids = {a["node_id"] for a in assignments}
        tee_ids = {n["node_id"] for n in tee_nodes}
        # All assigned nodes must be TEE-enabled
        assert assigned_ids.issubset(tee_ids)

    def test_tee_filter_reduces_pool_below_minimum(self):
        randomizer = PipelineRandomizer(min_pool_size=10)
        tee_nodes = _make_nodes(3, tee_enabled=True)
        non_tee = _make_nodes(30, tee_enabled=False)
        all_nodes = non_tee + tee_nodes

        with pytest.raises(ValueError, match="below the minimum"):
            randomizer.assign_pipeline(2, all_nodes, require_tee=True)

    def test_assignments_are_random(self):
        """Two calls with a large pool should produce different assignments
        with very high probability."""
        randomizer = PipelineRandomizer(min_pool_size=5)
        nodes = _make_nodes(100)

        results = set()
        for _ in range(10):
            assignments = randomizer.assign_pipeline(4, nodes)
            key = tuple(a["node_id"] for a in assignments)
            results.add(key)

        # With 100 nodes and 4 picks, the chance of all 10 being identical
        # is astronomically low.
        assert len(results) > 1

    def test_each_assignment_has_node_id_and_shard_index(self):
        randomizer = PipelineRandomizer(min_pool_size=5)
        nodes = _make_nodes(20)
        assignments = randomizer.assign_pipeline(3, nodes)
        for a in assignments:
            assert "node_id" in a
            assert "shard_index" in a


# ── TensorParallelExecutor tests ─────────────────────────────────────

import json
import numpy as np
from prsm.compute.model_sharding.executor import TensorParallelExecutor
from prsm.compute.model_sharding.collision_detector import CollisionDetector
from prsm.compute.model_sharding.models import ShardedModel, ModelShard, PipelineConfig


class TestTensorParallelExecutor:
    @pytest.mark.asyncio
    async def test_execute_parallel_produces_result(self):
        executor = TensorParallelExecutor()

        # Create a simple sharded model (2 shards of a 4x4 matrix)
        tensor = np.random.randn(4, 4)
        shard1_data = tensor[:2].tobytes()
        shard2_data = tensor[2:].tobytes()

        model = ShardedModel(
            model_id="test-model",
            model_name="TestModel",
            total_shards=2,
            shards=[
                ModelShard(
                    shard_id="s0", model_id="test-model", shard_index=0,
                    total_shards=2, tensor_data=shard1_data,
                    tensor_shape=(2, 4), size_bytes=len(shard1_data), checksum="abc",
                ),
                ModelShard(
                    shard_id="s1", model_id="test-model", shard_index=1,
                    total_shards=2, tensor_data=shard2_data,
                    tensor_shape=(2, 4), size_bytes=len(shard2_data), checksum="def",
                ),
            ],
        )

        # node_id="local" — Ring 8 in-process numpy execution path. The
        # executor's remote-dispatch seam is covered separately in
        # tests/integration/test_ring8_shield.py.
        assignments = [
            {"node_id": "local", "shard_index": 0},
            {"node_id": "local", "shard_index": 1},
        ]

        result = await executor.execute_parallel(model, b"", assignments)

        assert result["status"] == "success"
        assert result["shards_executed"] == 2
        assert result["aggregated_output"] is not None

    def test_all_reduce_averages(self):
        a = np.array([2.0, 4.0, 6.0])
        b = np.array([4.0, 6.0, 8.0])
        result = TensorParallelExecutor.all_reduce([a, b])
        np.testing.assert_array_almost_equal(result, [3.0, 5.0, 7.0])


# ── CollisionDetector tests ──────────────────────────────────────────


class TestCollisionDetector:
    def test_matching_outputs_pass(self):
        detector = CollisionDetector(dp_epsilon=8.0)
        a = json.dumps([1.0, 2.0, 3.0]).encode()
        b = json.dumps([1.0, 2.0, 3.0]).encode()
        match, div = detector.compare_pipelines(a, b)
        assert match is True
        assert div < 0.01

    def test_divergent_outputs_detected(self):
        detector = CollisionDetector(dp_epsilon=8.0, tolerance_multiplier=1.0)
        a = json.dumps([1.0, 2.0, 3.0]).encode()
        b = json.dumps([100.0, 200.0, 300.0]).encode()
        match, div = detector.compare_pipelines(a, b)
        assert match is False
        assert div > 0.5

    def test_dp_noise_within_tolerance(self):
        detector = CollisionDetector(dp_epsilon=8.0, tolerance_multiplier=5.0)
        base = [1.0, 2.0, 3.0]
        noisy = [1.01, 2.01, 3.01]  # Tiny noise
        a = json.dumps(base).encode()
        b = json.dumps(noisy).encode()
        match, div = detector.compare_pipelines(a, b)
        assert match is True

    def test_detect_collision_all_match(self):
        detector = CollisionDetector()
        outputs = [
            json.dumps([1.0, 2.0]).encode(),
            json.dumps([1.0, 2.0]).encode(),
            json.dumps([1.0, 2.0]).encode(),
        ]
        report = detector.detect_collision(outputs)
        assert report["match"] is True
        assert report["comparisons"] == 3
        assert len(report["flagged_indices"]) == 0

    def test_detect_collision_flags_divergent(self):
        detector = CollisionDetector(dp_epsilon=8.0, tolerance_multiplier=1.0)
        outputs = [
            json.dumps([1.0, 2.0]).encode(),
            json.dumps([1.0, 2.0]).encode(),
            json.dumps([999.0, 999.0]).encode(),  # Divergent
        ]
        report = detector.detect_collision(outputs)
        assert report["match"] is False
        assert 2 in report["flagged_indices"]
