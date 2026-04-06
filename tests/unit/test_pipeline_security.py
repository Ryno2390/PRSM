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
