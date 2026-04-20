"""Unit tests for TopologyRandomizer.

Phase 2.1 Line Item B. Verifies:
  - Basic assignment contract (correct count, correct shard indices).
  - Empty pool raises.
  - assign_unique enforces pool >= num_shards.
  - Acceptance criterion: 100 consecutive inferences from the same
    requester over a 20-node eligible pool, and no single node appears
    in more than 10% of assignments for shard position 0 (the most
    inversion-sensitive position).
  - Fairness: over many assignments, each node in the pool appears
    roughly uniformly (within a loose bound).
"""
from __future__ import annotations

from collections import Counter

import pytest

from prsm.compute.topology_randomizer import (
    InsufficientEligiblePoolError,
    ShardAssignment,
    TopologyRandomizer,
)


def test_assign_returns_one_per_shard():
    r = TopologyRandomizer()
    pool = [f"node-{i}" for i in range(5)]
    out = r.assign(pool, num_shards=3)
    assert len(out) == 3
    assert [a.shard_index for a in out] == [0, 1, 2]
    for a in out:
        assert a.node_id in pool


def test_assign_empty_pool_raises():
    r = TopologyRandomizer()
    with pytest.raises(InsufficientEligiblePoolError):
        r.assign([], num_shards=3)


def test_assign_unique_rejects_small_pool():
    r = TopologyRandomizer()
    pool = ["a", "b"]
    with pytest.raises(InsufficientEligiblePoolError):
        r.assign_unique(pool, num_shards=3)


def test_assign_unique_returns_distinct_nodes():
    r = TopologyRandomizer()
    pool = [f"node-{i}" for i in range(10)]
    out = r.assign_unique(pool, num_shards=5)
    node_ids = [a.node_id for a in out]
    assert len(set(node_ids)) == 5


def test_acceptance_criterion_shard0_appearance_bound():
    """Phase 2.1 Line Item B acceptance: across 100 consecutive
    inferences from the same requester, no single node appears in
    more than 10% of the assignments at shard position 0."""
    r = TopologyRandomizer()
    pool = [f"node-{i}" for i in range(20)]

    shard0_nodes = []
    for _ in range(100):
        assignment = r.assign(pool, num_shards=4)
        shard0_nodes.append(assignment[0].node_id)

    freq = Counter(shard0_nodes)
    most_common_count = freq.most_common(1)[0][1]
    assert most_common_count <= 10, (
        f"acceptance criterion violated: top node appeared "
        f"{most_common_count}/100 times (>10%); freq histogram={freq}"
    )


def test_assignments_differ_across_inferences():
    """Two consecutive inferences should not produce identical mappings
    under cryptographic RNG with a reasonable pool size. This is a
    probabilistic check — collision probability for 4 shards over a
    20-node pool under sampling-with-replacement is 1/20^4 ≈ 6e-6."""
    r = TopologyRandomizer()
    pool = [f"node-{i}" for i in range(20)]

    a1 = [a.node_id for a in r.assign(pool, num_shards=4)]
    a2 = [a.node_id for a in r.assign(pool, num_shards=4)]
    assert a1 != a2


def test_fairness_long_run():
    """Over 2000 assignments of 1 shard each, every node in a 10-node
    pool should be selected at least ~10% of the time — loose bound
    for cryptographic RNG uniformity."""
    r = TopologyRandomizer()
    pool = [f"node-{i}" for i in range(10)]

    all_picks = []
    for _ in range(2000):
        all_picks.extend(a.node_id for a in r.assign(pool, num_shards=1))

    freq = Counter(all_picks)
    assert len(freq) == 10, f"some nodes never selected: {set(pool) - set(freq)}"
    # Uniform would be 200 ± a few sigma of sqrt(200) ≈ 14. Use a loose
    # lower bound well below that.
    for node, count in freq.items():
        assert count >= 120, (
            f"node {node} selected only {count}/2000 times — uniformity "
            f"bound failed (freq={freq})"
        )
