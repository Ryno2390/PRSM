"""Sprint 296 — topology rotation per inference.

Vision §7 honest-limits: "Colluding minorities in the right
positions can reconstruct substantial fractions of a model.
Activation-inversion attacks ... Mitigations include
topology rotation per inference and activation-layer TEE
attestation, both of which are in the Phase 2+ roadmap."

The defense: for each new inference request, reshuffle which
nodes get which (stage_index, slot_index) positions. A
colluding subset of nodes that spans the pipeline can only
attack inferences where the rotation lands their nodes in
attack-useful positions. Random rotation each inference
forces an adversary to get lucky every time, not just once.

This sprint ships the primitive:

  TopologyAssignment
    Frozen dataclass: dict[(stage, slot)] → node_id. Carries
    the full assignment + a stable hash for de-duplication
    in history-based rotation strategies.

  TopologySelector
    Given (node_pool, stage_count, slots_per_stage, seed) →
    TopologyAssignment. Each node fills at most one position
    in a single topology (no double-duty within one
    inference; defends operator running the same node in
    two slots to amplify its attack surface).

  TopologyRotationPolicy
    Three strategies:
      "uniform_random" — independent random topology each call
      "beacon_seeded"  — deterministic from a seed (chain-
                         randomness verifiable downstream)
      "anti_repeat"    — must differ from last N topologies
                         (TopologyHistory enforces)

  TopologyHistory
    Bounded ring of recently-issued TopologyAssignments.
    Used by anti_repeat strategy.

  verify_topology_sequence
    Verifier predicate. Checks structural integrity (no dup
    positions, all stages filled) + rotation distinctness
    against a recent-history window.

Sprint 297 will wire this into RpcChainExecutor +
add a topology_assignment field to InferenceReceipt for
caller-side verification.
"""
from __future__ import annotations

import pytest

from prsm.compute.inference.topology_rotation import (
    TopologyAssignment,
    TopologyHistory,
    TopologyRotationPolicy,
    TopologySelector,
    verify_topology_sequence,
)


# ── TopologyAssignment dataclass ─────────────────────────


def test_assignment_to_dict_round_trip():
    a = TopologyAssignment(
        positions={
            (0, 0): "node-a",
            (0, 1): "node-b",
            (1, 0): "node-c",
            (1, 1): "node-d",
        },
        stage_count=2,
        slots_per_stage=2,
    )
    d = a.to_dict()
    assert d["stage_count"] == 2
    assert d["slots_per_stage"] == 2
    # Positions serialize as list of triples for JSON safety
    # (tuple keys aren't JSON-native)
    pos_list = d["positions"]
    assert len(pos_list) == 4
    assert ["1", "1", "node-d"] in [
        [str(p[0]), str(p[1]), p[2]] for p in pos_list
    ]
    restored = TopologyAssignment.from_dict(d)
    assert restored == a


def test_assignment_stable_hash():
    """Same positions → same hash; different positions →
    different hash. Used for de-dup in TopologyHistory."""
    a1 = TopologyAssignment(
        positions={(0, 0): "node-a", (1, 0): "node-b"},
        stage_count=2,
        slots_per_stage=1,
    )
    a2 = TopologyAssignment(
        positions={(0, 0): "node-a", (1, 0): "node-b"},
        stage_count=2,
        slots_per_stage=1,
    )
    a3 = TopologyAssignment(
        positions={(0, 0): "node-b", (1, 0): "node-a"},
        stage_count=2,
        slots_per_stage=1,
    )
    assert a1.stable_hash() == a2.stable_hash()
    assert a1.stable_hash() != a3.stable_hash()


# ── TopologySelector.select ──────────────────────────────


def test_selector_basic_fills_all_positions():
    sel = TopologySelector()
    pool = [f"node-{i}" for i in range(10)]
    topo = sel.select(
        node_pool=pool,
        stage_count=2,
        slots_per_stage=3,
        seed=42,
    )
    assert topo.stage_count == 2
    assert topo.slots_per_stage == 3
    assert len(topo.positions) == 6


def test_selector_no_duplicate_node_per_topology():
    """Each node fills at most one position. Defends an
    operator running the same node in two slots."""
    sel = TopologySelector()
    pool = [f"node-{i}" for i in range(10)]
    topo = sel.select(
        node_pool=pool,
        stage_count=2,
        slots_per_stage=3,
        seed=42,
    )
    assigned_nodes = list(topo.positions.values())
    assert len(assigned_nodes) == len(set(assigned_nodes))


def test_selector_seed_deterministic():
    """Same seed + pool + dims → same topology. Required
    for verifiable on-chain rotation downstream."""
    sel = TopologySelector()
    pool = [f"node-{i}" for i in range(10)]
    t1 = sel.select(
        node_pool=pool, stage_count=2,
        slots_per_stage=2, seed=12345,
    )
    t2 = sel.select(
        node_pool=pool, stage_count=2,
        slots_per_stage=2, seed=12345,
    )
    assert t1.stable_hash() == t2.stable_hash()


def test_selector_different_seeds_different_topologies():
    sel = TopologySelector()
    pool = [f"node-{i}" for i in range(10)]
    t1 = sel.select(
        node_pool=pool, stage_count=2,
        slots_per_stage=2, seed=1,
    )
    t2 = sel.select(
        node_pool=pool, stage_count=2,
        slots_per_stage=2, seed=2,
    )
    # Statistically very high probability of being different
    # with 10! permutations
    assert t1.stable_hash() != t2.stable_hash()


def test_selector_insufficient_pool_raises():
    """Need at least stage_count × slots_per_stage nodes."""
    sel = TopologySelector()
    pool = [f"node-{i}" for i in range(3)]
    with pytest.raises(ValueError):
        sel.select(
            node_pool=pool,
            stage_count=2,
            slots_per_stage=3,  # needs 6 nodes
            seed=42,
        )


def test_selector_validates_stage_count_positive():
    sel = TopologySelector()
    with pytest.raises(ValueError):
        sel.select(
            node_pool=["a", "b"],
            stage_count=0,
            slots_per_stage=1,
            seed=42,
        )


def test_selector_validates_slots_positive():
    sel = TopologySelector()
    with pytest.raises(ValueError):
        sel.select(
            node_pool=["a", "b"],
            stage_count=1,
            slots_per_stage=0,
            seed=42,
        )


def test_selector_empty_pool_raises():
    sel = TopologySelector()
    with pytest.raises(ValueError):
        sel.select(
            node_pool=[],
            stage_count=1,
            slots_per_stage=1,
            seed=42,
        )


# ── TopologyHistory ──────────────────────────────────────


def test_history_empty():
    h = TopologyHistory(max_entries=5)
    assert h.count() == 0
    assert h.recent_hashes() == []


def test_history_appends_and_orders():
    h = TopologyHistory(max_entries=5)
    a1 = TopologyAssignment(
        positions={(0, 0): "a"},
        stage_count=1, slots_per_stage=1,
    )
    a2 = TopologyAssignment(
        positions={(0, 0): "b"},
        stage_count=1, slots_per_stage=1,
    )
    h.record(a1)
    h.record(a2)
    assert h.count() == 2
    # Newest first
    assert h.recent_hashes()[0] == a2.stable_hash()


def test_history_evicts_oldest_at_capacity():
    h = TopologyHistory(max_entries=2)
    a1 = TopologyAssignment(
        positions={(0, 0): "a"},
        stage_count=1, slots_per_stage=1,
    )
    a2 = TopologyAssignment(
        positions={(0, 0): "b"},
        stage_count=1, slots_per_stage=1,
    )
    a3 = TopologyAssignment(
        positions={(0, 0): "c"},
        stage_count=1, slots_per_stage=1,
    )
    h.record(a1)
    h.record(a2)
    h.record(a3)
    hashes = h.recent_hashes()
    # a1 should be evicted
    assert a1.stable_hash() not in hashes
    assert a3.stable_hash() in hashes
    assert a2.stable_hash() in hashes


def test_history_contains_predicate():
    h = TopologyHistory(max_entries=5)
    a1 = TopologyAssignment(
        positions={(0, 0): "a"},
        stage_count=1, slots_per_stage=1,
    )
    h.record(a1)
    assert h.contains(a1) is True
    a2 = TopologyAssignment(
        positions={(0, 0): "b"},
        stage_count=1, slots_per_stage=1,
    )
    assert h.contains(a2) is False


def test_history_max_entries_validation():
    with pytest.raises(ValueError):
        TopologyHistory(max_entries=0)
    with pytest.raises(ValueError):
        TopologyHistory(max_entries=-1)


# ── TopologyRotationPolicy strategies ────────────────────


def test_uniform_random_strategy():
    policy = TopologyRotationPolicy(
        strategy="uniform_random",
    )
    pool = [f"node-{i}" for i in range(6)]
    topo = policy.next_topology(
        node_pool=pool,
        stage_count=2,
        slots_per_stage=2,
        history=TopologyHistory(max_entries=5),
        seed_hint=1,
    )
    assert len(topo.positions) == 4


def test_beacon_seeded_strategy_deterministic():
    """Seeded strategy MUST be deterministic from
    (pool, seed_hint, dims) so on-chain verifiers can replay
    the selection."""
    policy = TopologyRotationPolicy(strategy="beacon_seeded")
    pool = [f"node-{i}" for i in range(6)]
    h1 = TopologyHistory(max_entries=5)
    h2 = TopologyHistory(max_entries=5)
    t1 = policy.next_topology(
        node_pool=pool, stage_count=2,
        slots_per_stage=2, history=h1,
        seed_hint=12345,
    )
    t2 = policy.next_topology(
        node_pool=pool, stage_count=2,
        slots_per_stage=2, history=h2,
        seed_hint=12345,
    )
    assert t1.stable_hash() == t2.stable_hash()


def test_anti_repeat_strategy_avoids_history():
    """anti_repeat picks topologies NOT in the recent
    history window."""
    policy = TopologyRotationPolicy(
        strategy="anti_repeat",
        anti_repeat_window=3,
    )
    pool = [f"node-{i}" for i in range(6)]
    history = TopologyHistory(max_entries=10)

    # Build up 3 topologies in history
    for i in range(3):
        t = policy.next_topology(
            node_pool=pool, stage_count=2,
            slots_per_stage=2, history=history,
            seed_hint=i,
        )
        history.record(t)

    # Issue 5 more — none should duplicate any of the last 3
    for i in range(3, 8):
        t = policy.next_topology(
            node_pool=pool, stage_count=2,
            slots_per_stage=2, history=history,
            seed_hint=i,
        )
        last_3 = history.recent_hashes()[:3]
        assert t.stable_hash() not in last_3
        history.record(t)


def test_anti_repeat_falls_back_when_pool_too_small():
    """If the pool is so small that anti_repeat can't find
    a distinct topology within reasonable attempts, the
    policy should fall back (raise or return best-effort)
    rather than infinite-loop."""
    policy = TopologyRotationPolicy(
        strategy="anti_repeat",
        anti_repeat_window=10,
    )
    # 2 nodes, 1 stage, 2 slots — only 2 possible topologies
    # (which is fewer than the anti_repeat_window=10). Should
    # raise rather than loop forever.
    pool = ["a", "b"]
    history = TopologyHistory(max_entries=10)
    for i in range(2):
        t = policy.next_topology(
            node_pool=pool, stage_count=1,
            slots_per_stage=2, history=history,
            seed_hint=i,
        )
        history.record(t)
    with pytest.raises(
        ValueError, match="anti_repeat",
    ):
        policy.next_topology(
            node_pool=pool, stage_count=1,
            slots_per_stage=2, history=history,
            seed_hint=2,
        )


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        TopologyRotationPolicy(strategy="not_a_strategy")


# ── verify_topology_sequence ─────────────────────────────


def test_verify_valid_sequence():
    topologies = []
    sel = TopologySelector()
    pool = [f"node-{i}" for i in range(10)]
    for seed in range(5):
        topologies.append(
            sel.select(
                node_pool=pool, stage_count=2,
                slots_per_stage=2, seed=seed,
            )
        )
    ok, reason = verify_topology_sequence(
        topologies,
        expected_anti_repeat_window=3,
    )
    assert ok is True


def test_verify_rejects_duplicate_node_position():
    """A topology with the same node in two positions should
    be rejected (structurally invalid)."""
    bad = TopologyAssignment(
        positions={
            (0, 0): "node-a",
            (0, 1): "node-a",  # duplicate
            (1, 0): "node-b",
            (1, 1): "node-c",
        },
        stage_count=2,
        slots_per_stage=2,
    )
    ok, reason = verify_topology_sequence(
        [bad], expected_anti_repeat_window=0,
    )
    assert ok is False
    assert (
        "duplicate" in reason.lower()
        or "duplicated" in reason.lower()
    )


def test_verify_rejects_missing_position():
    """All (stage, slot) combinations within bounds MUST be
    filled. Missing positions indicate a malformed topology."""
    bad = TopologyAssignment(
        positions={
            (0, 0): "node-a",
            (1, 0): "node-b",
            # Missing (0, 1) and (1, 1)
        },
        stage_count=2,
        slots_per_stage=2,
    )
    ok, reason = verify_topology_sequence(
        [bad], expected_anti_repeat_window=0,
    )
    assert ok is False
    assert (
        "missing" in reason.lower()
        or "incomplete" in reason.lower()
    )


def test_verify_rejects_repeat_within_window():
    """A topology repeated within the anti_repeat window
    fails verification."""
    a = TopologyAssignment(
        positions={(0, 0): "node-a"},
        stage_count=1, slots_per_stage=1,
    )
    b = TopologyAssignment(
        positions={(0, 0): "node-b"},
        stage_count=1, slots_per_stage=1,
    )
    sequence = [a, b, a]  # 'a' repeats at distance 2
    ok, reason = verify_topology_sequence(
        sequence,
        expected_anti_repeat_window=3,
    )
    assert ok is False
    assert "repeat" in reason.lower()


def test_verify_accepts_repeat_outside_window():
    """A repeat outside the window is allowed."""
    a = TopologyAssignment(
        positions={(0, 0): "node-a"},
        stage_count=1, slots_per_stage=1,
    )
    b = TopologyAssignment(
        positions={(0, 0): "node-b"},
        stage_count=1, slots_per_stage=1,
    )
    c = TopologyAssignment(
        positions={(0, 0): "node-c"},
        stage_count=1, slots_per_stage=1,
    )
    sequence = [a, b, c, a]  # 'a' repeats at distance 3
    ok, reason = verify_topology_sequence(
        sequence,
        expected_anti_repeat_window=2,
    )
    assert ok is True
