"""Sprint 312 — pipeline partition spec.

Splits a model's layers across stages, each stage assigned
to a node. The partition is declarative — it says nothing
about how layers compute, only how they're distributed.

Validation invariants:
  - Total layers covered exactly once (no gaps, no overlaps)
  - Layer indices contiguous within each stage
  - One node_id per stage; n_stages == len(node_ids)
  - At least one layer per stage (no empty stages)
"""
from __future__ import annotations

import json

import pytest

from prsm.compute.inference.pipeline_partition import (
    PipelinePartition,
    PartitionValidationError,
    even_layer_partition,
)


# ── Construction + validation ───────────────────────


def test_valid_4_stage_12_layer():
    p = PipelinePartition(
        total_layers=12,
        stage_layer_ranges=[
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
        ],
        stage_node_ids=["n0", "n1", "n2", "n3"],
    )
    assert p.n_stages == 4
    p.validate()  # no raise


def test_validate_rejects_gap():
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [3]],  # missing 2
        stage_node_ids=["n0", "n1"],
    )
    with pytest.raises(PartitionValidationError, match="2"):
        p.validate()


def test_validate_rejects_overlap():
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1, 2], [2, 3]],
        stage_node_ids=["n0", "n1"],
    )
    with pytest.raises(
        PartitionValidationError, match="overlap|duplicate",
    ):
        p.validate()


def test_validate_rejects_out_of_range_layer():
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [2, 3, 99]],
        stage_node_ids=["n0", "n1"],
    )
    with pytest.raises(
        PartitionValidationError, match="range|99",
    ):
        p.validate()


def test_validate_rejects_negative_layer_index():
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[-1, 0, 1], [2, 3]],
        stage_node_ids=["n0", "n1"],
    )
    with pytest.raises(PartitionValidationError):
        p.validate()


def test_validate_rejects_empty_stage():
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1, 2, 3], []],
        stage_node_ids=["n0", "n1"],
    )
    with pytest.raises(
        PartitionValidationError, match="empty",
    ):
        p.validate()


def test_validate_rejects_non_contiguous_within_stage():
    """Each stage's layers must be a contiguous block.
    Allowing non-contiguous would let layer-0 + layer-7
    coexist on one stage with everything in between on
    another — the pipeline wire becomes ambiguous."""
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 2], [1, 3]],
        stage_node_ids=["n0", "n1"],
    )
    with pytest.raises(
        PartitionValidationError, match="contiguous",
    ):
        p.validate()


def test_validate_rejects_stage_node_mismatch():
    p = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [2, 3]],
        stage_node_ids=["n0"],  # 1 node for 2 stages
    )
    with pytest.raises(
        PartitionValidationError, match="node",
    ):
        p.validate()


def test_validate_rejects_zero_total_layers():
    p = PipelinePartition(
        total_layers=0,
        stage_layer_ranges=[],
        stage_node_ids=[],
    )
    with pytest.raises(
        PartitionValidationError, match="total_layers",
    ):
        p.validate()


# ── Convenience constructor ─────────────────────────


def test_even_partition_basic():
    p = even_layer_partition(
        total_layers=12,
        node_ids=["n0", "n1", "n2", "n3"],
    )
    p.validate()
    assert p.n_stages == 4
    # 12/4 = 3 layers per stage
    assert all(len(s) == 3 for s in p.stage_layer_ranges)


def test_even_partition_uneven_split():
    """13 layers across 4 nodes — first stage gets the
    remainder."""
    p = even_layer_partition(
        total_layers=13,
        node_ids=["n0", "n1", "n2", "n3"],
    )
    p.validate()
    sizes = [len(s) for s in p.stage_layer_ranges]
    assert sum(sizes) == 13
    # 4 + 3 + 3 + 3 = 13
    assert sizes == [4, 3, 3, 3]


def test_even_partition_rejects_more_nodes_than_layers():
    with pytest.raises(ValueError):
        even_layer_partition(
            total_layers=2,
            node_ids=["n0", "n1", "n2"],
        )


def test_even_partition_rejects_empty_node_pool():
    with pytest.raises(ValueError):
        even_layer_partition(
            total_layers=4, node_ids=[],
        )


# ── Serialization ──────────────────────────────────


def test_to_dict_round_trip():
    p = PipelinePartition(
        total_layers=8,
        stage_layer_ranges=[[0, 1, 2, 3], [4, 5, 6, 7]],
        stage_node_ids=["a", "b"],
    )
    restored = PipelinePartition.from_dict(p.to_dict())
    assert restored == p


def test_partition_hash_stable():
    """The partition_hash must be deterministic — same
    partition shape always yields the same hash, regardless
    of field order in the JSON wire format. The orchestrator
    binds this hash into the receipt so verifiers can check
    the partition wasn't substituted."""
    p1 = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [2, 3]],
        stage_node_ids=["n0", "n1"],
    )
    p2 = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [2, 3]],
        stage_node_ids=["n0", "n1"],
    )
    assert p1.partition_hash() == p2.partition_hash()


def test_partition_hash_varies_with_assignment():
    """Different node_ids → different partition_hash (even
    if layer ranges are identical) — preserves attribution
    in the receipt."""
    base = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [2, 3]],
        stage_node_ids=["n0", "n1"],
    )
    swapped = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1], [2, 3]],
        stage_node_ids=["n9", "n0"],
    )
    assert base.partition_hash() != swapped.partition_hash()
