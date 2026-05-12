"""Sprint 316 — tensor parallelism (math primitive v1).

Sprints 312-315 shipped PIPELINE parallelism: splitting a
model's LAYERS across nodes, with activations flowing
sequentially through stages. This sprint introduces the
sibling concept — TENSOR parallelism: splitting a single
layer's weights (matmul) across nodes, with activations
flowing in PARALLEL through shards.

The Megatron-style "ColumnParallelLinear" pattern:
  - Weight W of shape (in, out) is split column-wise into
    N partial weights W_k of shape (in, out/N)
  - Each node computes y_k = X @ W_k locally
  - Outputs are concatenated column-wise: y = concat(y_0,
    ..., y_{N-1}) along the last dimension
  - Mathematically equivalent to y = X @ W (single-node)

v1 simulates the parallel forward in-process. Real cross-
node all-reduce over HTTP = sprint 316a. The math
primitive in this sprint is load-bearing: ships the
correctness property "sharded forward equals monolithic
forward" that real distributed tensor parallelism must
preserve.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prsm.compute.inference.tensor_parallel import (
    ColumnParallelLinear,
    forward_column_parallel_sharded,
    split_weight_column_parallel,
)


# ── split_weight_column_parallel ───────────────────


def test_split_2way_produces_two_halves():
    W = torch.randn(4, 8)
    shards = split_weight_column_parallel(W, n_parts=2)
    assert len(shards) == 2
    assert shards[0].shape == (4, 4)
    assert shards[1].shape == (4, 4)
    # Concatenation reconstructs W
    assert torch.equal(
        torch.cat(shards, dim=1), W,
    )


def test_split_4way_produces_four_quarters():
    W = torch.randn(8, 16)
    shards = split_weight_column_parallel(W, n_parts=4)
    assert len(shards) == 4
    for s in shards:
        assert s.shape == (8, 4)
    assert torch.equal(
        torch.cat(shards, dim=1), W,
    )


def test_split_uneven_out_features_distributes_remainder():
    """W with 9 output cols split 2 ways: first shard
    gets 5, second gets 4."""
    W = torch.randn(4, 9)
    shards = split_weight_column_parallel(W, n_parts=2)
    assert shards[0].shape == (4, 5)
    assert shards[1].shape == (4, 4)
    assert torch.equal(
        torch.cat(shards, dim=1), W,
    )


def test_split_rejects_n_parts_zero():
    with pytest.raises(ValueError, match="n_parts"):
        split_weight_column_parallel(
            torch.randn(4, 8), n_parts=0,
        )


def test_split_rejects_n_parts_greater_than_out_features():
    with pytest.raises(ValueError, match="n_parts"):
        split_weight_column_parallel(
            torch.randn(4, 3), n_parts=8,
        )


def test_split_rejects_non_2d_tensor():
    with pytest.raises(ValueError, match="2D"):
        split_weight_column_parallel(
            torch.randn(4, 8, 2), n_parts=2,
        )


# ── forward_column_parallel_sharded ────────────────


def test_sharded_forward_matches_monolithic_2way():
    """The load-bearing correctness property: sharded
    forward across 2 nodes produces the IDENTICAL output
    (within float tolerance) as a single-node forward."""
    in_features, out_features = 4, 8
    W = torch.randn(in_features, out_features)
    x = torch.randn(3, in_features)

    monolithic = x @ W
    shards = split_weight_column_parallel(W, n_parts=2)
    sharded = forward_column_parallel_sharded(
        x, shards,
    )
    assert torch.allclose(monolithic, sharded, atol=1e-6)


def test_sharded_forward_matches_monolithic_4way():
    W = torch.randn(8, 16)
    x = torch.randn(2, 8)
    monolithic = x @ W
    shards = split_weight_column_parallel(W, n_parts=4)
    sharded = forward_column_parallel_sharded(
        x, shards,
    )
    assert torch.allclose(monolithic, sharded, atol=1e-6)


def test_sharded_forward_matches_monolithic_8way():
    """8-way split — the math has to compose at higher
    parallelism levels too."""
    W = torch.randn(16, 32)
    x = torch.randn(2, 16)
    monolithic = x @ W
    shards = split_weight_column_parallel(W, n_parts=8)
    sharded = forward_column_parallel_sharded(
        x, shards,
    )
    assert torch.allclose(monolithic, sharded, atol=1e-6)


def test_sharded_forward_handles_3d_input():
    """Real transformer activations are 3D
    (batch, seq, features). The math has to compose
    along the contraction dimension regardless."""
    W = torch.randn(8, 16)
    x = torch.randn(2, 5, 8)  # (batch, seq, in_features)
    monolithic = x @ W
    shards = split_weight_column_parallel(W, n_parts=4)
    sharded = forward_column_parallel_sharded(
        x, shards,
    )
    assert sharded.shape == (2, 5, 16)
    assert torch.allclose(monolithic, sharded, atol=1e-6)


def test_sharded_forward_rejects_empty_shards():
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="shard"):
        forward_column_parallel_sharded(x, [])


def test_sharded_forward_rejects_dimension_mismatch():
    x = torch.randn(2, 4)
    bad_shards = [
        torch.randn(99, 4),  # wrong in_features
    ]
    with pytest.raises(Exception):
        forward_column_parallel_sharded(x, bad_shards)


def test_sharded_forward_rejects_inconsistent_in_features():
    """All shards must share the same in_features
    (because each computes X @ W_k with the same X)."""
    x = torch.randn(2, 4)
    shards = [torch.randn(4, 3), torch.randn(8, 5)]
    with pytest.raises(ValueError, match="in_features"):
        forward_column_parallel_sharded(x, shards)


# ── ColumnParallelLinear nn.Module wrapper ─────────


def test_column_parallel_linear_matches_nn_linear():
    """ColumnParallelLinear should behave identically to
    nn.Linear (without bias for v1) — same input, same
    output, just internally sharded."""
    in_features, out_features = 4, 8
    torch.manual_seed(42)
    standard = torch.nn.Linear(
        in_features, out_features, bias=False,
    )
    cp = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        n_parts=2,
        weight=standard.weight.data.t(),  # (in, out)
    )
    x = torch.randn(3, in_features)
    expected = standard(x)
    got = cp(x)
    assert torch.allclose(expected, got, atol=1e-6)


def test_column_parallel_linear_construction_random_weight():
    """When no weight is supplied, ColumnParallelLinear
    samples its own. The forward pass still works (just
    not matching a specific external nn.Linear)."""
    cp = ColumnParallelLinear(
        in_features=4, out_features=8, n_parts=2,
    )
    x = torch.randn(2, 4)
    y = cp(x)
    assert y.shape == (2, 8)


def test_column_parallel_linear_exposes_shards():
    """For real distributed deployment, operators need to
    extract the per-shard weights to send to each worker
    node. The .weight_shards property exposes them."""
    cp = ColumnParallelLinear(
        in_features=4, out_features=8, n_parts=2,
    )
    assert len(cp.weight_shards) == 2
    assert all(
        s.shape == (4, 4) for s in cp.weight_shards
    )
