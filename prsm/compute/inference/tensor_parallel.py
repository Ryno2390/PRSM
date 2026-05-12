"""Sprint 316 — tensor parallelism math primitive (v1).

Sprints 312-315 ship PIPELINE parallelism: split a model's
LAYERS across nodes; activations flow SEQUENTIALLY. This
module introduces the sibling concept — TENSOR parallelism:
split a single layer's WEIGHTS across nodes; activations
flow in PARALLEL through shards, then are concatenated or
reduced.

The Megatron-style "ColumnParallelLinear" pattern shipped
here:
  - Weight W of shape (in, out) is split column-wise into
    N partial weights W_k of shape (in, out/N)
  - Each node computes y_k = X @ W_k locally
  - Outputs are concatenated column-wise along the last
    dim: y = concat(y_0, ..., y_{N-1})
  - Mathematically equivalent to single-node y = X @ W

v1 simulates parallel forward IN-PROCESS. Real cross-node
all-reduce / concat-over-HTTP = sprint 316a. The
correctness property that "sharded forward equals
monolithic forward" (load-bearing for distributed
deployment) is established here.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


# ── Splitting helpers ──────────────────────────────


def split_weight_column_parallel(
    W: torch.Tensor, *, n_parts: int,
) -> List[torch.Tensor]:
    """Split a 2D weight matrix column-wise into n_parts
    contiguous shards. First shard absorbs the remainder
    when out_features % n_parts != 0."""
    if n_parts < 1:
        raise ValueError(
            f"n_parts must be >= 1; got {n_parts}"
        )
    if W.dim() != 2:
        raise ValueError(
            f"weight must be 2D (in, out); got shape "
            f"{tuple(W.shape)}"
        )
    out_features = W.shape[1]
    if n_parts > out_features:
        raise ValueError(
            f"n_parts ({n_parts}) > out_features "
            f"({out_features}); reduce parallelism"
        )
    base = out_features // n_parts
    remainder = out_features % n_parts
    shards: List[torch.Tensor] = []
    cursor = 0
    for i in range(n_parts):
        size = base + (1 if i < remainder else 0)
        shards.append(W[:, cursor:cursor + size])
        cursor += size
    return shards


# ── Sharded forward ────────────────────────────────


def forward_column_parallel_sharded(
    x: torch.Tensor, weight_shards: List[torch.Tensor],
) -> torch.Tensor:
    """Run X @ W as N parallel partial matmuls + column-
    concat. Each shard is shape (in_features, k_i); inputs
    are shape (..., in_features); output is shape (...,
    sum_i(k_i)).

    In v1 the parallel forward is simulated in-process —
    each shard's matmul runs sequentially on the local
    machine. Distributed deployment hands each shard to a
    different worker (sprint 316a wires the cross-node
    transport)."""
    if not weight_shards:
        raise ValueError(
            "weight_shards must be non-empty"
        )
    first_in = weight_shards[0].shape[0]
    for i, s in enumerate(weight_shards):
        if s.shape[0] != first_in:
            raise ValueError(
                f"weight_shards must agree on in_features; "
                f"shard {i} has {s.shape[0]} but first "
                f"shard has {first_in}"
            )
    partial_outputs = [x @ shard for shard in weight_shards]
    return torch.cat(partial_outputs, dim=-1)


# ── ColumnParallelLinear nn.Module wrapper ─────────


class ColumnParallelLinear(nn.Module):
    """nn.Linear-like layer whose weight is internally
    sharded. Forward runs the sharded matmul + concat.
    Without bias for v1 (matches the Megatron pattern;
    bias is added post-concat).

    weight: optional override of shape (in_features,
        out_features). If None, sample from N(0, 1).
        NOTE: the standard nn.Linear stores weight as
        (out, in); ColumnParallelLinear takes (in, out)
        so the column-parallel split is along the
        out-dimension axis. Callers passing an nn.Linear
        weight should `.t()` it before handing in.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        n_parts: int,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parts = n_parts
        if weight is None:
            weight = torch.randn(
                in_features, out_features,
            )
        else:
            if tuple(weight.shape) != (
                in_features, out_features,
            ):
                raise ValueError(
                    f"weight shape "
                    f"{tuple(weight.shape)} must be "
                    f"(in_features, out_features) = "
                    f"({in_features}, {out_features})"
                )
        # Store as a single full-weight tensor; the shards
        # are derived per-call. Operators wanting to
        # physically distribute the shards extract via
        # .weight_shards
        self.weight = nn.Parameter(
            weight.clone(),
            requires_grad=False,
        )

    @property
    def weight_shards(self) -> List[torch.Tensor]:
        return split_weight_column_parallel(
            self.weight.data, n_parts=self.n_parts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return forward_column_parallel_sharded(
            x, self.weight_shards,
        )
