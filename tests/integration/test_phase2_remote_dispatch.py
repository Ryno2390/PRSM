"""Phase 2 integration test — 3-node sharded inference end-to-end.

Roadmap acceptance criterion: "3-node local cluster runs a sharded
inference end-to-end."

Design Q3 decision: in-process with real transport (option B). Three
Phase2TestNode instances share a loopback transport hub and a single
LocalLedger. The requester dispatches shards 1 and 2 remotely to
providers B and C; shard 0 runs locally. Assembled output is compared
bit-for-bit against a local baseline.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from prsm.compute.model_sharding.executor import execute_shard_locally
from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.node.local_ledger import TransactionType
from prsm.node.payment_escrow import EscrowStatus

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from conftest_phase2 import spin_up_three_node_cluster  # noqa: E402


def _make_shard(index: int, total: int, tensor_rows: np.ndarray) -> ModelShard:
    tensor_bytes = tensor_rows.tobytes()
    return ModelShard(
        shard_id=f"shard-{index}",
        model_id="phase2-test-model",
        shard_index=index,
        total_shards=total,
        tensor_data=tensor_bytes,
        tensor_shape=tuple(tensor_rows.shape),
        size_bytes=len(tensor_bytes),
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )


@pytest.mark.asyncio
async def test_three_node_sharded_inference_end_to_end():
    """Three nodes, each owning one row-parallel shard. Requester runs
    shard 0 locally, dispatches shard 1 to B and shard 2 to C. Assembled
    output matches local baseline bit-identically.

    Row-parallel: full tensor (9, 8) split along axis 0 into 3 shards of
    shape (3, 8). Each shard computes shard @ input_vec -> (3,).
    Concatenate along axis 0 -> (9,).
    """
    nodes = await spin_up_three_node_cluster()
    requester, provider_b, provider_c = nodes[0], nodes[1], nodes[2]

    rng = np.random.default_rng(seed=42)
    full_tensor = rng.random((9, 8), dtype=np.float64)
    input_tensor = rng.random(8, dtype=np.float64)
    expected_output = full_tensor @ input_tensor

    shard_rows = 3
    shards = [
        _make_shard(i, 3, full_tensor[i * shard_rows:(i + 1) * shard_rows, :])
        for i in range(3)
    ]

    await requester.ledger.credit(
        wallet_id=requester.identity.node_id,
        amount=100.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="test seed",
    )
    initial_requester_balance = await requester.ledger.get_balance(
        requester.identity.node_id
    )
    assert initial_requester_balance == 100.0

    input_bytes = input_tensor.tobytes()

    shard_outputs = []
    shard_outputs.append(execute_shard_locally(shards[0], input_bytes))

    out_b = await requester.remote_shard_dispatcher.dispatch(
        shard=shards[1],
        input_tensor=input_tensor,
        node_id=provider_b.identity.node_id,
        job_id="integration-job-1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_ftns=1.0,
    )
    shard_outputs.append(out_b)

    out_c = await requester.remote_shard_dispatcher.dispatch(
        shard=shards[2],
        input_tensor=input_tensor,
        node_id=provider_c.identity.node_id,
        job_id="integration-job-1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_ftns=1.0,
    )
    shard_outputs.append(out_c)

    full_output = np.concatenate(shard_outputs, axis=0)

    # Use assert_allclose rather than assert_array_equal — different BLAS
    # implementations (OpenBLAS on Ubuntu CI vs Accelerate on Apple
    # Silicon dev machines) produce results that differ by 1-2 ulps due
    # to floating-point operation ordering. Tolerance of 1e-12 is vastly
    # tighter than any semantically meaningful difference while forgiving
    # these benign numerical artifacts.
    np.testing.assert_allclose(full_output, expected_output, rtol=1e-12, atol=1e-12)

    escrows = list(requester.payment_escrow._escrows.values())
    b_escrows = [e for e in escrows if e.job_id.endswith(":shard:1")]
    c_escrows = [e for e in escrows if e.job_id.endswith(":shard:2")]
    assert len(b_escrows) == 1
    assert len(c_escrows) == 1
    assert b_escrows[0].status == EscrowStatus.RELEASED
    assert c_escrows[0].status == EscrowStatus.RELEASED

    assert await requester.ledger.get_balance(provider_b.identity.node_id) == 1.0
    assert await requester.ledger.get_balance(provider_c.identity.node_id) == 1.0
    assert await requester.ledger.get_balance(requester.identity.node_id) == 98.0
