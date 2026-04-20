"""Phase 3 integration test — 3-node marketplace end-to-end.

Roadmap acceptance criterion (docs/2026-04-20-phase3-marketplace-design.md §11):
  A 3-node local cluster with one requester and two providers advertising
  different prices successfully runs a sharded inference where:
    1. Orchestrator discovers both providers via gossip.
    2. EligibilityFilter with max_price_per_shard_ftns=0.05 correctly
       excludes the 0.10 FTNS provider and includes the 0.03 provider.
    3. Price handshake completes for every shard.
    4. Shards execute via Phase 2 dispatcher.
    5. Receipts verify (Phase 2 binding checks).
    6. Reputation tracker records success + latency percentiles.
    7. Assembled output matches local baseline bit-identically.
    8. Escrow state correct (N RELEASED, 0 REFUNDED).

Design §3.5: this test exercises the full local-ledger settlement path.
On-chain settlement (Phase 3.1) is out of scope.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys

import numpy as np
import pytest

from prsm.compute.model_sharding.executor import execute_shard_locally
from prsm.compute.model_sharding.models import ModelShard
from prsm.marketplace.policy import DispatchPolicy
from prsm.node.local_ledger import TransactionType
from prsm.node.payment_escrow import EscrowStatus

sys.path.insert(0, os.path.dirname(__file__))
from conftest_phase3 import spin_up_three_node_marketplace_cluster  # noqa: E402


def _make_shard(index: int, total: int, rows: np.ndarray) -> ModelShard:
    tensor_bytes = rows.tobytes()
    return ModelShard(
        shard_id=f"shard-{index}",
        model_id="phase3-test-model",
        shard_index=index,
        total_shards=total,
        tensor_data=tensor_bytes,
        tensor_shape=tuple(rows.shape),
        size_bytes=len(tensor_bytes),
        checksum=hashlib.sha256(tensor_bytes).hexdigest(),
    )


@pytest.mark.asyncio
async def test_phase3_marketplace_end_to_end_price_filtering():
    """The core acceptance test. Two providers, different prices;
    policy excludes the expensive one; all shards land on the cheap one.

    Row-parallel: full tensor (6, 8) split along axis 0 into 2 shards
    of (3, 8). Each shard computes tensor @ input_vec → (3,).
    Concatenate axis=0 → (6,).
    """
    nodes = await spin_up_three_node_marketplace_cluster(
        provider_prices=[0.03, 0.03, 0.10],  # requester_price irrelevant, B=0.03, C=0.10
    )
    requester, provider_b, provider_c = nodes[0], nodes[1], nodes[2]

    # Start advertisers on B and C so they broadcast listings.
    # Requester's own advertiser would drop its node_id into its own
    # directory — we skip starting it so the orchestrator's eligible
    # pool is just {B, C}, matching the test narrative.
    await provider_b.advertiser.start()
    await provider_c.advertiser.start()
    # Let the gossip fan-out tasks scheduled via call_soon run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Directory on the requester should now see both.
    assert requester.directory.size() == 2

    # Seed requester wallet so escrow can debit.
    await requester.ledger.credit(
        wallet_id=requester.identity.node_id,
        amount=100.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="e2e test seed",
    )

    rng = np.random.default_rng(seed=7)
    full_tensor = rng.random((6, 8), dtype=np.float64)
    input_tensor = rng.random(8, dtype=np.float64)
    expected_output = full_tensor @ input_tensor

    shard_rows = 3
    shards = [
        _make_shard(i, 2, full_tensor[i * shard_rows:(i + 1) * shard_rows, :])
        for i in range(2)
    ]

    policy = DispatchPolicy(
        max_price_per_shard_ftns=0.05,  # excludes C (0.10)
        min_price_per_shard_ftns=0.01,
        required_dtype="float64",
    )

    result = await requester.orchestrator.orchestrate_sharded_inference(
        shards=shards,
        input_tensor=input_tensor,
        job_id="phase3-e2e-1",
        policy=policy,
    )

    # Assertion 1: bit-identical vs local baseline.
    np.testing.assert_array_equal(result, expected_output)

    # Assertion 2: price filtering — all traffic landed on cheap provider B.
    rep_b = requester.reputation.get_reputation(provider_b.identity.node_id)
    rep_c = requester.reputation.get_reputation(provider_c.identity.node_id)
    assert rep_b is not None
    assert len(rep_b.successful_dispatches) == 2  # both shards
    # C shouldn't have been touched at all — no reputation entry created.
    assert rep_c is None

    # Assertion 3: escrow state.
    escrows = list(requester.payment_escrow._escrows.values())
    released = [e for e in escrows if e.status == EscrowStatus.RELEASED]
    assert len(released) == 2
    assert all(e.status == EscrowStatus.RELEASED for e in escrows)

    # Assertion 4: ledger balances.
    # Requester paid 2 * 0.03 = 0.06 FTNS; B received 0.06; C got nothing.
    assert abs(
        await requester.ledger.get_balance(provider_b.identity.node_id) - 0.06
    ) < 1e-9
    assert await requester.ledger.get_balance(provider_c.identity.node_id) == 0.0
    assert abs(
        await requester.ledger.get_balance(requester.identity.node_id) - (100.0 - 0.06)
    ) < 1e-9

    # Assertion 5: latency stats recorded.
    assert requester.reputation.latency_p50(provider_b.identity.node_id) is not None
    assert requester.reputation.latency_p50(provider_b.identity.node_id) >= 0

    # Stop advertisers cleanly.
    await provider_b.advertiser.stop()
    await provider_c.advertiser.stop()
