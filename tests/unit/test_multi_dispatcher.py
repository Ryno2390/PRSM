"""Unit tests for MultiShardDispatcher — Phase 7.1 Task 2.

Mocks the single RemoteShardDispatcher entirely. Tests verify the
gather-and-consense flow, partial-response handling, and error
classification (InsufficientResponses vs ConsensusFailed).
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock

import numpy as np
import pytest

from prsm.compute.model_sharding.models import ModelShard, PipelineStakeTier
from prsm.compute.multi_dispatcher import (
    ConsensusFailedError,
    ConsensusShardReceipt,
    InsufficientResponsesError,
    MultiShardDispatcher,
)
from prsm.compute.remote_dispatcher import (
    DispatchResult,
    MissingAttestationError,
    PeerNotConnectedError,
    ShardDispatchError,
    ShardPreemptedError,
)
from prsm.compute.shard_consensus import ShardConsensus
from prsm.node.result_consensus import ConsensusMode


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_shard() -> ModelShard:
    tensor = np.array([[1.0, 2.0]], dtype=np.float64)
    tb = tensor.tobytes()
    return ModelShard(
        shard_id="shard-test",
        model_id="m",
        shard_index=0,
        total_shards=1,
        tensor_data=tb,
        tensor_shape=tensor.shape,
        size_bytes=len(tb),
        checksum=hashlib.sha256(tb).hexdigest(),
    )


def _make_dispatch_result(
    node_id: str,
    output_hash: str,
    *,
    output: np.ndarray = None,
    job_id: str = "job-phase7.1",
) -> DispatchResult:
    if output is None:
        output = np.array([1.0, 2.0], dtype=np.float64)
    return DispatchResult(
        output=output,
        receipt={
            "job_id": job_id,
            "shard_index": 0,
            "provider_id": node_id,
            "provider_pubkey_b64": "PUBKEY",
            "output_hash": output_hash,
            "executed_at_unix": 1_700_000_000,
            "signature": "SIG",
        },
        provider_node_id=node_id,
        escrow_amount_ftns=0.05,
    )


def _make_dispatcher(k: int = 3, mode: ConsensusMode = ConsensusMode.MAJORITY):
    single = AsyncMock()
    consensus = ShardConsensus(mode, k=k)
    return MultiShardDispatcher(single_dispatcher=single, consensus=consensus), single


def _run(coro):
    return asyncio.run(coro)


# ── Happy paths ──────────────────────────────────────────────────────


def test_dispatch_with_consensus_3_of_3_all_agree():
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        _make_dispatch_result("provC", "HASH_OK"),
    ])

    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))

    assert isinstance(receipt, ConsensusShardReceipt)
    assert receipt.k == 3
    assert receipt.responded == 3
    assert receipt.agreed_output_hash == "HASH_OK"
    assert len(receipt.majority_receipts) == 3
    assert receipt.minority_receipts == []
    assert receipt.consensus_mode == "majority"


def test_dispatch_with_consensus_2_of_3_one_minority():
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        _make_dispatch_result("provC", "HASH_CHEAT"),
    ])

    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.PREMIUM,
        escrow_amount_per_provider_ftns=0.05,
    ))
    assert receipt.agreed_output_hash == "HASH_OK"
    assert {r.provider_id for r in receipt.majority_receipts} == {"provA", "provB"}
    assert [r.provider_id for r in receipt.minority_receipts] == ["provC"]


# ── Partial failures ─────────────────────────────────────────────────


def test_preemption_excluded_but_majority_still_reached():
    """1 preemption + 2 agreeing responses → (3//2)+1 = 2 threshold
    met. Consensus holds; preempted provider simply didn't contribute."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        ShardPreemptedError(shard_index=0, node_id="provC", reason="spot_evict"),
    ])
    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    assert receipt.responded == 2
    assert receipt.agreed_output_hash == "HASH_OK"
    assert len(receipt.majority_receipts) == 2


def test_tee_missing_excluded():
    """MissingAttestationError is classified as non-response (lying
    provider), not a disagreement. Excluded from consensus group."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        MissingAttestationError("no quote"),
    ])
    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.CRITICAL,
        escrow_amount_per_provider_ftns=0.05,
        require_tee_attestation=True,
    ))
    assert receipt.responded == 2


def test_dispatch_error_excluded():
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        ShardDispatchError("transport error"),
    ])
    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    assert receipt.responded == 2


def test_peer_not_connected_excluded():
    """Phase 7.1 §8.8 audit follow-up: a provider whose peer is offline
    raises PeerNotConnectedError from RemoteShardDispatcher (when no
    local_fallback is wired). Previously MultiDispatcher would treat
    this as a bug and abort the whole gather, which is wrong — offline
    in a k-of-n dispatch is a classic partial-response case. Must be
    classified alongside ShardPreempted."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        PeerNotConnectedError("provC is offline"),
    ])
    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    # Consensus still reached because 2-of-3 met the MAJORITY threshold.
    # PeerNotConnectedError was treated as a partial response, not a bug.
    assert receipt.responded == 2
    assert receipt.agreed_output_hash == "HASH_OK"


def test_empty_receipt_fallback_excluded():
    """Dispatcher's fallback paths (size-too-large or peer-not-connected
    with local_fallback) return DispatchResult with empty receipt dict.
    Can't consense without a signed receipt — exclude."""
    md, single = _make_dispatcher(k=3)

    fallback = DispatchResult(
        output=np.array([1.0], dtype=np.float64),
        receipt={},  # empty — fallback path
        provider_node_id="provC",
        escrow_amount_ftns=0.0,
    )
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        fallback,
    ])
    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    assert receipt.responded == 2


# ── Failure classifications ──────────────────────────────────────────


def test_insufficient_responses_raises():
    """2 preemptions + 1 response: 1 < (3//2)+1 = 2 → InsufficientResponses."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        ShardPreemptedError(shard_index=0, node_id="provB", reason="x"),
        ShardPreemptedError(shard_index=0, node_id="provC", reason="y"),
    ])
    with pytest.raises(InsufficientResponsesError) as excinfo:
        _run(md.dispatch_with_consensus(
            shard=_make_shard(),
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            node_ids=["provA", "provB", "provC"],
            job_id="job-phase7.1",
            stake_tier=PipelineStakeTier.STANDARD,
            escrow_amount_per_provider_ftns=0.05,
        ))
    assert excinfo.value.responded == 1
    assert excinfo.value.threshold == 2
    assert excinfo.value.k == 3


def test_consensus_failed_raises_when_all_disagree():
    """3 responses, 3 different hashes → threshold-count met but no
    agreement. ConsensusFailed, not InsufficientResponses."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_A"),
        _make_dispatch_result("provB", "HASH_B"),
        _make_dispatch_result("provC", "HASH_C"),
    ])
    with pytest.raises(ConsensusFailedError) as excinfo:
        _run(md.dispatch_with_consensus(
            shard=_make_shard(),
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            node_ids=["provA", "provB", "provC"],
            job_id="job-phase7.1",
            stake_tier=PipelineStakeTier.STANDARD,
            escrow_amount_per_provider_ftns=0.05,
        ))
    assert excinfo.value.responded == 3
    assert excinfo.value.unique_hashes == 3


def test_unexpected_exception_propagates():
    """Bugs (non-dispatch exceptions from the single dispatcher) must
    NOT be silently classified as 'didn't respond' — surface them so
    the caller can fix the bug."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        KeyError("oops, a bug"),
    ])
    with pytest.raises(KeyError):
        _run(md.dispatch_with_consensus(
            shard=_make_shard(),
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            node_ids=["provA", "provB", "provC"],
            job_id="job-phase7.1",
            stake_tier=PipelineStakeTier.STANDARD,
            escrow_amount_per_provider_ftns=0.05,
        ))


# ── Caller-contract validation ───────────────────────────────────────


def test_mismatched_node_ids_count_raises():
    md, single = _make_dispatcher(k=3)
    with pytest.raises(ValueError, match="node_ids count"):
        _run(md.dispatch_with_consensus(
            shard=_make_shard(),
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            node_ids=["only", "two"],  # k=3, length=2
            job_id="job-phase7.1",
            stake_tier=PipelineStakeTier.STANDARD,
            escrow_amount_per_provider_ftns=0.05,
        ))


def test_duplicate_node_ids_raises():
    """Same provider twice would give them two votes — never correct."""
    md, single = _make_dispatcher(k=3)
    with pytest.raises(ValueError, match="duplicates"):
        _run(md.dispatch_with_consensus(
            shard=_make_shard(),
            input_tensor=np.array([1.0, 1.0], dtype=np.float64),
            node_ids=["provA", "provB", "provA"],
            job_id="job-phase7.1",
            stake_tier=PipelineStakeTier.STANDARD,
            escrow_amount_per_provider_ftns=0.05,
        ))


def test_parallel_dispatch_uses_asyncio_gather():
    """All k dispatches should fire concurrently, not sequentially.
    Exact concurrency is hard to assert without real timing, but we
    can verify all coros are awaited before any completes by having
    each one await its own event."""
    md, single = _make_dispatcher(k=3)

    started = asyncio.Event()
    # Each mock call blocks until `started` is set. If dispatches were
    # sequential, the first would block forever and the test would
    # hang. Gather releases them together once set.
    async def slow_dispatch(**kwargs):
        # Yield once so gather can schedule all three before any completes
        await asyncio.sleep(0)
        return _make_dispatch_result(kwargs["node_id"], "HASH_OK")

    single.dispatch_with_receipt = slow_dispatch

    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    assert receipt.responded == 3


def test_pre_selected_node_ids_respected():
    """Dispatcher MUST dispatch to exactly the k node_ids the caller
    passed — no re-selection inside MultiDispatcher. Orchestrator's
    _select_top_k owns selection policy."""
    md, single = _make_dispatcher(k=3)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK"),
        _make_dispatch_result("provB", "HASH_OK"),
        _make_dispatch_result("provC", "HASH_OK"),
    ])
    _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    dispatched_ids = [
        call.kwargs["node_id"]
        for call in single.dispatch_with_receipt.await_args_list
    ]
    assert set(dispatched_ids) == {"provA", "provB", "provC"}
    assert single.dispatch_with_receipt.await_count == 3


def test_agreed_output_pulled_from_majority_provider():
    """When providers disagree, the returned agreed_output must be the
    ndarray the MAJORITY returned, not the minority's."""
    md, single = _make_dispatcher(k=3)
    majority_array = np.array([42.0, 42.0], dtype=np.float64)
    minority_array = np.array([99.0, 99.0], dtype=np.float64)
    single.dispatch_with_receipt = AsyncMock(side_effect=[
        _make_dispatch_result("provA", "HASH_OK", output=majority_array),
        _make_dispatch_result("provB", "HASH_OK", output=majority_array),
        _make_dispatch_result("provC", "HASH_CHEAT", output=minority_array),
    ])
    receipt = _run(md.dispatch_with_consensus(
        shard=_make_shard(),
        input_tensor=np.array([1.0, 1.0], dtype=np.float64),
        node_ids=["provA", "provB", "provC"],
        job_id="job-phase7.1",
        stake_tier=PipelineStakeTier.STANDARD,
        escrow_amount_per_provider_ftns=0.05,
    ))
    np.testing.assert_array_equal(receipt.agreed_output, majority_array)
