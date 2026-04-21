"""Unit tests for ShardConsensus — Phase 7.1 Task 1.

Pure-function resolver. No I/O, no mocks needed beyond synthetic
ShardExecutionReceipts.
"""
from __future__ import annotations

import pytest

from prsm.compute.shard_consensus import ConsensusOutcome, ShardConsensus
from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.node.result_consensus import ConsensusMode


def _receipt(
    provider_id: str,
    output_hash: str,
    *,
    job_id: str = "job-phase7.1",
    shard_index: int = 0,
) -> ShardExecutionReceipt:
    return ShardExecutionReceipt(
        job_id=job_id,
        shard_index=shard_index,
        provider_id=provider_id,
        provider_pubkey_b64="PUBKEY",
        output_hash=output_hash,
        executed_at_unix=1_700_000_000,
        signature="SIG",
    )


# ── Constructor validation ────────────────────────────────────────────


def test_constructor_rejects_k_zero():
    with pytest.raises(ValueError, match="k must be >= 1"):
        ShardConsensus(ConsensusMode.MAJORITY, k=0)


def test_constructor_rejects_negative_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        ShardConsensus(ConsensusMode.MAJORITY, k=-1)


def test_constructor_rejects_byzantine_mode():
    """Phase 7.1 MVP explicitly stubs BYZANTINE — surface at construction
    so callers don't silently think it's wired."""
    with pytest.raises(NotImplementedError, match="BYZANTINE"):
        ShardConsensus(ConsensusMode.BYZANTINE, k=4)


def test_constructor_rejects_non_enum_mode():
    with pytest.raises(TypeError, match="ConsensusMode"):
        ShardConsensus("majority", k=3)  # string instead of enum


# ── MAJORITY mode ────────────────────────────────────────────────────


def test_majority_3_of_3_all_agree():
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    receipts = [
        _receipt("provA", "HASH_OK"),
        _receipt("provB", "HASH_OK"),
        _receipt("provC", "HASH_OK"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is True
    assert outcome.agreed_output_hash == "HASH_OK"
    assert len(outcome.majority) == 3
    assert outcome.minority == []


def test_majority_2_of_3_with_one_minority():
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    receipts = [
        _receipt("provA", "HASH_OK"),
        _receipt("provB", "HASH_OK"),
        _receipt("provC", "HASH_CHEAT"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is True
    assert outcome.agreed_output_hash == "HASH_OK"
    assert {r.provider_id for r in outcome.majority} == {"provA", "provB"}
    assert [r.provider_id for r in outcome.minority] == ["provC"]


def test_majority_all_disagree_fails_consensus():
    """Three receipts, three different hashes → no group meets threshold
    AND top-groups tie at count 1. No agreement."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    receipts = [
        _receipt("provA", "HASH_A"),
        _receipt("provB", "HASH_B"),
        _receipt("provC", "HASH_C"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is False
    assert outcome.agreed_output_hash is None
    assert outcome.majority == []
    assert outcome.minority == []


def test_majority_3_of_5_threshold():
    """At k=5, MAJORITY threshold is 3. Test the boundary exactly."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=5)
    receipts = [
        _receipt("p1", "HASH_OK"),
        _receipt("p2", "HASH_OK"),
        _receipt("p3", "HASH_OK"),
        _receipt("p4", "HASH_X"),
        _receipt("p5", "HASH_Y"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is True
    assert outcome.agreed_output_hash == "HASH_OK"
    assert len(outcome.majority) == 3
    assert len(outcome.minority) == 2


def test_majority_tie_at_even_k_fails():
    """k=4, 2-2 split → no unambiguous winner. Consensus fails even
    though both groups meet (k//2)+1 = 3? Actually (4//2)+1 = 3, neither
    group reaches 3. But the tie check also protects against cases
    where both groups could tie at threshold in principle."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=4)
    receipts = [
        _receipt("p1", "HASH_A"),
        _receipt("p2", "HASH_A"),
        _receipt("p3", "HASH_B"),
        _receipt("p4", "HASH_B"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is False


def test_majority_partial_response_still_meets_threshold():
    """k=3 requested but only 2 responded; both agree. That's >= 2
    (the MAJORITY threshold for k=3) → consensus holds."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    receipts = [
        _receipt("provA", "HASH_OK"),
        _receipt("provB", "HASH_OK"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is True
    assert len(outcome.majority) == 2


def test_majority_partial_response_below_threshold():
    """k=3 requested but only 1 responded. (3//2)+1 = 2, 1 < 2 → fails."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    outcome = resolver.resolve([_receipt("provA", "HASH_OK")])
    assert outcome.agreed is False


# ── UNANIMOUS mode ───────────────────────────────────────────────────


def test_unanimous_all_k_agree():
    resolver = ShardConsensus(ConsensusMode.UNANIMOUS, k=3)
    receipts = [
        _receipt("p1", "HASH_OK"),
        _receipt("p2", "HASH_OK"),
        _receipt("p3", "HASH_OK"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is True
    assert len(outcome.majority) == 3


def test_unanimous_one_disagrees_fails():
    """Stricter than MAJORITY — 2-of-3 agreement is NOT acceptable."""
    resolver = ShardConsensus(ConsensusMode.UNANIMOUS, k=3)
    receipts = [
        _receipt("p1", "HASH_OK"),
        _receipt("p2", "HASH_OK"),
        _receipt("p3", "HASH_CHEAT"),
    ]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is False


def test_unanimous_requires_full_k_response():
    """All existing receipts agree but response count < k → fails.
    UNANIMOUS is stricter than MAJORITY's partial-response tolerance."""
    resolver = ShardConsensus(ConsensusMode.UNANIMOUS, k=3)
    outcome = resolver.resolve([
        _receipt("p1", "HASH_OK"),
        _receipt("p2", "HASH_OK"),
    ])
    assert outcome.agreed is False


# ── SINGLE mode ──────────────────────────────────────────────────────


def test_single_trivially_agrees_with_one_receipt():
    """Phase 7's default: k=1, no consensus involved. Included in
    ShardConsensus for uniform orchestrator routing."""
    resolver = ShardConsensus(ConsensusMode.SINGLE, k=1)
    receipts = [_receipt("provA", "HASH_OK")]
    outcome = resolver.resolve(receipts)
    assert outcome.agreed is True
    assert outcome.agreed_output_hash == "HASH_OK"
    assert len(outcome.majority) == 1
    assert outcome.minority == []


# ── Edge cases ───────────────────────────────────────────────────────


def test_empty_receipts_no_agreement():
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    outcome = resolver.resolve([])
    assert outcome.agreed is False
    assert outcome.agreed_output_hash is None
    assert outcome.majority == []
    assert outcome.minority == []


def test_mixed_job_id_raises():
    """Caller must build the consensus group from a single shard. Mixing
    is a bug we refuse to paper over."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    with pytest.raises(ValueError, match="job_id"):
        resolver.resolve([
            _receipt("p1", "HASH_OK", job_id="job-a"),
            _receipt("p2", "HASH_OK", job_id="job-b"),
        ])


def test_mixed_shard_index_raises():
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    with pytest.raises(ValueError, match="shard_index"):
        resolver.resolve([
            _receipt("p1", "HASH_OK", shard_index=0),
            _receipt("p2", "HASH_OK", shard_index=1),
        ])


def test_outcome_is_frozen_dataclass():
    """ConsensusOutcome must be immutable so orchestrators / auditors
    can pass it around without defensive copying."""
    resolver = ShardConsensus(ConsensusMode.MAJORITY, k=3)
    outcome = resolver.resolve([
        _receipt("p1", "HASH_OK"),
        _receipt("p2", "HASH_OK"),
        _receipt("p3", "HASH_OK"),
    ])
    with pytest.raises((AttributeError, Exception)):
        outcome.agreed = False  # frozen dataclass refuses
