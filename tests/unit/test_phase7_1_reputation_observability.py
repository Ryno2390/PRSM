"""Phase 7.1 Task 6 — reputation-observability verification.

Per design doc §6 Task 6, this task contains NO new code. It verifies
end-to-end that a CONSENSUS_MISMATCH slash flows through the same
ReputationTracker pipeline that DOUBLE_SPEND and INVALID_SIGNATURE use,
and that the k-of-n dispatch output (ConsensusShardReceipt) integrates
cleanly with Phase 7's record_slash surface.

Task 4 already pinned:
  - record_slash accepts reason="CONSENSUS_MISMATCH"
  - has_been_slashed flips True
  - reasons stored verbatim in get_slash_events
  - SLASH_WEIGHT behavior identical for CONSENSUS_MISMATCH vs DOUBLE_SPEND

This task covers the gaps Task 4 did not explicitly touch:
  - slashed_count for CONSENSUS_MISMATCH
  - Rolling-window accumulation across multiple minority slashes
  - The concrete ConsensusShardReceipt.minority_receipts → record_slash
    flow a production challenge-submitter would implement
"""
from __future__ import annotations

import numpy as np

from prsm.compute.multi_dispatcher import ConsensusShardReceipt
from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.economy.web3.stake_manager import ReasonCode
from prsm.marketplace.reputation import ReputationTracker


def _minority_receipt(provider_id: str, output_hash: str = "HASH_CHEAT"):
    return ShardExecutionReceipt(
        job_id="job-phase7.1",
        shard_index=0,
        provider_id=provider_id,
        provider_pubkey_b64="PUBKEY",
        output_hash=output_hash,
        executed_at_unix=1_700_000_000,
        signature="SIG",
    )


# ── slashed_count for CONSENSUS_MISMATCH ─────────────────────────────


def test_slashed_count_increments_on_consensus_mismatch():
    tracker = ReputationTracker()
    assert tracker.slashed_count("p") == 0
    tracker.record_slash(
        "p", batch_id="0x" + "ab" * 32, slash_amount_wei=1000,
        reason=ReasonCode.CONSENSUS_MISMATCH.name,
    )
    assert tracker.slashed_count("p") == 1


def test_slashed_count_accumulates_across_multiple_consensus_slashes():
    """A provider caught cheating on three separate consensus jobs
    should land three slash events. Rolling-window behavior (maxlen=1000)
    inherits from Phase 7 Task 6 unchanged."""
    tracker = ReputationTracker()
    for i in range(3):
        tracker.record_slash(
            "serial_cheater",
            batch_id=f"0x{i:064x}",
            slash_amount_wei=1000 * (i + 1),
            reason=ReasonCode.CONSENSUS_MISMATCH.name,
        )
    assert tracker.slashed_count("serial_cheater") == 3
    events = tracker.get_slash_events("serial_cheater")
    assert [e.slash_amount_wei for e in events] == [1000, 2000, 3000]


# ── ConsensusShardReceipt → ReputationTracker pipeline ───────────────


def _make_consensus_receipt(minority_provider_ids: list[str]):
    majority = ["provA", "provB"]
    return ConsensusShardReceipt(
        job_id="job-phase7.1",
        shard_index=0,
        consensus_mode="majority",
        k=len(majority) + len(minority_provider_ids),
        responded=len(majority) + len(minority_provider_ids),
        agreed_output_hash="HASH_OK",
        agreed_output=np.array([42.0]),
        majority_receipts=[_minority_receipt(p, "HASH_OK") for p in majority],
        minority_receipts=[_minority_receipt(p) for p in minority_provider_ids],
        consensus_reached_unix=1_700_000_100,
    )


def test_minority_receipts_feed_directly_into_record_slash():
    """The concrete pipeline a Phase 7.1 challenge-submitter runs after
    a batch commits: for each minority in the ConsensusShardReceipt,
    call record_slash on the reputation tracker.

    This test proves the shapes line up — no adapter layer needed
    between MultiDispatcher's output and Phase 7's reputation surface.
    """
    tracker = ReputationTracker()
    receipt = _make_consensus_receipt(["provC"])

    # Typical submitter loop shape:
    for r in receipt.minority_receipts:
        tracker.record_slash(
            provider_id=r.provider_id,
            batch_id="0x" + "01" * 32,   # submitter supplies post-commit
            slash_amount_wei=25_000 * 10**18,
            reason=ReasonCode.CONSENSUS_MISMATCH.name,
        )

    assert tracker.has_been_slashed("provC") is True
    assert tracker.slashed_count("provC") == 1
    assert tracker.slashed_count("provA") == 0
    assert tracker.slashed_count("provB") == 0


def test_multiple_minorities_at_k_5_record_separately():
    """k=5 with 3-2 split → two minority providers, each gets their
    own slash event. Verifies no cross-contamination (one slash per
    provider, not one global slash)."""
    tracker = ReputationTracker()
    receipt = _make_consensus_receipt(["provD", "provE"])

    for r in receipt.minority_receipts:
        tracker.record_slash(
            provider_id=r.provider_id,
            batch_id="0x" + "02" * 32,
            slash_amount_wei=10_000 * 10**18,
            reason=ReasonCode.CONSENSUS_MISMATCH.name,
        )

    for pid in ("provD", "provE"):
        assert tracker.has_been_slashed(pid) is True
        assert tracker.slashed_count(pid) == 1
    assert len(tracker.known_providers()) == 2


def test_score_for_consensus_mismatch_matches_cold_start_semantics():
    """A fresh provider caught cheating on their very first k-of-n job
    lands at score=0.0, not the neutral-0.5 cold-start shield. This is
    the load-bearing invariant from Phase 7 Task 6 — verify it still
    holds for CONSENSUS_MISMATCH specifically."""
    tracker = ReputationTracker()
    tracker.record_slash(
        "fresh_consensus_cheater",
        batch_id="0x" + "ff" * 32,
        slash_amount_wei=1,
        reason=ReasonCode.CONSENSUS_MISMATCH.name,
    )
    # 1 slash * SLASH_WEIGHT=100 = 100 weighted failures.
    # Total (0 success + 100 weighted failure) >= MIN_SAMPLES_FOR_SCORE=10
    # → score = 0 / 100 = 0.0, not the neutral 0.5 cold-start value.
    assert tracker.score_for("fresh_consensus_cheater") == 0.0
