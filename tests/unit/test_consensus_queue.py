"""Unit tests for ConsensusChallengeQueue + process_submittable_queue runner.

Uses in-memory SQLite (`:memory:`) for test isolation — each test gets
a fresh queue, nothing leaks between tests. Runner tests mock the
submitter entirely.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.marketplace.consensus_queue import (
    ConsensusChallengeQueue,
    PendingChallenge,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SUBMITTABLE,
    STATUS_SUBMITTED,
)
from prsm.marketplace.consensus_submitter import (
    ChallengeResult,
    process_submittable_queue,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _receipt(provider_id: str, output_hash_hex: str = None) -> ShardExecutionReceipt:
    if output_hash_hex is None:
        output_hash_hex = ("aa" * 32) if provider_id.startswith("maj") else ("bb" * 32)
    return ShardExecutionReceipt(
        job_id="job-phase7.1x",
        shard_index=0,
        provider_id=provider_id,
        provider_pubkey_b64=f"PUBKEY_{provider_id}",
        output_hash=output_hash_hex,
        executed_at_unix=1_700_000_000,
        signature=f"SIG_{provider_id}",
    )


def _drained_entry(
    *,
    minorities: list[str] = None,
    majorities: list[str] = None,
    job_id: str = "job-phase7.1x",
    shard_index: int = 0,
) -> dict:
    # Distinguish "caller passed nothing" (use defaults) from "caller
    # passed []" (intentional empty list — keep it).
    if minorities is None:
        minorities = ["provC"]
    if majorities is None:
        majorities = ["majA", "majB"]
    return {
        "job_id": job_id,
        "shard_index": shard_index,
        "agreed_output_hash": "aa" * 32,
        "majority_receipts": [_receipt(p, "aa" * 32) for p in majorities],
        "minority_receipts": [_receipt(p, "bb" * 32) for p in minorities],
        "value_ftns_per_provider_wei": 5 * 10**18,
    }


@pytest.fixture
def queue():
    q = ConsensusChallengeQueue(":memory:")
    yield q
    q.close()


# ── enqueue_from_drained ─────────────────────────────────────────────


def test_enqueue_one_minority_one_majority_creates_one_row(queue):
    row_ids = queue.enqueue_from_drained([
        _drained_entry(minorities=["provC"], majorities=["majA"]),
    ])
    assert len(row_ids) == 1
    row = queue.get(row_ids[0])
    assert row.status == STATUS_PENDING
    assert row.minority_provider_id == "provC"
    assert row.majority_provider_id == "majA"
    assert row.minority_batch_id is None
    assert row.majority_batch_id is None
    assert row.attempts == 0


def test_enqueue_expands_multiple_minorities_into_multiple_rows(queue):
    """k=5 with 3-2 split → 2 minorities → 2 separate rows,
    each paired with the same (canonical) majority."""
    row_ids = queue.enqueue_from_drained([
        _drained_entry(
            minorities=["provD", "provE"],
            majorities=["majA", "majB", "majC"],
        ),
    ])
    assert len(row_ids) == 2
    rows = [queue.get(r) for r in row_ids]
    # Both paired with the lexicographically lowest majority provider.
    assert {r.majority_provider_id for r in rows} == {"majA"}
    # Different minority providers in each row.
    assert {r.minority_provider_id for r in rows} == {"provD", "provE"}


def test_enqueue_skips_entry_with_no_majority(queue):
    """Entry with empty majority_receipts → no row. Don't stall drain."""
    entry = _drained_entry(minorities=["provC"], majorities=[])
    row_ids = queue.enqueue_from_drained([entry])
    assert row_ids == []
    assert queue.count_by_status() == {}


def test_enqueue_picks_lowest_provider_id_as_canonical_majority(queue):
    """Deterministic majority selection — lowest provider_id wins.
    Audit replay reproduces the same pairing."""
    row_ids = queue.enqueue_from_drained([
        _drained_entry(
            minorities=["provC"],
            majorities=["majZ", "majA", "majM"],
        ),
    ])
    row = queue.get(row_ids[0])
    assert row.majority_provider_id == "majA"


def test_enqueue_multiple_entries_preserves_order(queue):
    queue.enqueue_from_drained([
        _drained_entry(minorities=["p1"], shard_index=0),
        _drained_entry(minorities=["p2"], shard_index=1),
        _drained_entry(minorities=["p3"], shard_index=2),
    ])
    pending = queue.list_pending()
    assert [r.minority_provider_id for r in pending] == ["p1", "p2", "p3"]
    assert [r.shard_index for r in pending] == [0, 1, 2]


# ── record_batch_commit ──────────────────────────────────────────────


def test_record_batch_commit_sets_minority_batch_id(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    touched = queue.record_batch_commit(
        "provC", "job-phase7.1x", 0, b"\x01" * 32,
    )
    assert touched >= 1
    row = queue.get(row_ids[0])
    assert row.minority_batch_id == b"\x01" * 32
    assert row.majority_batch_id is None
    # Still PENDING because majority hasn't committed.
    assert row.status == STATUS_PENDING


def test_record_batch_commit_sets_majority_batch_id(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    row = queue.get(row_ids[0])
    assert row.majority_batch_id == b"\x02" * 32
    assert row.status == STATUS_PENDING   # minority still missing


def test_record_batch_commit_promotes_to_submittable_when_both_set(queue):
    """Once BOTH batch_ids are recorded, status flips to SUBMITTABLE
    so the runner picks the row up."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    submittable = queue.list_submittable()
    assert len(submittable) == 1
    assert submittable[0].status == STATUS_SUBMITTABLE
    assert submittable[0].minority_batch_id == b"\x01" * 32
    assert submittable[0].majority_batch_id == b"\x02" * 32


def test_record_batch_commit_commit_order_does_not_matter(queue):
    """Majority-commit-first then minority-commit works the same as
    the reverse. Both orderings must land in SUBMITTABLE."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    assert len(queue.list_submittable()) == 1


def test_record_batch_commit_is_idempotent_on_same_provider(queue):
    """Calling record_batch_commit twice with the same provider doesn't
    double-set the batch_id or create duplicate work."""
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    # Second call with a different batch_id must NOT overwrite — we
    # already know this provider's batch for this (job, shard).
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x09" * 32)
    row = queue.get(row_ids[0])
    assert row.minority_batch_id == b"\x01" * 32   # still the first value


def test_record_batch_commit_does_nothing_for_unrelated_provider(queue):
    queue.enqueue_from_drained([_drained_entry()])
    touched = queue.record_batch_commit(
        "someone-else", "job-phase7.1x", 0, b"\x03" * 32,
    )
    assert touched == 0


def test_record_batch_commit_rejects_wrong_batch_id_length(queue):
    with pytest.raises(ValueError, match="32 bytes"):
        queue.record_batch_commit("p", "j", 0, b"\x01" * 16)


def test_record_batch_commit_multi_shard_scoping(queue):
    """A provider participates in multiple shards of the same job.
    A commit for shard 0 must NOT affect rows for shard 1."""
    queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
    ])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    pending = queue.list_pending()
    rows_by_shard = {r.shard_index: r for r in pending}
    assert rows_by_shard[0].minority_batch_id == b"\x01" * 32
    assert rows_by_shard[1].minority_batch_id is None


# ── list_submittable / list_pending ─────────────────────────────────


def test_list_submittable_orders_oldest_first(queue):
    """Runner processes FIFO — ensures pending work doesn't starve."""
    queue.enqueue_from_drained([_drained_entry(shard_index=0)])
    time.sleep(0.01)  # guarantee distinct created_at values
    queue.enqueue_from_drained([_drained_entry(shard_index=1)])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.record_batch_commit("provC", "job-phase7.1x", 1, b"\x03" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 1, b"\x04" * 32)

    rows = queue.list_submittable()
    assert [r.shard_index for r in rows] == [0, 1]


def test_list_submittable_limit(queue):
    for i in range(5):
        queue.enqueue_from_drained([_drained_entry(shard_index=i)])
        queue.record_batch_commit("provC", "job-phase7.1x", i, b"\x01" * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, b"\x02" * 32)
    rows = queue.list_submittable(limit=3)
    assert len(rows) == 3


def test_list_pending_includes_both_pending_and_submittable(queue):
    queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
    ])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    # Shard 0 is SUBMITTABLE, shard 1 is still PENDING.
    pending = queue.list_pending()
    statuses = {r.status for r in pending}
    assert statuses == {STATUS_PENDING, STATUS_SUBMITTABLE}
    assert len(pending) == 2


# ── mark_submitted / mark_failed ────────────────────────────────────


def test_mark_submitted_transitions_to_terminal_state(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_submitted(row_ids[0], "0x" + "ab" * 32)

    row = queue.get(row_ids[0])
    assert row.status == STATUS_SUBMITTED
    assert row.last_tx_hash == "0x" + "ab" * 32
    assert row.attempts == 1
    # SUBMITTED rows NOT in list_submittable any more.
    assert queue.list_submittable() == []


def test_mark_failed_terminal_moves_to_failed(queue):
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_failed(
        row_ids[0], "OnChainRevertedError", "ChallengeNotProven",
        terminal=True,
    )
    row = queue.get(row_ids[0])
    assert row.status == STATUS_FAILED
    assert "ChallengeNotProven" in row.last_error
    assert row.attempts == 1


def test_mark_failed_non_terminal_stays_submittable_for_retry(queue):
    """BroadcastFailedError (RPC transient) → leave submittable so the
    next runner tick retries. Attempts counter increments."""
    row_ids = queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_failed(
        row_ids[0], "BroadcastFailedError", "rpc unreachable",
        terminal=False,
    )
    row = queue.get(row_ids[0])
    assert row.status == STATUS_SUBMITTABLE
    assert row.attempts == 1
    assert "rpc unreachable" in row.last_error


def test_count_by_status(queue):
    row_ids = queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
        _drained_entry(shard_index=2),
    ])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    queue.mark_submitted(row_ids[0], "0xaa")
    counts = queue.count_by_status()
    assert counts.get(STATUS_SUBMITTED) == 1
    assert counts.get(STATUS_PENDING) == 2


# ── Persistence across instances (the whole point) ─────────────────


def test_queue_survives_reopen(tmp_path):
    """Crash-safety: write to a file-backed DB, close it, reopen, and
    the pending work is still there. This is the §8.6 mitigation —
    a submitter crash mid-flight doesn't drop challenges."""
    db_path = str(tmp_path / "consensus.sqlite")
    q1 = ConsensusChallengeQueue(db_path)
    row_ids = q1.enqueue_from_drained([_drained_entry()])
    q1.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    q1.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)
    q1.close()

    # Reopen the DB — as if the process crashed and restarted.
    q2 = ConsensusChallengeQueue(db_path)
    try:
        rows = q2.list_submittable()
        assert len(rows) == 1
        assert rows[0].row_id == row_ids[0]
        assert rows[0].minority_batch_id == b"\x01" * 32
        assert rows[0].majority_batch_id == b"\x02" * 32
        # Receipt fields rehydrate correctly through JSON round-trip.
        assert isinstance(rows[0].minority_receipt, ShardExecutionReceipt)
        assert rows[0].minority_receipt.provider_id == "provC"
    finally:
        q2.close()


# ── process_submittable_queue runner ────────────────────────────────


def _stub_submitter(results: list[ChallengeResult]) -> MagicMock:
    sub = MagicMock()
    sub.submit_one = MagicMock(side_effect=results)
    return sub


def test_runner_processes_submittable_and_marks_submitted(queue):
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=True, tx_hash_hex="0xabc", error_type=None,
            error_message=None,
        ),
    ])
    results = process_submittable_queue(queue, submitter)
    assert len(results) == 1
    assert results[0].success is True
    # Row transitioned to SUBMITTED.
    assert queue.count_by_status() == {STATUS_SUBMITTED: 1}


def test_runner_marks_onchain_reverted_as_terminal(queue):
    """Contract revert → terminal FAILED. No retry."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="OnChainRevertedError",
            error_message="ChallengeNotProven",
        ),
    ])
    process_submittable_queue(queue, submitter)
    assert queue.count_by_status() == {STATUS_FAILED: 1}


def test_runner_marks_broadcast_failed_as_retryable(queue):
    """Transient RPC failure → SUBMITTABLE for next tick."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex=None,
            error_type="BroadcastFailedError",
            error_message="rpc unreachable",
        ),
    ])
    process_submittable_queue(queue, submitter)
    assert queue.count_by_status() == {STATUS_SUBMITTABLE: 1}
    row = queue.list_submittable()[0]
    assert row.attempts == 1


def test_runner_marks_onchain_pending_as_terminal_needs_manual(queue):
    """OnChainPending is UNSAFE to auto-retry — tx may land. Mark
    terminal so a human reconciles via tx_hash rather than
    double-submitting."""
    queue.enqueue_from_drained([_drained_entry()])
    queue.record_batch_commit("provC", "job-phase7.1x", 0, b"\x01" * 32)
    queue.record_batch_commit("majA", "job-phase7.1x", 0, b"\x02" * 32)

    submitter = _stub_submitter([
        ChallengeResult(
            success=False, tx_hash_hex="0xmaybe",
            error_type="OnChainPendingError",
            error_message="timeout",
        ),
    ])
    process_submittable_queue(queue, submitter)
    assert queue.count_by_status() == {STATUS_FAILED: 1}


def test_runner_processes_multiple_rows_independently(queue):
    """Each row's outcome is independent — one terminal failure must
    not block subsequent successes."""
    queue.enqueue_from_drained([
        _drained_entry(shard_index=0),
        _drained_entry(shard_index=1),
        _drained_entry(shard_index=2),
    ])
    for i in range(3):
        queue.record_batch_commit("provC", "job-phase7.1x", i, bytes([i + 1]) * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, bytes([i + 4]) * 32)

    submitter = _stub_submitter([
        ChallengeResult(success=True, tx_hash_hex="0xaa",
                        error_type=None, error_message=None),
        ChallengeResult(success=False, tx_hash_hex=None,
                        error_type="OnChainRevertedError", error_message="x"),
        ChallengeResult(success=True, tx_hash_hex="0xcc",
                        error_type=None, error_message=None),
    ])
    results = process_submittable_queue(queue, submitter)
    assert [r.success for r in results] == [True, False, True]
    counts = queue.count_by_status()
    assert counts.get(STATUS_SUBMITTED) == 2
    assert counts.get(STATUS_FAILED) == 1


def test_runner_on_empty_queue_is_noop(queue):
    submitter = MagicMock()
    results = process_submittable_queue(queue, submitter)
    assert results == []
    submitter.submit_one.assert_not_called()


def test_runner_respects_limit(queue):
    for i in range(5):
        queue.enqueue_from_drained([_drained_entry(shard_index=i)])
        queue.record_batch_commit("provC", "job-phase7.1x", i, bytes([i + 1]) * 32)
        queue.record_batch_commit("majA", "job-phase7.1x", i, bytes([i + 10]) * 32)

    submitter = _stub_submitter([
        ChallengeResult(success=True, tx_hash_hex="0x", error_type=None,
                        error_message=None) for _ in range(2)
    ])
    results = process_submittable_queue(queue, submitter, limit=2)
    assert len(results) == 2
    assert queue.count_by_status().get(STATUS_SUBMITTED) == 2
    # 3 still submittable for the next tick.
    assert queue.count_by_status().get(STATUS_SUBMITTABLE) == 3
