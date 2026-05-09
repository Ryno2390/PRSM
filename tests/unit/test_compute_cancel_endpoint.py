"""POST /compute/cancel/{job_id} — operator-side job cancellation.

Closes the B8 deferred sub-item: operator UX gap where in-progress
jobs ran to completion even when the operator wanted to abort.

Behavior:
  - 503 if neither JobHistoryStore nor PaymentEscrow wired
  - 404 if neither has the job
  - 409 if job already terminal (history.status in COMPLETED/FAILED/
    CANCELLED, or escrow.status in RELEASED/REFUNDED)
  - 200 if cancellable: marks history.status = CANCELLED (when
    history present) AND refunds the escrow (when escrow PENDING)
  - v1 caveat: in-flight Python coroutines are NOT interrupted —
    cancellation marks intent + refunds the budget. If the
    coroutine completes successfully later it'll fail at
    release_escrow_split (REFUNDED → EscrowAlreadyFinalizedError),
    which is the correct race-loss outcome.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryRecord,
    JobHistoryStore,
    JobStatus,
)
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _entry(
    *,
    job_id: str = "forge-abc",
    amount: float = 5.0,
    status: EscrowStatus = EscrowStatus.PENDING,
) -> EscrowEntry:
    return EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id="test-requester",
        amount=amount,
        status=status,
    )


def _record(
    *,
    job_id: str = "forge-abc",
    status: JobStatus = JobStatus.IN_PROGRESS,
) -> JobHistoryRecord:
    return JobHistoryRecord(
        job_id=job_id,
        query=f"q-{job_id}",
        status=status,
        started_at=time.time(),
    )


def _node_with(*, history=None, escrow=None, refund_returns=True,
               refund_raises=None):
    """Build a node stub with optional history + escrow surfaces.
    refund_returns controls async refund_escrow's return value;
    refund_raises injects an exception into refund_escrow."""
    node = MagicMock()
    node.identity.node_id = "test-node"

    if history is None:
        node._job_history = None
    else:
        node._job_history = history

    if escrow is None:
        node._payment_escrow = None
    else:
        node._payment_escrow = escrow
        # Wire async refund_escrow
        if refund_raises is not None:
            escrow.refund_escrow = AsyncMock(side_effect=refund_raises)
        else:
            escrow.refund_escrow = AsyncMock(return_value=refund_returns)
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Service availability
# ──────────────────────────────────────────────────────────────────────


class TestCancelServiceAvailability:
    def test_503_when_neither_subsystem_wired(self):
        node = _node_with()  # no history, no escrow
        resp = _client(node).post("/compute/cancel/forge-x")
        assert resp.status_code == 503
        assert "Neither" in resp.json()["detail"]


# ──────────────────────────────────────────────────────────────────────
# 404 — unknown job
# ──────────────────────────────────────────────────────────────────────


class TestCancelUnknown:
    def test_404_when_neither_has_the_job(self):
        history = JobHistoryStore()
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node_with(history=history, escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-unknown")
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────────────
# 409 — already terminal
# ──────────────────────────────────────────────────────────────────────


class TestCancelAlreadyTerminal:
    def test_409_when_history_completed(self):
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.COMPLETED))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node_with(history=history, escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 409
        assert "completed" in resp.json()["detail"].lower()

    def test_409_when_history_failed(self):
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.FAILED))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node_with(history=history, escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 409

    def test_409_when_escrow_already_released(self):
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            status=EscrowStatus.RELEASED,
        ))
        node = _node_with(escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 409

    def test_409_when_escrow_already_refunded(self):
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            status=EscrowStatus.REFUNDED,
        ))
        node = _node_with(escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 409


# ──────────────────────────────────────────────────────────────────────
# 200 — successful cancellation
# ──────────────────────────────────────────────────────────────────────


class TestCancelSuccess:
    def test_in_progress_job_with_pending_escrow_cancels_both(self):
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.IN_PROGRESS))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            amount=12.5, status=EscrowStatus.PENDING,
        ))
        node = _node_with(history=history, escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "forge-abc"
        assert body["history_cancelled"] is True
        assert body["escrow_refunded"] is True
        assert body["refund_amount_ftns"] == 12.5
        # History record reflects new CANCELLED state.
        assert history.get("forge-abc").status == JobStatus.CANCELLED
        # refund_escrow was awaited.
        escrow_mock.refund_escrow.assert_awaited_once()

    def test_pending_escrow_only_no_history_cancels_escrow(self):
        """Job exists in escrow but not history (e.g., legacy
        fixture or LRU eviction). Cancel still works on the
        escrow leg."""
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            amount=3.0, status=EscrowStatus.PENDING,
        ))
        node = _node_with(escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["history_cancelled"] is False
        assert body["escrow_refunded"] is True
        assert body["refund_amount_ftns"] == 3.0

    def test_in_progress_history_only_no_escrow_marks_cancelled(self):
        """Job has IN_PROGRESS history but no escrow entry (e.g.,
        zero-budget test fixture). Mark history CANCELLED but
        don't try to refund."""
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.IN_PROGRESS))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=None)
        node = _node_with(history=history, escrow=escrow_mock)
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["history_cancelled"] is True
        assert body["escrow_refunded"] is False
        assert body["refund_amount_ftns"] == 0.0
        # refund_escrow was NOT called (no PENDING escrow).
        escrow_mock.refund_escrow.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────
# Refund-side failure modes
# ──────────────────────────────────────────────────────────────────────


class TestCancelRefundFailures:
    def test_refund_returning_false_does_not_break_response(self):
        """refund_escrow returns False if escrow lookup race-lost
        (e.g., release_escrow ran in another fiber between
        cancel's check + refund call). Surface escrow_refunded:
        False but still mark history CANCELLED — partial
        cancellation is informative."""
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.IN_PROGRESS))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            status=EscrowStatus.PENDING,
        ))
        node = _node_with(
            history=history, escrow=escrow_mock,
            refund_returns=False,
        )
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["history_cancelled"] is True
        assert body["escrow_refunded"] is False

    def test_refund_raising_does_not_break_response(self):
        """refund_escrow raises EscrowAlreadyFinalizedError if the
        in-flight coroutine reached release_escrow_split between
        our check + our refund call. Treat as race-loss + return
        200 with escrow_refunded: False."""
        from prsm.node.payment_escrow import EscrowAlreadyFinalizedError
        history = JobHistoryStore()
        history.put(_record(status=JobStatus.IN_PROGRESS))
        escrow_mock = MagicMock()
        escrow_mock.get_escrow = MagicMock(return_value=_entry(
            status=EscrowStatus.PENDING,
        ))
        node = _node_with(
            history=history, escrow=escrow_mock,
            refund_raises=EscrowAlreadyFinalizedError("race"),
        )
        resp = _client(node).post("/compute/cancel/forge-abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["history_cancelled"] is True
        assert body["escrow_refunded"] is False


# ──────────────────────────────────────────────────────────────────────
# JobStatus.CANCELLED enum value
# ──────────────────────────────────────────────────────────────────────


class TestCancelledEnumValue:
    def test_cancelled_status_in_enum(self):
        # Enum extension must be additive — existing values intact.
        assert JobStatus.CANCELLED.value == "cancelled"
        assert JobStatus.IN_PROGRESS.value == "in_progress"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
