"""JobReaper — per-job duration cap enforcement."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.node.job_reaper import JobReaper
from prsm.node.job_history import (
    JobHistoryRecord, JobHistoryStore, JobStatus,
)


def _record(job_id, *, status=JobStatus.IN_PROGRESS, started_at=None):
    return JobHistoryRecord(
        job_id=job_id, query=f"q-{job_id}", status=status,
        started_at=time.time() if started_at is None else started_at,
    )


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_max_duration_must_be_positive(self):
        with pytest.raises(ValueError):
            JobReaper(
                job_history=JobHistoryStore(),
                payment_escrow=None,
                max_duration_seconds=0,
            )

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError):
            JobReaper(
                job_history=JobHistoryStore(),
                payment_escrow=None,
                max_duration_seconds=60.0,
                interval_seconds=0,
            )


# ──────────────────────────────────────────────────────────────────────
# Reap behavior
# ──────────────────────────────────────────────────────────────────────


class TestReapOnce:
    @pytest.mark.asyncio
    async def test_old_in_progress_jobs_marked_failed(self):
        history = JobHistoryStore()
        old_ts = time.time() - 600  # 10 min ago
        history.put(_record("forge-old", started_at=old_ts))
        history.put(_record("forge-recent", started_at=time.time()))

        reaper = JobReaper(
            job_history=history,
            payment_escrow=None,
            max_duration_seconds=300,  # 5 min
        )
        reaped = await reaper.reap_once()
        assert reaped == 1
        # Old job is FAILED with duration-cap reason
        old_rec = history.get("forge-old")
        assert old_rec.status == JobStatus.FAILED
        assert "duration cap exceeded" in old_rec.error
        # Recent job untouched
        recent_rec = history.get("forge-recent")
        assert recent_rec.status == JobStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_recent_jobs_not_reaped(self):
        history = JobHistoryStore()
        for i in range(3):
            history.put(_record(f"forge-{i}"))  # all just-now
        reaper = JobReaper(
            job_history=history,
            payment_escrow=None,
            max_duration_seconds=300,
        )
        reaped = await reaper.reap_once()
        assert reaped == 0

    @pytest.mark.asyncio
    async def test_completed_jobs_not_reaped(self):
        """Even if completed_at is old, reaper only touches
        IN_PROGRESS records."""
        history = JobHistoryStore()
        history.put(_record(
            "forge-1", status=JobStatus.COMPLETED,
            started_at=time.time() - 600,
        ))
        reaper = JobReaper(
            job_history=history,
            payment_escrow=None,
            max_duration_seconds=300,
        )
        reaped = await reaper.reap_once()
        assert reaped == 0

    @pytest.mark.asyncio
    async def test_refunds_associated_escrow(self):
        history = JobHistoryStore()
        history.put(_record("forge-old", started_at=time.time() - 600))
        escrow = MagicMock()
        escrow.refund_escrow = AsyncMock(return_value=True)
        reaper = JobReaper(
            job_history=history,
            payment_escrow=escrow,
            max_duration_seconds=300,
        )
        await reaper.reap_once()
        escrow.refund_escrow.assert_awaited_once()
        # Reason should reference the cap.
        call = escrow.refund_escrow.await_args
        assert "duration cap exceeded" in str(call)

    @pytest.mark.asyncio
    async def test_refund_failure_does_not_break_reap(self):
        """If refund_escrow raises (race-loss with release), the
        history is still marked FAILED."""
        from prsm.node.payment_escrow import EscrowAlreadyFinalizedError
        history = JobHistoryStore()
        history.put(_record("forge-old", started_at=time.time() - 600))
        escrow = MagicMock()
        escrow.refund_escrow = AsyncMock(
            side_effect=EscrowAlreadyFinalizedError("race"),
        )
        reaper = JobReaper(
            job_history=history,
            payment_escrow=escrow,
            max_duration_seconds=300,
        )
        reaped = await reaper.reap_once()
        assert reaped == 1
        assert history.get("forge-old").status == JobStatus.FAILED


# ──────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_run_forever_can_be_stopped(self):
        reaper = JobReaper(
            job_history=JobHistoryStore(),
            payment_escrow=None,
            max_duration_seconds=300,
            interval_seconds=0.05,
        )
        task = asyncio.create_task(reaper.run_forever())
        await asyncio.sleep(0.15)  # let it tick at least once
        await reaper.stop()
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()
