"""Per-job duration cap enforcement via background reaper.

When PRSM_FORGE_MAX_DURATION_SEC is set, this reaper periodically
scans JobHistoryStore for IN_PROGRESS records older than the
cap and:
  1. Marks the history record FAILED with error="duration cap exceeded"
  2. Refunds the associated escrow (if PENDING)

Decouples duration enforcement from /compute/forge's complex
body. The forge handler keeps its existing semantics; the reaper
provides a separate timeout safety net.

The reaper is a long-running asyncio task — same lifecycle
pattern as periodic_cleanup_escrows. Operators can monitor
liveness via the daemon-task probe surfaces shipped earlier in
this session (cleanup_task_running pattern).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)


_DEFAULT_INTERVAL_SECONDS = 60.0


class JobReaper:
    """Reaps IN_PROGRESS JobHistoryRecords older than ``max_duration_seconds``.

    Required deps (constructor args):
      job_history: JobHistoryStore (for list+put)
      payment_escrow: PaymentEscrow (for refund_escrow)

    Optional:
      max_duration_seconds: cap above which jobs are reaped
      interval_seconds: how often to scan (default 60s)
    """

    def __init__(
        self,
        *,
        job_history: Any,
        payment_escrow: Optional[Any],
        max_duration_seconds: float,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        if max_duration_seconds <= 0:
            raise ValueError(
                f"max_duration_seconds must be positive, "
                f"got {max_duration_seconds}"
            )
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, "
                f"got {interval_seconds}"
            )
        self._job_history = job_history
        self._payment_escrow = payment_escrow
        self._max_duration_seconds = max_duration_seconds
        self._interval_seconds = interval_seconds
        self._running = False

    @property
    def max_duration_seconds(self) -> float:
        return self._max_duration_seconds

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def reap_once(self) -> int:
        """One-shot scan + reap. Returns count of records reaped."""
        # Lazy import — avoid circular at module load.
        from prsm.node.job_history import JobStatus, JobHistoryRecord

        cutoff = time.time() - self._max_duration_seconds
        try:
            in_progress = self._job_history.list(
                status_filter=JobStatus.IN_PROGRESS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("JobReaper list raised: %s", exc)
            return 0

        reaped = 0
        for rec in in_progress:
            if rec.started_at >= cutoff:
                continue  # within cap, skip
            # Mark history FAILED with reaper marker.
            try:
                failed_record = JobHistoryRecord(
                    job_id=rec.job_id,
                    query=rec.query,
                    status=JobStatus.FAILED,
                    started_at=rec.started_at,
                    completed_at=time.time(),
                    route=rec.route,
                    response=rec.response,
                    aggregator_node_id=rec.aggregator_node_id,
                    contributing_shards=rec.contributing_shards,
                    participants=rec.participants,
                    traces_collected=rec.traces_collected,
                    error=(
                        f"duration cap exceeded "
                        f"({self._max_duration_seconds}s)"
                    ),
                )
                self._job_history.put(failed_record)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JobReaper history.put failed for %s: %s",
                    rec.job_id, exc,
                )
                continue
            # Refund escrow if PENDING.
            if self._payment_escrow is not None:
                try:
                    await self._payment_escrow.refund_escrow(
                        rec.job_id,
                        reason=(
                            f"duration cap exceeded "
                            f"({self._max_duration_seconds}s)"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    # refund_escrow raises EscrowAlreadyFinalizedError
                    # in the race-loss case; not a hard failure for
                    # the reaper (history is now FAILED regardless).
                    logger.info(
                        "JobReaper refund skipped for %s: %s",
                        rec.job_id, exc,
                    )
            reaped += 1
        if reaped > 0:
            logger.info(
                "JobReaper: reaped %d job(s) over duration cap",
                reaped,
            )
        return reaped

    async def run_forever(self) -> None:
        """Long-running loop. Same lifecycle pattern as
        PaymentEscrow.periodic_cleanup."""
        self._running = True
        while self._running:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self.reap_once()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "JobReaper loop error: %s", exc,
                )

    async def stop(self) -> None:
        self._running = False
