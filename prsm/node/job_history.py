"""B8 async-dispatch follow-on — JobHistoryRecord + JobHistoryStore.

The synchronous-from-caller-view ``/compute/forge`` pipeline
returns the answer in-band. Once async dispatch lands, callers
will need a way to retrieve a job's result *after* the dispatch
call has returned. ``JobHistoryRecord`` is that surface: a
per-job record capturing the load-bearing state (route,
response, aggregator, participants, timing, error if failed).

``/compute/status/{job_id}`` reads from this store first (richer
data) and falls back to ``PaymentEscrow`` (escrow lifecycle
only) when the job isn't in history.

v1 is in-memory LRU-bounded — sufficient for the synchronous
path that's currently the only path live. Filesystem persistence
is the right v2 once jobs span beyond a single node-startup
window.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    """Lifecycle states for a /compute/forge job.

    Mirrors the escrow lifecycle but with semantics tied to the
    compute pipeline rather than the payment leg:
      - IN_PROGRESS: pipeline started; result not yet available.
      - COMPLETED: pipeline finished; ``response`` populated.
      - FAILED: pipeline raised; ``error`` populated.
    """
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ──────────────────────────────────────────────────────────────────────
# Record
# ──────────────────────────────────────────────────────────────────────


@dataclass
class JobHistoryRecord:
    """One job's compute-pipeline state. Distinct from the escrow
    record (which tracks payment) — together they form the full
    audit picture surfaced via ``/compute/status``.
    """

    job_id: str
    query: str
    status: JobStatus
    started_at: float

    # Lifecycle timing
    completed_at: Optional[float] = None

    # Result fields (populated on COMPLETED)
    route: Optional[str] = None
    response: Optional[str] = None
    aggregator_node_id: Optional[str] = None
    contributing_shards: Tuple[str, ...] = field(default_factory=tuple)
    participants: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    traces_collected: int = 0

    # Error field (populated on FAILED)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        if self.started_at < 0:
            raise ValueError(
                f"started_at must be non-negative, got {self.started_at}"
            )
        if self.completed_at is not None:
            if self.completed_at < self.started_at:
                raise ValueError(
                    f"completed_at ({self.completed_at}) must be >= "
                    f"started_at ({self.started_at})"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "query": self.query,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "route": self.route,
            "response": self.response,
            "aggregator_node_id": self.aggregator_node_id,
            "contributing_shards": list(self.contributing_shards),
            "participants": list(self.participants),
            "traces_collected": self.traces_collected,
            "error": self.error,
        }


# ──────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_MAX_ENTRIES = 1024


class JobHistoryStore:
    """In-memory LRU-bounded store. Thread-safety: not enforced —
    /compute/forge handles serialize per-job-id naturally (each
    job has a distinct ``forge-<uuid>`` job_id), and the store's
    only multi-thread surface is concurrent reads via
    /compute/status which OrderedDict's GIL-protected get
    handles safely enough for v1.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError(
                f"max_entries must be a positive integer, got {max_entries!r}"
            )
        self._max_entries = max_entries
        # OrderedDict preserves insertion order; move_to_end on
        # get() implements LRU semantics.
        self._records: "OrderedDict[str, JobHistoryRecord]" = OrderedDict()

    def put(self, record: JobHistoryRecord) -> None:
        """Insert or update a record. Updates promote to most-
        recently-used; new inserts may trigger LRU eviction of the
        oldest entry."""
        if record.job_id in self._records:
            # Overwrite + promote.
            self._records[record.job_id] = record
            self._records.move_to_end(record.job_id)
            return
        self._records[record.job_id] = record
        while len(self._records) > self._max_entries:
            self._records.popitem(last=False)  # evict oldest

    def get(self, job_id: str) -> Optional[JobHistoryRecord]:
        """Look up a record. Promotes to most-recently-used."""
        rec = self._records.get(job_id)
        if rec is None:
            return None
        self._records.move_to_end(job_id)
        return rec

    def size(self) -> int:
        return len(self._records)
