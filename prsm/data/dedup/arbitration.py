"""PRSM-PROV-1 T6.5 — disputed-attribution arbitration queue.

When an upload's similarity to an existing CID lands in the
``[arbitration_floor, derivative_threshold)`` band defined by
``ThresholdResolver``, ``ContentUploader`` does NOT auto-attribute
the candidate parent. Instead it enqueues a
``DisputedAttributionRecord`` here for council review.

Two implementations:
  - ``InMemoryArbitrationQueue`` — unit-test fixture; mirrors
    ``InMemoryModelRegistry`` (Phase 3.x.2) and
    ``InMemoryPrivacyBudgetStore`` (Phase 3.x.4) shapes.
  - ``FilesystemArbitrationQueue`` — node-local persistence; mirrors
    ``FilesystemPrivacyBudgetStore`` (Phase 3.x.4) shape.

Cross-node arbitration via the Phase-6 DHT is parked under R10
(see design doc §"What we deliberately defer"). v1 is single-node.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Record + decision enum
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DisputedAttributionRecord:
    """One disputed-attribution flag, awaiting council adjudication.

    The record captures both candidates (uploader + alleged parent)
    plus the similarity score that landed it in the disputed band.
    The ``proposal_id`` field is set after the Phase-6 governance
    proposal is created; ``None`` means the proposal has not yet
    been opened (e.g. governance not wired in single-node mode).
    """

    new_cid: str
    new_creator: str
    candidate_parent_cid: str
    candidate_parent_creator: str
    similarity: float
    fingerprint_kind: str
    flagged_at: int  # unix seconds
    proposal_id: Optional[str]

    def __post_init__(self) -> None:
        if not self.new_cid:
            raise ValueError("new_cid must not be empty")
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(
                f"similarity must be in [0,1], got {self.similarity!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisputedAttributionRecord":
        return cls(
            new_cid=data["new_cid"],
            new_creator=data["new_creator"],
            candidate_parent_cid=data["candidate_parent_cid"],
            candidate_parent_creator=data["candidate_parent_creator"],
            similarity=float(data["similarity"]),
            fingerprint_kind=data["fingerprint_kind"],
            flagged_at=int(data["flagged_at"]),
            proposal_id=data.get("proposal_id"),
        )


class ArbitrationDecision(str, Enum):
    UPHELD_PARENT = "upheld_parent"
    """Council confirms derivative; auto-attribute now."""
    REJECTED_PARENT = "rejected_parent"
    """Council rejects derivative claim; no attribution."""
    INSUFFICIENT = "insufficient"
    """Not enough info; record kept open for re-review."""


@dataclass(frozen=True)
class _ResolutionRecord:
    """Internal: tracks how an arbitration was resolved.

    Held alongside the original ``DisputedAttributionRecord`` so a
    re-resolve with the SAME decision is idempotent (governance-webhook
    double-delivery defense), but a re-resolve with a CONFLICTING
    decision raises (catches data-integrity bugs in the council
    proposal flow)."""

    decision: ArbitrationDecision
    by_council: tuple


# ──────────────────────────────────────────────────────────────────────
# Protocol
# ──────────────────────────────────────────────────────────────────────


@runtime_checkable
class ArbitrationQueue(Protocol):
    async def enqueue(self, record: DisputedAttributionRecord) -> str: ...
    async def get(
        self, record_id: str,
    ) -> Optional[DisputedAttributionRecord]: ...
    async def list_pending(self) -> List[DisputedAttributionRecord]: ...
    async def resolve(
        self,
        record_id: str,
        *,
        decision: ArbitrationDecision,
        by_council: List[str],
    ) -> None: ...
    async def set_proposal_id(
        self, record_id: str, proposal_id: str,
    ) -> None: ...


# ──────────────────────────────────────────────────────────────────────
# In-memory implementation
# ──────────────────────────────────────────────────────────────────────


class InMemoryArbitrationQueue:
    """Unit-test fixture. NOT for production — survives only as long
    as the process. Use ``FilesystemArbitrationQueue`` on real nodes."""

    def __init__(self) -> None:
        self._records: Dict[str, DisputedAttributionRecord] = {}
        self._resolutions: Dict[str, _ResolutionRecord] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, record: DisputedAttributionRecord) -> str:
        record_id = uuid.uuid4().hex
        async with self._lock:
            self._records[record_id] = record
        return record_id

    async def get(
        self, record_id: str,
    ) -> Optional[DisputedAttributionRecord]:
        async with self._lock:
            return self._records.get(record_id)

    async def list_pending(self) -> List[DisputedAttributionRecord]:
        async with self._lock:
            pending = [
                rec for rid, rec in self._records.items()
                if rid not in self._resolutions
            ]
        # Older flag wins — councils review oldest first.
        return sorted(pending, key=lambda r: r.flagged_at)

    async def resolve(
        self,
        record_id: str,
        *,
        decision: ArbitrationDecision,
        by_council: List[str],
    ) -> None:
        async with self._lock:
            if record_id not in self._records:
                raise KeyError(f"unknown arbitration record: {record_id}")
            existing = self._resolutions.get(record_id)
            if existing is not None:
                if existing.decision != decision:
                    raise ValueError(
                        f"conflicting resolve for {record_id}: was "
                        f"{existing.decision.value}, now {decision.value}"
                    )
                # Same decision — idempotent no-op.
                return
            self._resolutions[record_id] = _ResolutionRecord(
                decision=decision,
                by_council=tuple(by_council),
            )

    async def set_proposal_id(
        self, record_id: str, proposal_id: str,
    ) -> None:
        async with self._lock:
            if record_id not in self._records:
                raise KeyError(f"unknown arbitration record: {record_id}")
            self._records[record_id] = dataclasses.replace(
                self._records[record_id], proposal_id=proposal_id,
            )


# ──────────────────────────────────────────────────────────────────────
# Filesystem implementation
# ──────────────────────────────────────────────────────────────────────


class FilesystemArbitrationQueue:
    """Node-local persistence: one JSON file per record.

    Layout::

        <queue_dir>/<record_id>.json

    Each file is a single JSON object with fields ``record`` (the
    serialised ``DisputedAttributionRecord``) and ``resolution``
    (``None`` or ``{"decision": ..., "by_council": [...]}``).

    Reconstruction is lazy: ``__init__`` scans the dir once and loads
    all valid files; corrupt files are logged + skipped (never raises
    out of construction).
    """

    def __init__(self, queue_dir: Path) -> None:
        self._queue_dir = Path(queue_dir)
        self._queue_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, DisputedAttributionRecord] = {}
        self._resolutions: Dict[str, _ResolutionRecord] = {}
        self._lock = asyncio.Lock()
        self._reload()

    def _reload(self) -> None:
        for fp in self._queue_dir.glob("*.json"):
            try:
                payload = json.loads(fp.read_text(encoding="utf-8"))
                record = DisputedAttributionRecord.from_dict(
                    payload["record"],
                )
                self._records[fp.stem] = record
                resolution = payload.get("resolution")
                if resolution is not None:
                    self._resolutions[fp.stem] = _ResolutionRecord(
                        decision=ArbitrationDecision(resolution["decision"]),
                        by_council=tuple(resolution["by_council"]),
                    )
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                OSError,
            ) as exc:
                logger.warning(
                    "arbitration queue: skipping unreadable record at "
                    "%s: %s",
                    fp,
                    exc,
                )

    def _write(self, record_id: str) -> None:
        record = self._records[record_id]
        resolution = self._resolutions.get(record_id)
        payload = {
            "record": record.to_dict(),
            "resolution": (
                None if resolution is None else {
                    "decision": resolution.decision.value,
                    "by_council": list(resolution.by_council),
                }
            ),
        }
        path = self._queue_dir / f"{record_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    async def enqueue(self, record: DisputedAttributionRecord) -> str:
        record_id = uuid.uuid4().hex
        async with self._lock:
            self._records[record_id] = record
            self._write(record_id)
        return record_id

    async def get(
        self, record_id: str,
    ) -> Optional[DisputedAttributionRecord]:
        async with self._lock:
            return self._records.get(record_id)

    async def list_pending(self) -> List[DisputedAttributionRecord]:
        async with self._lock:
            pending = [
                rec for rid, rec in self._records.items()
                if rid not in self._resolutions
            ]
        return sorted(pending, key=lambda r: r.flagged_at)

    async def resolve(
        self,
        record_id: str,
        *,
        decision: ArbitrationDecision,
        by_council: List[str],
    ) -> None:
        async with self._lock:
            if record_id not in self._records:
                raise KeyError(f"unknown arbitration record: {record_id}")
            existing = self._resolutions.get(record_id)
            if existing is not None:
                if existing.decision != decision:
                    raise ValueError(
                        f"conflicting resolve for {record_id}: was "
                        f"{existing.decision.value}, now {decision.value}"
                    )
                return
            self._resolutions[record_id] = _ResolutionRecord(
                decision=decision,
                by_council=tuple(by_council),
            )
            self._write(record_id)

    async def set_proposal_id(
        self, record_id: str, proposal_id: str,
    ) -> None:
        async with self._lock:
            if record_id not in self._records:
                raise KeyError(f"unknown arbitration record: {record_id}")
            self._records[record_id] = dataclasses.replace(
                self._records[record_id], proposal_id=proposal_id,
            )
            self._write(record_id)
