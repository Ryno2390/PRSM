"""Sprint 308 — federated-learning orchestrator.

Vision §7 Enterprise Confidentiality Mode capstone: trains
a shared model across a fleet of TEE-attested PRSM worker
nodes that never see plaintext data. The enterprise:

  1. Recipient-encrypts the training data to the worker
     fleet (sprint 304 OR-decrypt or sprint 307 t-of-n
     threshold)
  2. Proposes a FederatedJob declaring the model,
     dataset CIDs, worker pool, rounds target, min-quorum,
     aggregation strategy
  3. The orchestrator issues each round to a subset of
     the worker pool with assignments
  4. Workers train locally INSIDE their TEE (gated by
     sprint 305a TEE policy + sprint 306a $CORP
     capability) on the decrypted shards and return ONLY
     gradient updates
  5. Orchestrator aggregates gradients (FedAvg or
     FedMedian) and surfaces the combined update
  6. Enterprise applies the update to their own model
     copy locally and proposes the next round

PRSM never holds the trained model — only the aggregated
gradient updates flow through. The plaintext data never
leaves a TEE.

This sprint ships the orchestration layer + aggregation
math + lifecycle + filesystem persistence. Worker-side
/compute/train integration is deferred to 308a.

Aggregation:
  FEDAVG     — weighted average by sample_count
  FEDMEDIAN  — element-wise median (Byzantine-robust;
               single outlier cannot swing the result)

Gradient encoding: little-endian packed float32 array,
base64-encoded for JSON transport. Pure Python; no numpy
dependency.
"""
from __future__ import annotations

import json
import logging
import os
import random
import statistics
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────


class AggregationStrategy(str, Enum):
    FEDAVG = "fedavg"
    FEDMEDIAN = "fedmedian"


class JobStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RoundStatus(str, Enum):
    ISSUED = "issued"
    COLLECTING = "collecting"
    AGGREGATED = "aggregated"
    FAILED = "failed"


_TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED, JobStatus.FAILED,
    JobStatus.CANCELLED,
}


# ── Gradient encoding ───────────────────────────────


def encode_gradient(values: List[float]) -> bytes:
    """Pack a list of floats into little-endian float32
    bytes. Stable wire format for cross-language
    consumers."""
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def decode_gradient(blob: bytes) -> List[float]:
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise ValueError(
            f"gradient blob length {len(blob)} not "
            f"divisible by 4 (float32 size)"
        )
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ── Dataclasses ──────────────────────────────────────


@dataclass
class WorkerAssignment:
    node_id: str
    dataset_cid: str
    assigned_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dataset_cid": self.dataset_cid,
            "assigned_at": float(self.assigned_at),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "WorkerAssignment":
        return cls(
            node_id=d["node_id"],
            dataset_cid=d["dataset_cid"],
            assigned_at=float(d["assigned_at"]),
        )


@dataclass
class GradientUpdate:
    job_id: str
    round_index: int
    worker_node_id: str
    gradient_b64: str
    sample_count: int
    worker_attestation_b64: str
    worker_signature_b64: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "round_index": int(self.round_index),
            "worker_node_id": self.worker_node_id,
            "gradient_b64": self.gradient_b64,
            "sample_count": int(self.sample_count),
            "worker_attestation_b64": (
                self.worker_attestation_b64
            ),
            "worker_signature_b64": (
                self.worker_signature_b64
            ),
            "timestamp": float(self.timestamp),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "GradientUpdate":
        return cls(
            job_id=d["job_id"],
            round_index=int(d["round_index"]),
            worker_node_id=d["worker_node_id"],
            gradient_b64=d["gradient_b64"],
            sample_count=int(d["sample_count"]),
            worker_attestation_b64=d.get(
                "worker_attestation_b64", "",
            ),
            worker_signature_b64=d.get(
                "worker_signature_b64", "",
            ),
            timestamp=float(d["timestamp"]),
        )


@dataclass
class FederatedRound:
    job_id: str
    round_index: int
    status: RoundStatus
    worker_assignments: List[WorkerAssignment] = field(
        default_factory=list,
    )
    gradient_updates_received: List[GradientUpdate] = field(
        default_factory=list,
    )
    aggregated_update: bytes = b""
    issued_at: float = 0.0
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        import base64 as _b64
        return {
            "job_id": self.job_id,
            "round_index": int(self.round_index),
            "status": self.status.value,
            "worker_assignments": [
                a.to_dict() for a in self.worker_assignments
            ],
            "gradient_updates_received": [
                u.to_dict()
                for u in self.gradient_updates_received
            ],
            "aggregated_update_b64": _b64.b64encode(
                self.aggregated_update,
            ).decode("ascii"),
            "issued_at": float(self.issued_at),
            "completed_at": (
                float(self.completed_at)
                if self.completed_at is not None else None
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "FederatedRound":
        import base64 as _b64
        return cls(
            job_id=d["job_id"],
            round_index=int(d["round_index"]),
            status=RoundStatus(d["status"]),
            worker_assignments=[
                WorkerAssignment.from_dict(a)
                for a in (
                    d.get("worker_assignments") or []
                )
            ],
            gradient_updates_received=[
                GradientUpdate.from_dict(u)
                for u in (
                    d.get("gradient_updates_received")
                    or []
                )
            ],
            aggregated_update=_b64.b64decode(
                d.get("aggregated_update_b64") or "",
            ),
            issued_at=float(d.get("issued_at", 0.0)),
            completed_at=(
                float(d["completed_at"])
                if d.get("completed_at") is not None
                else None
            ),
        )


@dataclass
class FederatedJob:
    job_id: str
    model_id: str
    dataset_cids: List[str]
    worker_pool: List[str]
    rounds_target: int
    min_workers_per_round: int
    aggregation: AggregationStrategy
    status: JobStatus
    current_round: int = 0
    started_at: float = 0.0
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "dataset_cids": list(self.dataset_cids),
            "worker_pool": list(self.worker_pool),
            "rounds_target": int(self.rounds_target),
            "min_workers_per_round": int(
                self.min_workers_per_round,
            ),
            "aggregation": self.aggregation.value,
            "status": self.status.value,
            "current_round": int(self.current_round),
            "started_at": float(self.started_at),
            "completed_at": (
                float(self.completed_at)
                if self.completed_at is not None else None
            ),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "FederatedJob":
        return cls(
            job_id=d["job_id"],
            model_id=d["model_id"],
            dataset_cids=list(d.get("dataset_cids") or []),
            worker_pool=list(d.get("worker_pool") or []),
            rounds_target=int(d["rounds_target"]),
            min_workers_per_round=int(
                d["min_workers_per_round"],
            ),
            aggregation=AggregationStrategy(
                d["aggregation"],
            ),
            status=JobStatus(d["status"]),
            current_round=int(d.get("current_round", 0)),
            started_at=float(d.get("started_at", 0.0)),
            completed_at=(
                float(d["completed_at"])
                if d.get("completed_at") is not None
                else None
            ),
        )


# ── Aggregation ──────────────────────────────────────


def _validate_updates_consistent(
    updates: List[GradientUpdate],
) -> int:
    """Returns the common gradient length. Raises on
    empty list or length mismatch."""
    if not updates:
        raise ValueError(
            "aggregate requires at least one update"
        )
    import base64 as _b64
    first_len = None
    for u in updates:
        raw = _b64.b64decode(u.gradient_b64)
        if first_len is None:
            first_len = len(raw)
        elif len(raw) != first_len:
            raise ValueError(
                f"gradient length mismatch: expected "
                f"{first_len} bytes, got {len(raw)} on "
                f"worker {u.worker_node_id!r}"
            )
    assert first_len is not None
    return first_len


def aggregate_fedavg(
    updates: List[GradientUpdate],
) -> bytes:
    """Weighted-average aggregation: each worker's gradient
    is weighted by their `sample_count`. Returns packed
    float32 bytes."""
    import base64 as _b64

    _validate_updates_consistent(updates)
    for u in updates:
        if u.sample_count < 0:
            raise ValueError(
                f"negative sample_count on worker "
                f"{u.worker_node_id!r}: {u.sample_count}"
            )
    total_samples = sum(u.sample_count for u in updates)
    if total_samples <= 0:
        raise ValueError(
            "aggregate_fedavg requires total sample_count "
            "> 0 across all updates"
        )

    grads = [
        decode_gradient(_b64.b64decode(u.gradient_b64))
        for u in updates
    ]
    n = len(grads[0])
    out = [0.0] * n
    for u, g in zip(updates, grads):
        w = u.sample_count / total_samples
        for i in range(n):
            out[i] += w * g[i]
    return encode_gradient(out)


def aggregate_fedmedian(
    updates: List[GradientUpdate],
) -> bytes:
    """Element-wise median aggregation. Byzantine-robust:
    a single outlier worker cannot swing the result."""
    import base64 as _b64

    _validate_updates_consistent(updates)
    grads = [
        decode_gradient(_b64.b64decode(u.gradient_b64))
        for u in updates
    ]
    n = len(grads[0])
    out = [
        statistics.median(
            [g[i] for g in grads],
        )
        for i in range(n)
    ]
    return encode_gradient(out)


_AGGREGATORS = {
    AggregationStrategy.FEDAVG: aggregate_fedavg,
    AggregationStrategy.FEDMEDIAN: aggregate_fedmedian,
}


# ── Orchestrator ─────────────────────────────────────


class FederatedLearningOrchestrator:
    """In-memory + filesystem-persisted federated-learning
    coordinator. Jobs and rounds are stored as JSON files
    under persist_dir."""

    def __init__(
        self,
        *,
        persist_dir: Optional[Path] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._jobs: Dict[str, FederatedJob] = {}
        # rounds keyed by (job_id, round_index)
        self._rounds: Dict[
            tuple[str, int], FederatedRound,
        ] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir)
            if persist_dir is not None else None
        )
        self._rng = rng if rng is not None else random.Random()
        if self._persist_dir is not None:
            self._persist_dir.mkdir(
                parents=True, exist_ok=True,
            )
            self._load_from_disk()

    @classmethod
    def from_env(
        cls,
    ) -> "FederatedLearningOrchestrator":
        raw = os.environ.get("PRSM_FEDERATED_LEARNING_DIR")
        persist_dir = Path(raw) if raw else None
        return cls(persist_dir=persist_dir)

    # ── Job lifecycle ────────────────────────────────

    def propose_job(
        self,
        *,
        model_id: str,
        dataset_cids: List[str],
        worker_pool: List[str],
        rounds_target: int,
        min_workers_per_round: int,
        aggregation: AggregationStrategy,
    ) -> FederatedJob:
        if not worker_pool:
            raise ValueError(
                "worker_pool must be non-empty"
            )
        if rounds_target < 1:
            raise ValueError(
                f"rounds_target must be >= 1; got "
                f"{rounds_target}"
            )
        if min_workers_per_round < 1:
            raise ValueError(
                f"min_workers_per_round must be >= 1; got "
                f"{min_workers_per_round}"
            )
        if min_workers_per_round > len(worker_pool):
            raise ValueError(
                f"min_workers_per_round "
                f"({min_workers_per_round}) exceeds pool "
                f"size ({len(worker_pool)})"
            )
        job = FederatedJob(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            dataset_cids=list(dataset_cids),
            worker_pool=list(worker_pool),
            rounds_target=int(rounds_target),
            min_workers_per_round=int(
                min_workers_per_round,
            ),
            aggregation=aggregation,
            status=JobStatus.PROPOSED,
            started_at=time.time(),
        )
        self._jobs[job.job_id] = job
        self._persist_job(job)
        return job

    def get_job(
        self, job_id: str,
    ) -> Optional[FederatedJob]:
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        *,
        status: Optional[JobStatus] = None,
    ) -> List[FederatedJob]:
        out = list(self._jobs.values())
        out.sort(
            key=lambda j: j.started_at, reverse=True,
        )
        if status is not None:
            out = [j for j in out if j.status == status]
        return out

    # ── Round lifecycle ──────────────────────────────

    def issue_round(self, job_id: str) -> FederatedRound:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(
                f"job {job_id!r} not found"
            )
        if job.status in _TERMINAL_JOB_STATUSES:
            raise ValueError(
                f"job {job_id!r} is complete or terminal "
                f"(status={job.status.value!r}); cannot "
                f"issue more rounds"
            )
        if job.current_round >= job.rounds_target:
            raise ValueError(
                f"job {job_id!r} has issued all "
                f"{job.rounds_target} rounds; cannot "
                f"issue more"
            )
        # Pick the round-of-workers: a deterministic
        # subset of size >= min_workers_per_round. For v1,
        # sample exactly min_workers_per_round (smallest
        # quorum). Operators can extend later.
        pool = list(job.worker_pool)
        # Use the RNG seeded per-orchestrator for testability
        self._rng.shuffle(pool)
        assigned = pool[:job.min_workers_per_round]
        now = time.time()
        # Round-robin a dataset_cid for each assigned worker
        # (cycles if dataset count < worker count)
        assignments = [
            WorkerAssignment(
                node_id=node_id,
                dataset_cid=job.dataset_cids[
                    i % max(1, len(job.dataset_cids))
                ] if job.dataset_cids else "",
                assigned_at=now,
            )
            for i, node_id in enumerate(assigned)
        ]
        rnd = FederatedRound(
            job_id=job_id,
            round_index=job.current_round,
            status=RoundStatus.ISSUED,
            worker_assignments=assignments,
            issued_at=now,
        )
        self._rounds[(job_id, job.current_round)] = rnd
        # Transition job out of PROPOSED on first round
        if job.status == JobStatus.PROPOSED:
            job.status = JobStatus.RUNNING
            self._persist_job(job)
        self._persist_round(rnd)
        return rnd

    def get_round(
        self, job_id: str, round_index: int,
    ) -> Optional[FederatedRound]:
        return self._rounds.get((job_id, round_index))

    def accept_gradient_update(
        self, update: GradientUpdate,
    ) -> None:
        job = self._jobs.get(update.job_id)
        if job is None:
            raise ValueError(
                f"job {update.job_id!r} not found"
            )
        rnd = self._rounds.get(
            (update.job_id, update.round_index),
        )
        if rnd is None:
            raise ValueError(
                f"round {update.round_index} not issued "
                f"for job {update.job_id!r}"
            )
        if rnd.status == RoundStatus.AGGREGATED:
            raise ValueError(
                f"round {update.round_index} is already "
                f"aggregated; cannot accept more updates"
            )
        # Worker must be in the assignment list
        assigned_nodes = {
            a.node_id for a in rnd.worker_assignments
        }
        if update.worker_node_id not in assigned_nodes:
            raise ValueError(
                f"worker {update.worker_node_id!r} not "
                f"assigned to round {update.round_index}"
            )
        # No duplicate per worker
        for u in rnd.gradient_updates_received:
            if u.worker_node_id == update.worker_node_id:
                raise ValueError(
                    f"worker {update.worker_node_id!r} "
                    f"already submitted an update for "
                    f"this round (duplicate)"
                )
        rnd.gradient_updates_received.append(update)
        if rnd.status == RoundStatus.ISSUED:
            rnd.status = RoundStatus.COLLECTING
        self._persist_round(rnd)

    def aggregate_round(
        self, job_id: str, round_index: int,
    ) -> FederatedRound:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(
                f"job {job_id!r} not found"
            )
        rnd = self._rounds.get((job_id, round_index))
        if rnd is None:
            raise ValueError(
                f"round {round_index} not issued for job "
                f"{job_id!r}"
            )
        if rnd.status == RoundStatus.AGGREGATED:
            raise ValueError(
                f"round {round_index} already aggregated "
                f"(status={rnd.status.value!r})"
            )
        if (
            len(rnd.gradient_updates_received)
            < job.min_workers_per_round
        ):
            raise ValueError(
                f"round {round_index} below quorum: got "
                f"{len(rnd.gradient_updates_received)} "
                f"updates, need "
                f"{job.min_workers_per_round} "
                f"(min_workers_per_round)"
            )
        aggregator = _AGGREGATORS[job.aggregation]
        try:
            rnd.aggregated_update = aggregator(
                rnd.gradient_updates_received,
            )
        except ValueError as e:
            rnd.status = RoundStatus.FAILED
            self._persist_round(rnd)
            raise ValueError(
                f"aggregation failed for round "
                f"{round_index}: {e}"
            )
        rnd.status = RoundStatus.AGGREGATED
        rnd.completed_at = time.time()
        self._persist_round(rnd)

        # Advance job state
        job.current_round += 1
        if job.current_round >= job.rounds_target:
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
        self._persist_job(job)
        return rnd

    # ── Persistence ──────────────────────────────────

    def _job_path(self, job_id: str) -> Path:
        assert self._persist_dir is not None
        safe = (
            job_id.replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        return self._persist_dir / f"job-{safe}.json"

    def _round_path(
        self, job_id: str, round_index: int,
    ) -> Path:
        assert self._persist_dir is not None
        safe = (
            job_id.replace("/", "_")
            .replace("\\", "_")
            .replace("..", "_")
        )
        return (
            self._persist_dir
            / f"round-{safe}-{int(round_index)}.json"
        )

    def _persist_job(self, job: FederatedJob) -> None:
        if self._persist_dir is None:
            return
        path = self._job_path(job.job_id)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(job.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FederatedLearningOrchestrator: job "
                "persist failed for %s: %s",
                job.job_id, exc,
            )

    def _persist_round(self, rnd: FederatedRound) -> None:
        if self._persist_dir is None:
            return
        path = self._round_path(rnd.job_id, rnd.round_index)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(rnd.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FederatedLearningOrchestrator: round "
                "persist failed for %s-%d: %s",
                rnd.job_id, rnd.round_index, exc,
            )

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        # Jobs first so rounds find their parent
        for path in self._persist_dir.glob("job-*.json"):
            try:
                job = FederatedJob.from_dict(
                    json.loads(path.read_text()),
                )
                self._jobs[job.job_id] = job
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FederatedLearningOrchestrator: "
                    "skipping corrupt job %s: %s",
                    path, exc,
                )
        for path in self._persist_dir.glob("round-*.json"):
            try:
                rnd = FederatedRound.from_dict(
                    json.loads(path.read_text()),
                )
                self._rounds[
                    (rnd.job_id, rnd.round_index)
                ] = rnd
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FederatedLearningOrchestrator: "
                    "skipping corrupt round %s: %s",
                    path, exc,
                )
