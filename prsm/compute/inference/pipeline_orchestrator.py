"""Sprint 312 — pipeline inference orchestrator.

Coordinates multi-stage TEE-attested inference across a
partitioned model. Routes the prompt through stage 0's
runner, captures activations + attestation, feeds the
output into stage 1, and so on. Records the full chain in
a `PipelineInferenceReceipt` signed by the orchestrator's
Ed25519 key.

v1 runs all stages in the orchestrator's process (the
StageRunner Protocol is sync). Cross-node activation
streaming = sprint 313. Real PyTorch per-stage forward
pass = sprint 314.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
)
from prsm.compute.inference.pipeline_partition import (
    PipelinePartition,
)
from prsm.compute.inference.pipeline_receipt import (
    PerStageReceipt,
    PipelineInferenceReceipt,
    sign_pipeline_receipt,
)
from prsm.compute.inference.pipeline_stage import (
    StageRunner,
)
from prsm.enterprise.federated_learning import (
    _load_ed25519_priv,
)

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────


class PipelineJobStatus(str, Enum):
    PROPOSED = "proposed"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRoundStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Dataclasses ─────────────────────────────────────


@dataclass
class PipelineInferenceJob:
    job_id: str
    model_id: str
    partition: PipelinePartition
    status: PipelineJobStatus
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "partition": self.partition.to_dict(),
            "status": self.status.value,
            "created_at": float(self.created_at),
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "PipelineInferenceJob":
        return cls(
            job_id=d["job_id"],
            model_id=d["model_id"],
            partition=PipelinePartition.from_dict(
                d["partition"],
            ),
            status=PipelineJobStatus(d["status"]),
            created_at=float(d.get("created_at", 0.0)),
        )


@dataclass
class PipelineInferenceRound:
    job_id: str
    round_id: str
    status: PipelineRoundStatus
    receipt: Optional[PipelineInferenceReceipt] = None
    started_at: float = 0.0
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "round_id": self.round_id,
            "status": self.status.value,
            "receipt": (
                self.receipt.to_dict()
                if self.receipt is not None else None
            ),
            "started_at": float(self.started_at),
            "completed_at": (
                float(self.completed_at)
                if self.completed_at is not None else None
            ),
            "error": self.error,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any],
    ) -> "PipelineInferenceRound":
        rcpt_raw = d.get("receipt")
        return cls(
            job_id=d["job_id"],
            round_id=d["round_id"],
            status=PipelineRoundStatus(d["status"]),
            receipt=(
                PipelineInferenceReceipt.from_dict(rcpt_raw)
                if rcpt_raw is not None else None
            ),
            started_at=float(d.get("started_at", 0.0)),
            completed_at=(
                float(d["completed_at"])
                if d.get("completed_at") is not None
                else None
            ),
            error=d.get("error"),
        )


# ── Orchestrator ────────────────────────────────────


class PipelineInferenceOrchestrator:
    def __init__(
        self,
        *,
        orchestrator_privkey_b64: str,
        persist_dir: Optional[Path] = None,
    ) -> None:
        if not orchestrator_privkey_b64:
            raise ValueError(
                "orchestrator_privkey_b64 is required "
                "(used to sign pipeline receipts)"
            )
        # Validate eagerly so misconfig is loud
        try:
            _load_ed25519_priv(orchestrator_privkey_b64)
        except ValueError as e:
            raise ValueError(
                f"orchestrator_privkey_b64 malformed: {e}"
            )
        self._privkey = orchestrator_privkey_b64
        self._jobs: Dict[str, PipelineInferenceJob] = {}
        # One round per job for v1 (pipeline inference is
        # one-shot; multi-round = sprint 315 / retry semantics)
        self._rounds: Dict[
            str, PipelineInferenceRound,
        ] = {}
        self._persist_dir: Optional[Path] = (
            Path(persist_dir)
            if persist_dir is not None else None
        )
        if self._persist_dir is not None:
            self._persist_dir.mkdir(
                parents=True, exist_ok=True,
            )
            self._load_from_disk()

    # ── Job lifecycle ────────────────────────────────

    def propose_job(
        self,
        *,
        model_id: str,
        partition: PipelinePartition,
    ) -> PipelineInferenceJob:
        partition.validate()
        job = PipelineInferenceJob(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            partition=partition,
            status=PipelineJobStatus.PROPOSED,
            created_at=time.time(),
        )
        self._jobs[job.job_id] = job
        self._persist_job(job)
        return job

    def get_job(
        self, job_id: str,
    ) -> Optional[PipelineInferenceJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[PipelineInferenceJob]:
        return list(self._jobs.values())

    def get_round(
        self, job_id: str,
    ) -> Optional[PipelineInferenceRound]:
        return self._rounds.get(job_id)

    # ── Execute ──────────────────────────────────────

    def execute(
        self,
        job_id: str,
        *,
        prompt: bytes,
        stage_runners: List[StageRunner],
        stage_attestations: Optional[
            List[AttestationVerificationResult]
        ] = None,
    ) -> PipelineInferenceRound:
        """Run the full pipeline. Routes the prompt through
        each stage's runner in order, records per-stage
        attestation + activation hashes, signs the
        receipt, marks the job COMPLETED.

        Raises ValueError when the job is unknown, the
        runner count doesn't match the partition, or the
        job is already completed. If a stage runner
        raises, the round is marked FAILED + the exception
        propagates."""
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(
                f"job {job_id!r} not found"
            )
        if job.status == PipelineJobStatus.COMPLETED:
            raise ValueError(
                f"job {job_id!r} already completed; "
                f"pipeline jobs are one-shot"
            )
        n_stages = job.partition.n_stages
        if len(stage_runners) != n_stages:
            raise ValueError(
                f"job has {n_stages} stages but "
                f"{len(stage_runners)} runners supplied"
            )
        if stage_attestations is not None and len(
            stage_attestations,
        ) != n_stages:
            raise ValueError(
                f"job has {n_stages} stages but "
                f"{len(stage_attestations)} attestations "
                f"supplied"
            )

        rnd = PipelineInferenceRound(
            job_id=job_id,
            round_id=str(uuid.uuid4()),
            status=PipelineRoundStatus.RUNNING,
            started_at=time.time(),
        )
        self._rounds[job_id] = rnd
        self._persist_round(rnd)

        prompt_hash = hashlib.sha256(prompt).hexdigest()
        current_activations = prompt
        stage_receipts: List[PerStageReceipt] = []
        try:
            for stage_idx in range(n_stages):
                input_hash = hashlib.sha256(
                    current_activations,
                ).hexdigest()
                output_bytes = stage_runners[stage_idx](
                    input_activations=current_activations,
                    stage_id=stage_idx,
                    layer_indices=(
                        job.partition.stage_layer_ranges[
                            stage_idx
                        ]
                    ),
                )
                output_hash = hashlib.sha256(
                    output_bytes,
                ).hexdigest()
                att = (
                    stage_attestations[stage_idx]
                    if stage_attestations is not None
                    else AttestationVerificationResult(
                        vendor="software-fallback",
                        structural_parse_ok=True,
                    )
                )
                stage_receipts.append(PerStageReceipt(
                    stage_id=stage_idx,
                    layer_indices=list(
                        job.partition.stage_layer_ranges[
                            stage_idx
                        ],
                    ),
                    input_activation_hash=input_hash,
                    output_activation_hash=output_hash,
                    attestation=att,
                ))
                current_activations = output_bytes
        except Exception as exc:
            rnd.status = PipelineRoundStatus.FAILED
            rnd.error = (
                f"{type(exc).__name__}: {exc}"
            )
            rnd.completed_at = time.time()
            self._persist_round(rnd)
            raise

        output_hash = hashlib.sha256(
            current_activations,
        ).hexdigest()
        receipt = PipelineInferenceReceipt(
            prompt_hash=prompt_hash,
            output_hash=output_hash,
            partition_hash=job.partition.partition_hash(),
            stage_receipts=stage_receipts,
            orchestrator_signature_b64="",
        )
        sign_pipeline_receipt(
            receipt, orchestrator_privkey_b64=self._privkey,
        )

        rnd.receipt = receipt
        rnd.status = PipelineRoundStatus.COMPLETED
        rnd.completed_at = time.time()
        self._persist_round(rnd)

        job.status = PipelineJobStatus.COMPLETED
        self._persist_job(job)
        return rnd

    # ── Persistence ──────────────────────────────────

    def _persist_job(
        self, job: PipelineInferenceJob,
    ) -> None:
        if self._persist_dir is None:
            return
        safe = (
            job.job_id.replace("/", "_")
            .replace("\\", "_").replace("..", "_")
        )
        path = self._persist_dir / f"job-{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(job.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PipelineInferenceOrchestrator: job "
                "persist failed for %s: %s",
                job.job_id, exc,
            )

    def _persist_round(
        self, rnd: PipelineInferenceRound,
    ) -> None:
        if self._persist_dir is None:
            return
        safe = (
            rnd.job_id.replace("/", "_")
            .replace("\\", "_").replace("..", "_")
        )
        path = self._persist_dir / f"round-{safe}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(rnd.to_dict()))
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PipelineInferenceOrchestrator: round "
                "persist failed for %s: %s",
                rnd.job_id, exc,
            )

    def _load_from_disk(self) -> None:
        assert self._persist_dir is not None
        for path in self._persist_dir.glob("job-*.json"):
            try:
                job = PipelineInferenceJob.from_dict(
                    json.loads(path.read_text()),
                )
                self._jobs[job.job_id] = job
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PipelineInferenceOrchestrator: "
                    "skipping corrupt job %s: %s",
                    path, exc,
                )
        for path in self._persist_dir.glob("round-*.json"):
            try:
                rnd = PipelineInferenceRound.from_dict(
                    json.loads(path.read_text()),
                )
                self._rounds[rnd.job_id] = rnd
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PipelineInferenceOrchestrator: "
                    "skipping corrupt round %s: %s",
                    path, exc,
                )
