"""Sprint 318b — end-to-end demo runnable.

A single script that exercises the §7 enterprise FL flow
+ federated inference flow against an in-process
deployment. Operators run this to verify a fresh deploy
actually works, see what each component contributes
end-to-end, and get a reproducible reference for their
own integration tests.

The demo uses orchestrator primitives directly (no HTTP
/ Docker), so it's testable from pytest. Sprint 318a's
Docker variant will run THIS script inside the container
as a post-bringup health check.
"""
from __future__ import annotations

import base64
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DemoEnvironment:
    """In-process deployment state for the demo."""

    persist_root: Path

    # Orchestrator keypairs (Ed25519 b64)
    fl_orchestrator_priv: str = ""
    fl_orchestrator_pub: str = ""
    pipeline_orchestrator_priv: str = ""
    pipeline_orchestrator_pub: str = ""

    # Worker keypairs: list of (priv, pub) Ed25519 b64
    worker_keypairs: List[Tuple[str, str]] = field(
        default_factory=list,
    )

    # Wired orchestrator instances (concrete types are
    # late-bound to keep import cost off this module)
    fl_orchestrator: object = None
    pipeline_orchestrator: object = None


@dataclass
class DemoStepResult:
    ok: bool
    message: str
    aggregated_bytes: bytes = b""
    pipeline_receipt: object = None


def _step(label: str) -> None:
    sys.stdout.write(f"  → {label}\n")
    sys.stdout.flush()


def setup_demo_environment(
    *,
    persist_root: Path,
    n_workers: int = 2,
) -> DemoEnvironment:
    """Build a fresh in-process deployment. Generates
    keypairs, instantiates orchestrators under
    persist_root."""
    from prsm.compute.inference.pipeline_orchestrator import (
        PipelineInferenceOrchestrator,
    )
    from prsm.enterprise.federated_learning import (
        FederatedLearningOrchestrator,
        generate_worker_keypair,
    )

    persist_root = Path(persist_root)
    fl_dir = persist_root / "fl"
    pipeline_dir = persist_root / "pipeline"

    fl_priv, fl_pub = generate_worker_keypair()
    pipeline_priv, pipeline_pub = generate_worker_keypair()

    workers = [
        generate_worker_keypair() for _ in range(n_workers)
    ]

    fl_orch = FederatedLearningOrchestrator(
        persist_dir=fl_dir,
    )
    pipeline_orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=pipeline_priv,
        persist_dir=pipeline_dir,
    )

    return DemoEnvironment(
        persist_root=persist_root,
        fl_orchestrator_priv=fl_priv,
        fl_orchestrator_pub=fl_pub,
        pipeline_orchestrator_priv=pipeline_priv,
        pipeline_orchestrator_pub=pipeline_pub,
        worker_keypairs=workers,
        fl_orchestrator=fl_orch,
        pipeline_orchestrator=pipeline_orch,
    )


# ── FL demo ─────────────────────────────────────────


def run_fl_demo(env: DemoEnvironment) -> DemoStepResult:
    """Execute the full §7 federated learning flow:
      1. Register worker keys with the orchestrator
      2. Propose a signed-updates-required job
      3. Each worker generates + signs a gradient update
      4. Orchestrator aggregates via FedAvg
      5. Return the aggregated update bytes
    """
    sys.stdout.write(
        "Running §7 federated-learning demo:\n",
    )
    from prsm.enterprise.federated_learning import (
        AggregationStrategy,
        GradientUpdate,
        WorkerKey,
        encode_gradient,
        sign_gradient_update,
    )

    orch = env.fl_orchestrator

    _step("Registering worker keys with FL orchestrator")
    for i, (_priv, pub) in enumerate(env.worker_keypairs):
        orch.register_worker_key(WorkerKey(
            node_id=f"demo-worker-{i}",
            signing_pubkey_b64=pub,
        ))

    _step(
        "Proposing FL job (FedAvg, "
        "require_signed_updates=True)",
    )
    job = orch.propose_job(
        model_id="demo-tiny-mlp",
        dataset_cids=[
            f"Qm-demo-{i}"
            for i in range(len(env.worker_keypairs))
        ],
        worker_pool=[
            f"demo-worker-{i}"
            for i in range(len(env.worker_keypairs))
        ],
        rounds_target=1,
        min_workers_per_round=len(env.worker_keypairs),
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
    )
    sys.stdout.write(
        f"    job_id={job.job_id}\n",
    )

    _step("Issuing round 0")
    orch.issue_round(job.job_id)

    _step("Each worker signs a gradient update")
    for i, (priv, _pub) in enumerate(env.worker_keypairs):
        gradient = [
            (i + 1) * 0.1, (i + 1) * 0.2,
            (i + 1) * 0.3, (i + 1) * 0.4,
        ]
        update = GradientUpdate(
            job_id=job.job_id,
            round_index=0,
            worker_node_id=f"demo-worker-{i}",
            gradient_b64=base64.b64encode(
                encode_gradient(gradient),
            ).decode(),
            sample_count=10,
            worker_attestation_b64="",
            worker_signature_b64="",
            timestamp=100.0 + i,
        )
        signed = sign_gradient_update(
            update, worker_privkey_b64=priv,
        )
        orch.accept_gradient_update(signed)

    _step("Orchestrator aggregates round")
    rnd = orch.aggregate_round(job.job_id, 0)
    sys.stdout.write(
        f"    aggregated_bytes={len(rnd.aggregated_update)} "
        f"bytes\n",
    )

    return DemoStepResult(
        ok=True,
        message=(
            f"FL demo complete: job {job.job_id} "
            f"aggregated {len(env.worker_keypairs)} "
            f"signed worker updates"
        ),
        aggregated_bytes=rnd.aggregated_update,
    )


# ── Pipeline inference demo ────────────────────────


def run_pipeline_demo(
    env: DemoEnvironment,
) -> DemoStepResult:
    """Execute a 2-stage federated inference run:
      1. Propose a job with a 4-layer / 2-stage partition
      2. Execute with the deterministic stub stage runner
      3. Verify the receipt signature + activation chain
    """
    sys.stdout.write(
        "\nRunning federated pipeline-inference demo:\n",
    )
    from prsm.compute.inference.pipeline_partition import (
        even_layer_partition,
    )
    from prsm.compute.inference.pipeline_receipt import (
        verify_pipeline_receipt,
    )
    from prsm.compute.inference.pipeline_stage import (
        deterministic_stub_stage_runner,
    )

    orch = env.pipeline_orchestrator
    partition = even_layer_partition(
        total_layers=4,
        node_ids=["demo-stage-0", "demo-stage-1"],
    )

    _step("Proposing pipeline inference job")
    job = orch.propose_job(
        model_id="demo-pipeline-model",
        partition=partition,
    )
    sys.stdout.write(
        f"    job_id={job.job_id} "
        f"(2 stages × 2 layers)\n",
    )

    _step(
        "Executing inference with stub stage runners "
        "(in-process)",
    )
    rnd = orch.execute(
        job.job_id,
        prompt=b"demo prompt for federated inference",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    sys.stdout.write(
        f"    output_hash="
        f"{rnd.receipt.output_hash[:24]}...\n",
    )

    _step(
        "Verifying receipt (signature + activation chain)",
    )
    verification = verify_pipeline_receipt(
        rnd.receipt,
        orchestrator_pubkey_b64=(
            env.pipeline_orchestrator_pub
        ),
    )
    if not verification.ok:
        return DemoStepResult(
            ok=False,
            message=(
                f"receipt verification failed: "
                f"{verification.diagnostic}"
            ),
            pipeline_receipt=rnd.receipt,
        )
    sys.stdout.write("    ✓ receipt verifies end-to-end\n")

    return DemoStepResult(
        ok=True,
        message=(
            f"Pipeline demo complete: job {job.job_id} "
            f"produced verifiable receipt over 2 stages"
        ),
        pipeline_receipt=rnd.receipt,
    )


# ── Full demo (runs both) ───────────────────────────


def run_full_demo(
    env: DemoEnvironment,
) -> List[Tuple[str, DemoStepResult]]:
    """Run both demos in sequence. Returns a list of
    (name, result) tuples. Prints a summary line at the
    end."""
    results: List[Tuple[str, DemoStepResult]] = []
    results.append(("fl", run_fl_demo(env)))
    results.append(("pipeline", run_pipeline_demo(env)))

    sys.stdout.write("\n")
    all_ok = all(r.ok for _, r in results)
    if all_ok:
        sys.stdout.write(
            "✓ All demos passed — PRSM enterprise stack "
            "verified end-to-end.\n",
        )
    else:
        sys.stdout.write(
            "✗ One or more demos failed:\n",
        )
        for name, result in results:
            if not result.ok:
                sys.stdout.write(
                    f"  - {name}: {result.message}\n",
                )
    return results
