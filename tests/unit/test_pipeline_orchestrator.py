"""Sprint 312 — pipeline inference orchestrator.

Coordinates multi-stage inference: routes the prompt
through stage 0's runner, captures its output, feeds it
into stage 1, and so on. Records per-stage attestations
+ activation hashes, then signs the resulting
PipelineInferenceReceipt.

v1 runs all stages in the orchestrator's process (the
StageRunner Protocol is sync). Cross-node activation
streaming is sprint 313.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from prsm.compute.inference.attestation_backends import (
    AttestationVerificationResult,
)
from prsm.compute.inference.pipeline_orchestrator import (
    PipelineInferenceJob,
    PipelineInferenceOrchestrator,
    PipelineJobStatus,
    PipelineRoundStatus,
)
from prsm.compute.inference.pipeline_partition import (
    PipelinePartition, even_layer_partition,
)
from prsm.compute.inference.pipeline_receipt import (
    verify_pipeline_receipt,
)
from prsm.compute.inference.pipeline_stage import (
    deterministic_stub_stage_runner,
)
from prsm.enterprise.federated_learning import (
    generate_worker_keypair,
)


def _orch_with_key():
    priv, pub = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
    )
    return orch, priv, pub


def _basic_partition():
    return even_layer_partition(
        total_layers=8, node_ids=["n0", "n1"],
    )


# ── Job lifecycle ───────────────────────────────────


def test_propose_job_assigns_id_and_status():
    orch, _, _ = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    assert job.job_id
    assert job.status == PipelineJobStatus.PROPOSED


def test_propose_job_rejects_invalid_partition():
    orch, _, _ = _orch_with_key()
    bad = PipelinePartition(
        total_layers=4,
        stage_layer_ranges=[[0, 1]],  # gap
        stage_node_ids=["n0"],
    )
    with pytest.raises(ValueError):
        orch.propose_job(model_id="m1", partition=bad)


def test_orchestrator_requires_privkey():
    with pytest.raises(ValueError, match="privkey"):
        PipelineInferenceOrchestrator(
            orchestrator_privkey_b64="",
        )


# ── Execute happy path ─────────────────────────────


def test_execute_produces_signed_receipt():
    orch, priv, pub = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    rnd = orch.execute(
        job.job_id,
        prompt=b"hello world",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    assert rnd.status == PipelineRoundStatus.COMPLETED
    assert rnd.receipt is not None
    # Receipt verifies end-to-end
    result = verify_pipeline_receipt(
        rnd.receipt, orchestrator_pubkey_b64=pub,
    )
    assert result.ok, result.diagnostic


def test_execute_chain_is_correct():
    """Stage K's output equals stage K+1's input — the
    orchestrator MUST thread activations through stages
    in order, not in parallel."""
    orch, _, _ = _orch_with_key()
    partition = even_layer_partition(
        total_layers=6, node_ids=["a", "b", "c"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    rnd = orch.execute(
        job.job_id, prompt=b"test",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    # Hash chain holds — receipt verification confirms
    # this end-to-end.
    receipt = rnd.receipt
    assert len(receipt.stage_receipts) == 3
    for k in range(2):
        assert (
            receipt.stage_receipts[k].output_activation_hash
            == receipt.stage_receipts[k + 1].input_activation_hash
        )


def test_execute_records_partition_hash():
    orch, _, _ = _orch_with_key()
    partition = _basic_partition()
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    rnd = orch.execute(
        job.job_id, prompt=b"x",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    assert (
        rnd.receipt.partition_hash
        == partition.partition_hash()
    )


def test_execute_records_prompt_and_output_hashes():
    orch, _, _ = _orch_with_key()
    prompt = b"prompt bytes"
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    rnd = orch.execute(
        job.job_id, prompt=prompt,
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    expected_prompt_hash = hashlib.sha256(prompt).hexdigest()
    assert rnd.receipt.prompt_hash == expected_prompt_hash
    # output_hash matches the final stage's output (not
    # checking the bytes directly since they're stub-
    # generated, just the chain consistency)
    assert (
        rnd.receipt.output_hash
        == rnd.receipt.stage_receipts[-1].output_activation_hash
    )


def test_execute_deterministic_with_stub_runners():
    """Same prompt + same partition + stub runners → same
    receipt (modulo timestamp + non-load-bearing fields)."""
    orch, _, _ = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    r1 = orch.execute(
        job.job_id, prompt=b"same prompt",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    # Reset job for second run — propose a new one
    job2 = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    r2 = orch.execute(
        job2.job_id, prompt=b"same prompt",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    # Output hash same → pipeline computation is
    # deterministic
    assert r1.receipt.output_hash == r2.receipt.output_hash


# ── Failure modes ───────────────────────────────────


def test_execute_with_stage_runner_failure_marks_round_failed():
    orch, _, _ = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )

    def failing_runner(*, input_activations,
                       stage_id, layer_indices):
        raise RuntimeError(
            f"stage {stage_id} simulated failure",
        )

    with pytest.raises(RuntimeError, match="simulated"):
        orch.execute(
            job.job_id, prompt=b"x",
            stage_runners=[
                deterministic_stub_stage_runner(),
                failing_runner,
            ],
        )
    rnd = orch.get_round(job.job_id)
    assert rnd.status == PipelineRoundStatus.FAILED


def test_execute_wrong_runner_count_raises():
    """n_stages == 2 but only 1 runner supplied — refuse
    loud."""
    orch, _, _ = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    with pytest.raises(ValueError, match="runner"):
        orch.execute(
            job.job_id, prompt=b"x",
            stage_runners=[
                deterministic_stub_stage_runner(),
            ],
        )


def test_execute_unknown_job_raises():
    orch, _, _ = _orch_with_key()
    with pytest.raises(ValueError, match="not found"):
        orch.execute(
            "nope", prompt=b"x",
            stage_runners=[
                deterministic_stub_stage_runner(),
            ],
        )


def test_execute_after_completion_raises():
    """Pipeline jobs are one-shot — re-executing a
    completed job is operator confusion. Refuse."""
    orch, _, _ = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    orch.execute(
        job.job_id, prompt=b"x",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    with pytest.raises(ValueError, match="completed"):
        orch.execute(
            job.job_id, prompt=b"x",
            stage_runners=[
                deterministic_stub_stage_runner(),
                deterministic_stub_stage_runner(),
            ],
        )


# ── Per-stage attestation captured ──────────────────


def test_execute_records_per_stage_attestations():
    orch, _, _ = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    # Supply per-stage attestation blobs (sprint 305 shape)
    # via the optional stage_attestations parameter
    fake_att = AttestationVerificationResult(
        vendor="intel-sgx",
        vendor_verified=False,
        structural_parse_ok=True,
    )
    rnd = orch.execute(
        job.job_id, prompt=b"x",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
        stage_attestations=[fake_att, fake_att],
    )
    for stage_receipt in rnd.receipt.stage_receipts:
        assert stage_receipt.attestation.vendor == (
            "intel-sgx"
        )


# ── Filesystem persistence ──────────────────────────


def test_persist_round_trip(tmp_path: Path):
    priv, _ = generate_worker_keypair()
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
        persist_dir=tmp_path,
    )
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    orch.execute(
        job.job_id, prompt=b"x",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )

    # Reload
    orch2 = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
        persist_dir=tmp_path,
    )
    j2 = orch2.get_job(job.job_id)
    assert j2 is not None
    assert j2.status == PipelineJobStatus.COMPLETED
    rnd = orch2.get_round(job.job_id)
    assert rnd.status == PipelineRoundStatus.COMPLETED
    assert rnd.receipt is not None


def test_persist_corrupt_file_failsoft(tmp_path: Path):
    priv, _ = generate_worker_keypair()
    (tmp_path / "broken.json").write_text("{not json")
    orch = PipelineInferenceOrchestrator(
        orchestrator_privkey_b64=priv,
        persist_dir=tmp_path,
    )
    assert orch.list_jobs() == []


# ── Receipt verifies against orchestrator pubkey ────


def test_receipt_verifies_with_orchestrator_pubkey():
    orch, _, pub = _orch_with_key()
    job = orch.propose_job(
        model_id="m1", partition=_basic_partition(),
    )
    rnd = orch.execute(
        job.job_id, prompt=b"end-to-end",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    result = verify_pipeline_receipt(
        rnd.receipt, orchestrator_pubkey_b64=pub,
    )
    assert result.ok
    assert result.signature_valid
    assert result.chain_valid
