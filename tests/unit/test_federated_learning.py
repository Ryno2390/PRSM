"""Sprint 308 — federated-learning orchestrator.

Vision §7 Enterprise Confidentiality Mode capstone: a
training campaign coordinated across a fleet of TEE-attested
PRSM worker nodes. Workers train locally on recipient-
encrypted shards (sprint 304/307) inside their TEE (sprint
305/305a) under $CORP authorization (sprint 306/306a), and
return ONLY gradient updates — never plaintext data.

The orchestrator (this module) pools the gradients,
aggregates them via FedAvg or FedMedian, and produces the
round's combined update. The enterprise applies the update
to their own model copy locally; PRSM never holds the
trained model.

This sprint ships the orchestration layer + aggregation
math + filesystem persistence + endpoints + MCP. Live
worker-side `/compute/train` and encrypted-gradient
transport are deferred to 308a.
"""
from __future__ import annotations

import base64
import json
import struct

import pytest

from prsm.enterprise.federated_learning import (
    AggregationStrategy,
    FederatedJob,
    FederatedLearningOrchestrator,
    FederatedRound,
    GradientUpdate,
    JobStatus,
    RoundStatus,
    aggregate_fedavg,
    aggregate_fedmedian,
    decode_gradient,
    encode_gradient,
)


# ── Gradient encoding ───────────────────────────────


def test_encode_decode_round_trip():
    grad = [0.1, -0.2, 3.14, -42.0, 0.0]
    out = decode_gradient(encode_gradient(grad))
    assert len(out) == len(grad)
    for a, b in zip(grad, out):
        assert abs(a - b) < 1e-6


def test_encode_returns_bytes():
    out = encode_gradient([1.0, 2.0])
    assert isinstance(out, bytes)
    assert len(out) == 8  # 2 floats × 4 bytes


def test_decode_truncated_raises():
    with pytest.raises(ValueError):
        decode_gradient(b"\x00\x00\x00")


def test_decode_empty_is_empty():
    assert decode_gradient(b"") == []


# ── Aggregation: FedAvg ─────────────────────────────


def _update(node, grad, samples=100, job_id="j1",
            round_index=0):
    return GradientUpdate(
        job_id=job_id,
        round_index=round_index,
        worker_node_id=node,
        gradient_b64=base64.b64encode(
            encode_gradient(grad),
        ).decode(),
        sample_count=samples,
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=100.0,
    )


def test_fedavg_unweighted_mean_when_samples_equal():
    a = _update("n1", [1.0, 2.0, 3.0], samples=100)
    b = _update("n2", [3.0, 4.0, 5.0], samples=100)
    out = decode_gradient(aggregate_fedavg([a, b]))
    # Equal weights → mean of pairs: (2.0, 3.0, 4.0)
    assert out == pytest.approx([2.0, 3.0, 4.0], abs=1e-6)


def test_fedavg_weighted_by_sample_count():
    # Worker A has 3× the data → its gradient weighs 3× more
    a = _update("n1", [10.0, 10.0], samples=300)
    b = _update("n2", [0.0, 0.0], samples=100)
    out = decode_gradient(aggregate_fedavg([a, b]))
    # (300*10 + 100*0) / 400 = 7.5
    assert out == pytest.approx([7.5, 7.5], abs=1e-6)


def test_fedavg_single_update_returns_same():
    u = _update("n1", [0.5, -0.25, 17.0], samples=42)
    out = decode_gradient(aggregate_fedavg([u]))
    assert out == pytest.approx(
        [0.5, -0.25, 17.0], abs=1e-6,
    )


def test_fedavg_empty_raises():
    with pytest.raises(ValueError, match="empty|at least"):
        aggregate_fedavg([])


def test_fedavg_zero_samples_total_raises():
    """All workers reporting sample_count=0 → division-by-
    zero waiting to happen. Refuse loud."""
    a = _update("n1", [1.0], samples=0)
    b = _update("n2", [2.0], samples=0)
    with pytest.raises(ValueError, match="sample"):
        aggregate_fedavg([a, b])


def test_fedavg_negative_samples_rejected():
    a = _update("n1", [1.0], samples=-5)
    with pytest.raises(ValueError, match="negative|sample"):
        aggregate_fedavg([a])


def test_fedavg_length_mismatch_rejected():
    a = _update("n1", [1.0, 2.0], samples=10)
    b = _update("n2", [1.0, 2.0, 3.0], samples=10)
    with pytest.raises(ValueError, match="length"):
        aggregate_fedavg([a, b])


# ── Aggregation: FedMedian ──────────────────────────


def test_fedmedian_elementwise():
    a = _update("n1", [1.0, 10.0, 100.0], samples=10)
    b = _update("n2", [2.0, 20.0, 200.0], samples=10)
    c = _update("n3", [3.0, 30.0, 300.0], samples=10)
    out = decode_gradient(
        aggregate_fedmedian([a, b, c]),
    )
    assert out == pytest.approx(
        [2.0, 20.0, 200.0], abs=1e-6,
    )


def test_fedmedian_byzantine_resistance():
    """A Byzantine worker reporting extreme values doesn't
    swing the median (unlike FedAvg). This is the central
    value-prop of FedMedian over FedAvg."""
    honest_a = _update("n1", [1.0], samples=10)
    honest_b = _update("n2", [1.0], samples=10)
    byzantine = _update("n3", [1_000_000.0], samples=10)
    out = decode_gradient(
        aggregate_fedmedian(
            [honest_a, honest_b, byzantine],
        ),
    )
    assert out == pytest.approx([1.0], abs=1e-6)


def test_fedmedian_empty_raises():
    with pytest.raises(ValueError, match="empty|at least"):
        aggregate_fedmedian([])


def test_fedmedian_length_mismatch_rejected():
    a = _update("n1", [1.0, 2.0], samples=10)
    b = _update("n2", [1.0], samples=10)
    with pytest.raises(ValueError, match="length"):
        aggregate_fedmedian([a, b])


# ── FederatedJob ─────────────────────────────────────


def test_propose_job_assigns_id_and_initial_status():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="mock-llama-3-8b",
        dataset_cids=["QmA", "QmB"],
        worker_pool=["n1", "n2", "n3", "n4"],
        rounds_target=5,
        min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
    )
    assert job.job_id
    assert job.status == JobStatus.PROPOSED
    assert job.current_round == 0


def test_propose_job_rejects_min_workers_over_pool():
    orch = FederatedLearningOrchestrator()
    with pytest.raises(ValueError, match="min_workers"):
        orch.propose_job(
            model_id="x", dataset_cids=["QmA"],
            worker_pool=["n1", "n2"],
            rounds_target=3,
            min_workers_per_round=5,
            aggregation=AggregationStrategy.FEDAVG,
        )


def test_propose_job_rejects_zero_rounds():
    orch = FederatedLearningOrchestrator()
    with pytest.raises(ValueError, match="rounds_target"):
        orch.propose_job(
            model_id="x", dataset_cids=["QmA"],
            worker_pool=["n1"],
            rounds_target=0,
            min_workers_per_round=1,
            aggregation=AggregationStrategy.FEDAVG,
        )


def test_propose_job_rejects_empty_pool():
    orch = FederatedLearningOrchestrator()
    with pytest.raises(ValueError, match="worker_pool"):
        orch.propose_job(
            model_id="x", dataset_cids=["QmA"],
            worker_pool=[],
            rounds_target=3,
            min_workers_per_round=1,
            aggregation=AggregationStrategy.FEDAVG,
        )


# ── Round issuance ──────────────────────────────────


def test_issue_round_creates_assignments():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA", "QmB"],
        worker_pool=["n1", "n2", "n3"],
        rounds_target=3, min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
    )
    r = orch.issue_round(job.job_id)
    assert r.round_index == 0
    assert r.status == RoundStatus.ISSUED
    assert len(r.worker_assignments) >= 2
    # Job transitioned out of PROPOSED
    refreshed = orch.get_job(job.job_id)
    assert refreshed.status == JobStatus.RUNNING


def test_issue_round_increments_index():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"],
        rounds_target=3, min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    r0 = orch.issue_round(job.job_id)
    # Round 0 must complete (or fail) before round 1 issues
    orch.accept_gradient_update(_update(
        "n1", [1.0], job_id=job.job_id, round_index=0,
    ))
    orch.aggregate_round(job.job_id, 0)
    r1 = orch.issue_round(job.job_id)
    assert r1.round_index == 1


def test_issue_round_unknown_job_raises():
    orch = FederatedLearningOrchestrator()
    with pytest.raises(ValueError, match="not found"):
        orch.issue_round("nope")


def test_issue_round_refuses_after_completion():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    orch.accept_gradient_update(_update(
        "n1", [1.0], job_id=job.job_id, round_index=0,
    ))
    orch.aggregate_round(job.job_id, 0)
    # Job done (rounds_target=1) — no more rounds
    with pytest.raises(ValueError, match="complete|done"):
        orch.issue_round(job.job_id)


# ── Gradient updates ────────────────────────────────


def test_accept_update_records_it():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1", "n2"],
        rounds_target=2, min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    u = _update(
        "n1", [1.0, 2.0], job_id=job.job_id, round_index=0,
    )
    orch.accept_gradient_update(u)
    r = orch.get_round(job.job_id, 0)
    assert len(r.gradient_updates_received) == 1
    assert r.gradient_updates_received[0].worker_node_id == "n1"


def test_accept_update_unknown_job_rejected():
    orch = FederatedLearningOrchestrator()
    u = _update("n1", [1.0])
    with pytest.raises(ValueError, match="not found"):
        orch.accept_gradient_update(u)


def test_accept_update_wrong_round_rejected():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    # Submitting for round=5 when current round is 0
    u = _update(
        "n1", [1.0], job_id=job.job_id, round_index=5,
    )
    with pytest.raises(ValueError, match="round"):
        orch.accept_gradient_update(u)


def test_accept_update_worker_not_in_assignments_rejected():
    """Workers not assigned this round can't slip an update
    in. Guards against attack-from-the-pool."""
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1", "n2"],
        rounds_target=2, min_workers_per_round=1,
        # min=1 → only one worker assigned; the other is
        # unassigned for this round
        aggregation=AggregationStrategy.FEDAVG,
    )
    r = orch.issue_round(job.job_id)
    assigned = {a.node_id for a in r.worker_assignments}
    pool = {"n1", "n2"}
    unassigned = (pool - assigned).pop()
    u = _update(
        unassigned, [1.0],
        job_id=job.job_id, round_index=0,
    )
    with pytest.raises(ValueError, match="not assigned"):
        orch.accept_gradient_update(u)


def test_accept_update_duplicate_from_same_worker_rejected():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    u = _update(
        "n1", [1.0], job_id=job.job_id, round_index=0,
    )
    orch.accept_gradient_update(u)
    with pytest.raises(ValueError, match="duplicate|already"):
        orch.accept_gradient_update(u)


# ── Aggregation lifecycle ───────────────────────────


def test_aggregate_round_produces_combined_update():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1", "n2"],
        rounds_target=2, min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    orch.accept_gradient_update(_update(
        "n1", [1.0, 2.0], samples=10,
        job_id=job.job_id, round_index=0,
    ))
    orch.accept_gradient_update(_update(
        "n2", [3.0, 4.0], samples=10,
        job_id=job.job_id, round_index=0,
    ))
    out = orch.aggregate_round(job.job_id, 0)
    aggregated = decode_gradient(out.aggregated_update)
    assert aggregated == pytest.approx(
        [2.0, 3.0], abs=1e-6,
    )
    assert out.status == RoundStatus.AGGREGATED


def test_aggregate_round_below_quorum_refuses():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1", "n2", "n3"],
        rounds_target=2, min_workers_per_round=3,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    # Only 1 of required 3 update arrived
    orch.accept_gradient_update(_update(
        "n1", [1.0], job_id=job.job_id, round_index=0,
    ))
    with pytest.raises(ValueError, match="quorum|min_workers"):
        orch.aggregate_round(job.job_id, 0)


def test_aggregate_round_unknown_round_rejected():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    with pytest.raises(ValueError, match="round"):
        orch.aggregate_round(job.job_id, 5)


def test_aggregate_round_idempotent_refuses_re_aggregation():
    """Once aggregated, re-aggregation must be refused
    (otherwise downstream consumers can't trust the value)."""
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    orch.accept_gradient_update(_update(
        "n1", [1.0], job_id=job.job_id, round_index=0,
    ))
    orch.aggregate_round(job.job_id, 0)
    with pytest.raises(ValueError, match="already|status"):
        orch.aggregate_round(job.job_id, 0)


def test_job_completes_after_rounds_target_aggregations():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    for r in range(2):
        orch.issue_round(job.job_id)
        orch.accept_gradient_update(_update(
            "n1", [float(r)],
            job_id=job.job_id, round_index=r,
        ))
        orch.aggregate_round(job.job_id, r)
    refreshed = orch.get_job(job.job_id)
    assert refreshed.status == JobStatus.COMPLETED
    assert refreshed.current_round == 2


# ── Filters / queries ───────────────────────────────


def test_list_jobs_filter_by_status():
    orch = FederatedLearningOrchestrator()
    j1 = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    j2 = orch.propose_job(
        model_id="y", dataset_cids=["QmB"],
        worker_pool=["n2"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(j2.job_id)
    proposed = orch.list_jobs(status=JobStatus.PROPOSED)
    running = orch.list_jobs(status=JobStatus.RUNNING)
    assert [j.job_id for j in proposed] == [j1.job_id]
    assert [j.job_id for j in running] == [j2.job_id]


# ── Persistence ─────────────────────────────────────


def test_persist_round_trip(tmp_path):
    orch = FederatedLearningOrchestrator(
        persist_dir=tmp_path,
    )
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    orch.accept_gradient_update(_update(
        "n1", [1.0, 2.0],
        job_id=job.job_id, round_index=0,
    ))

    # Reload
    orch2 = FederatedLearningOrchestrator(
        persist_dir=tmp_path,
    )
    j2 = orch2.get_job(job.job_id)
    assert j2 is not None
    assert j2.status == JobStatus.RUNNING
    r = orch2.get_round(job.job_id, 0)
    assert len(r.gradient_updates_received) == 1


def test_persist_corrupt_file_failsoft(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    orch = FederatedLearningOrchestrator(
        persist_dir=tmp_path,
    )
    assert orch.list_jobs() == []


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PRSM_FEDERATED_LEARNING_DIR", str(tmp_path),
    )
    orch = FederatedLearningOrchestrator.from_env()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    # Persisted to disk
    files = list(tmp_path.glob("*.json"))
    assert files


# ── Serialization ───────────────────────────────────


def test_job_to_dict_round_trip():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1", "n2"],
        rounds_target=3, min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDMEDIAN,
    )
    restored = FederatedJob.from_dict(job.to_dict())
    assert restored == job


def test_round_to_dict_round_trip():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    r = orch.get_round(job.job_id, 0)
    restored = FederatedRound.from_dict(r.to_dict())
    assert restored == r


def test_update_to_dict_round_trip():
    u = _update("n1", [1.0, 2.0, 3.0])
    restored = GradientUpdate.from_dict(u.to_dict())
    assert restored == u
