"""Sprint 308a — FL orchestrator hardening.

Two incremental gates layered on top of sprint 308:

1. Per-update Ed25519 worker signatures (opt-in via
   FederatedJob.require_signed_updates). Without it, any
   PRSM node can submit a fake gradient claiming to be
   from a worker. With it, only the holder of worker X's
   private key can submit for worker X — closes the
   MITM/spoof attack surface.

2. Central differential-privacy noise on aggregation
   (opt-in via FederatedJob.dp_policy). The orchestrator
   clips each element of the aggregated gradient and adds
   Gaussian noise with σ derived from (ε, δ, clip_norm).
   This is CENTRAL DP — guarantees apply regardless of
   worker behavior, no need to trust individual workers
   on local DP.

Both gates default OFF, preserving sprint 308's
backwards-compat surface.
"""
from __future__ import annotations

import base64
import math
import statistics

import pytest

from prsm.enterprise.federated_learning import (
    AggregationStrategy,
    DPPolicy,
    FederatedLearningOrchestrator,
    GradientUpdate,
    WorkerKey,
    aggregate_fedavg,
    decode_gradient,
    encode_gradient,
    generate_worker_keypair,
    sign_gradient_update,
    verify_gradient_update_signature,
)


# ── Worker keypair ──────────────────────────────────


def test_generate_worker_keypair_b64():
    priv, pub = generate_worker_keypair()
    assert len(base64.b64decode(priv)) == 32
    assert len(base64.b64decode(pub)) == 32


def test_generate_worker_keypair_unique():
    a = generate_worker_keypair()
    b = generate_worker_keypair()
    assert a != b


# ── Sign / verify gradient update ───────────────────


def _unsigned(node="n1", job_id="j1", round_index=0,
              gradient=None, samples=10, timestamp=100.0):
    if gradient is None:
        gradient = [1.0, 2.0, 3.0]
    return GradientUpdate(
        job_id=job_id,
        round_index=round_index,
        worker_node_id=node,
        gradient_b64=base64.b64encode(
            encode_gradient(gradient),
        ).decode(),
        sample_count=samples,
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=timestamp,
    )


def test_sign_then_verify_passes():
    priv, pub = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(), worker_privkey_b64=priv,
    )
    assert verify_gradient_update_signature(u, pub)


def test_signature_tamper_detected():
    priv, pub = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(), worker_privkey_b64=priv,
    )
    u.sample_count = 999_999  # invalidates signature
    assert not verify_gradient_update_signature(u, pub)


def test_signature_gradient_tamper_detected():
    priv, pub = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(), worker_privkey_b64=priv,
    )
    u.gradient_b64 = base64.b64encode(
        encode_gradient([0.0, 0.0, 0.0]),
    ).decode()
    assert not verify_gradient_update_signature(u, pub)


def test_signature_wrong_pubkey_rejected():
    priv, _ = generate_worker_keypair()
    _, other_pub = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(), worker_privkey_b64=priv,
    )
    assert not verify_gradient_update_signature(
        u, other_pub,
    )


def test_signature_node_id_in_canonical_payload():
    """Signature binds worker_node_id — relabeling the
    update to claim a different node must fail
    verification (otherwise a worker could spoof updates
    'from' another worker)."""
    priv, pub = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(node="n1"), worker_privkey_b64=priv,
    )
    u.worker_node_id = "n2"
    assert not verify_gradient_update_signature(u, pub)


# ── Worker key registry on orchestrator ─────────────


def test_register_and_get_worker_key():
    orch = FederatedLearningOrchestrator()
    _, pub = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub))
    got = orch.get_worker_key("n1")
    assert got is not None
    assert got.signing_pubkey_b64 == pub


def test_register_worker_key_invalid_pubkey_rejected():
    orch = FederatedLearningOrchestrator()
    with pytest.raises(ValueError):
        orch.register_worker_key(WorkerKey(
            "n1", "not-base64!",
        ))


def test_register_worker_key_rotation():
    """Re-registering the same node_id rotates the pubkey
    (legitimate key-rotation path)."""
    orch = FederatedLearningOrchestrator()
    _, pub1 = generate_worker_keypair()
    _, pub2 = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub1))
    orch.register_worker_key(WorkerKey("n1", pub2))
    assert (
        orch.get_worker_key("n1").signing_pubkey_b64
        == pub2
    )


def test_list_worker_keys():
    orch = FederatedLearningOrchestrator()
    _, pa = generate_worker_keypair()
    _, pb = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pa))
    orch.register_worker_key(WorkerKey("n2", pb))
    ids = {k.node_id for k in orch.list_worker_keys()}
    assert ids == {"n1", "n2"}


# ── require_signed_updates enforcement ──────────────


def _setup_job(*, require_signed=False, dp_policy=None):
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=require_signed,
        dp_policy=dp_policy,
    )
    orch.issue_round(job.job_id)
    return orch, job


def test_unsigned_update_accepted_when_not_required():
    """Backwards-compat: default behavior unchanged."""
    orch, job = _setup_job(require_signed=False)
    orch.accept_gradient_update(_unsigned(
        node="n1", job_id=job.job_id,
    ))
    r = orch.get_round(job.job_id, 0)
    assert len(r.gradient_updates_received) == 1


def test_unsigned_update_refused_when_required():
    orch, job = _setup_job(require_signed=True)
    _, pub = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub))
    with pytest.raises(ValueError, match="sign|signature"):
        orch.accept_gradient_update(_unsigned(
            node="n1", job_id=job.job_id,
        ))


def test_signed_update_accepted_when_required():
    orch, job = _setup_job(require_signed=True)
    priv, pub = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub))
    u = sign_gradient_update(
        _unsigned(node="n1", job_id=job.job_id),
        worker_privkey_b64=priv,
    )
    orch.accept_gradient_update(u)
    r = orch.get_round(job.job_id, 0)
    assert len(r.gradient_updates_received) == 1


def test_signed_update_with_wrong_key_refused():
    orch, job = _setup_job(require_signed=True)
    _, registered_pub = generate_worker_keypair()
    orch.register_worker_key(
        WorkerKey("n1", registered_pub),
    )
    attacker_priv, _ = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(node="n1", job_id=job.job_id),
        worker_privkey_b64=attacker_priv,
    )
    with pytest.raises(ValueError, match="signature"):
        orch.accept_gradient_update(u)


def test_signed_update_unregistered_worker_refused():
    """Worker isn't in the key registry — even with a
    valid signature, refuse (we can't verify against
    anything)."""
    orch, job = _setup_job(require_signed=True)
    priv, _ = generate_worker_keypair()
    u = sign_gradient_update(
        _unsigned(node="n1", job_id=job.job_id),
        worker_privkey_b64=priv,
    )
    with pytest.raises(ValueError, match="registered|key"):
        orch.accept_gradient_update(u)


# ── DPPolicy validation ─────────────────────────────


def test_dp_policy_validation():
    DPPolicy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        DPPolicy(epsilon=0.0, delta=1e-5, clip_norm=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        DPPolicy(epsilon=-1.0, delta=1e-5, clip_norm=1.0)
    with pytest.raises(ValueError, match="delta"):
        DPPolicy(epsilon=1.0, delta=0.0, clip_norm=1.0)
    with pytest.raises(ValueError, match="delta"):
        DPPolicy(epsilon=1.0, delta=1.0, clip_norm=1.0)
    with pytest.raises(ValueError, match="clip_norm"):
        DPPolicy(epsilon=1.0, delta=1e-5, clip_norm=0.0)


def test_dp_policy_round_trip():
    p = DPPolicy(epsilon=2.5, delta=1e-6, clip_norm=3.0)
    assert DPPolicy.from_dict(p.to_dict()) == p


# ── Central DP noise on aggregation ─────────────────


def test_dp_clips_gradient_elements():
    """With a clip_norm well below the gradient values
    and ε very large (→ negligible noise), the aggregated
    output must be element-wise clipped to ±clip_norm."""
    policy = DPPolicy(
        epsilon=1e9,  # near-zero noise
        delta=1e-9,
        clip_norm=1.0,
    )
    orch, job = _setup_job(dp_policy=policy)
    # Worker reports gradient with elements far above clip
    orch.accept_gradient_update(_unsigned(
        node="n1", job_id=job.job_id,
        gradient=[100.0, -100.0, 0.5],
    ))
    rnd = orch.aggregate_round(job.job_id, 0)
    out = decode_gradient(rnd.aggregated_update)
    # 100 → clipped to 1.0; -100 → -1.0; 0.5 → unchanged
    # Noise σ ≈ 0 at ε=1e9
    assert abs(out[0] - 1.0) < 0.01
    assert abs(out[1] - (-1.0)) < 0.01
    assert abs(out[2] - 0.5) < 0.01


def test_dp_no_policy_means_no_clipping():
    """Without dp_policy, aggregation must NOT clip — the
    sprint 308 default behavior."""
    orch, job = _setup_job(dp_policy=None)
    orch.accept_gradient_update(_unsigned(
        node="n1", job_id=job.job_id,
        gradient=[100.0, -100.0, 0.5],
    ))
    rnd = orch.aggregate_round(job.job_id, 0)
    out = decode_gradient(rnd.aggregated_update)
    assert out == pytest.approx(
        [100.0, -100.0, 0.5], abs=1e-6,
    )


def test_dp_records_sigma_in_round():
    """The round must record the σ actually used so the
    enterprise has an audit trail of the noise scale."""
    policy = DPPolicy(
        epsilon=1.0, delta=1e-5, clip_norm=1.0,
    )
    orch, job = _setup_job(dp_policy=policy)
    orch.accept_gradient_update(_unsigned(
        node="n1", job_id=job.job_id,
        gradient=[0.5, -0.5],
    ))
    rnd = orch.aggregate_round(job.job_id, 0)
    # σ = clip_norm * sqrt(2 * ln(1.25/δ)) / ε
    # For ε=1, δ=1e-5, clip_norm=1:
    expected_sigma = (
        1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5))
        / 1.0
    )
    assert abs(
        rnd.dp_noise_sigma_applied - expected_sigma,
    ) < 1e-6


def test_dp_noise_actually_perturbs_gradient():
    """At ε=0.1 (modest privacy), σ ~ 50 — many runs
    should produce visibly different aggregated outputs
    even with identical inputs."""
    policy = DPPolicy(
        epsilon=0.1, delta=1e-5, clip_norm=1.0,
    )
    outs = []
    for _ in range(5):
        orch, job = _setup_job(dp_policy=policy)
        orch.accept_gradient_update(_unsigned(
            node="n1", job_id=job.job_id,
            gradient=[0.0],
        ))
        rnd = orch.aggregate_round(job.job_id, 0)
        outs.append(decode_gradient(rnd.aggregated_update)[0])
    # All 5 distinct (probability of collision ~ 0)
    assert len(set(outs)) == 5


# ── FederatedJob serialization carries new fields ──


def test_job_to_dict_includes_require_signed_and_dp_policy():
    orch = FederatedLearningOrchestrator()
    policy = DPPolicy(
        epsilon=1.0, delta=1e-5, clip_norm=2.0,
    )
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
        dp_policy=policy,
    )
    d = job.to_dict()
    assert d["require_signed_updates"] is True
    assert d["dp_policy"]["epsilon"] == 1.0
    assert d["dp_policy"]["clip_norm"] == 2.0


def test_job_backwards_compat_default_fields():
    """Sprint 308 jobs (no require_signed_updates / no
    dp_policy field) must continue to round-trip."""
    from prsm.enterprise.federated_learning import (
        FederatedJob, JobStatus,
    )
    # Construct a sprint-308-shape dict — both new fields
    # absent from the wire format
    sprint308_dict = {
        "job_id": "j-1",
        "model_id": "x",
        "dataset_cids": ["QmA"],
        "worker_pool": ["n1"],
        "rounds_target": 1,
        "min_workers_per_round": 1,
        "aggregation": "fedavg",
        "status": "proposed",
        "current_round": 0,
        "started_at": 0.0,
        "completed_at": None,
    }
    restored = FederatedJob.from_dict(sprint308_dict)
    assert restored.require_signed_updates is False
    assert restored.dp_policy is None


# ── Persistence carries new fields ──────────────────


def test_persist_round_trip_with_hardening(tmp_path):
    orch = FederatedLearningOrchestrator(
        persist_dir=tmp_path,
    )
    priv, pub = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub))
    policy = DPPolicy(
        epsilon=1.0, delta=1e-5, clip_norm=1.0,
    )
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
        dp_policy=policy,
    )
    orch.issue_round(job.job_id)
    u = sign_gradient_update(
        _unsigned(node="n1", job_id=job.job_id),
        worker_privkey_b64=priv,
    )
    orch.accept_gradient_update(u)

    # Reload
    orch2 = FederatedLearningOrchestrator(
        persist_dir=tmp_path,
    )
    j2 = orch2.get_job(job.job_id)
    assert j2.require_signed_updates is True
    assert j2.dp_policy == policy
    # Worker key registry persisted
    assert (
        orch2.get_worker_key("n1").signing_pubkey_b64
        == pub
    )
