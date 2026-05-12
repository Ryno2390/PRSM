"""Sprint 311 — cross-organization federated learning.

Multiple enterprises pool model training without sharing
data. Architecture: hierarchical aggregation.

  - Each enterprise runs its OWN FederatedLearningOrchestrator
    with its own worker pool (sprint 308) and its own
    encrypted dataset shards (sprint 304)
  - A consortium orchestrator runs at a third location;
    its "worker pool" is the list of participating
    enterprise IDs, not individual nodes
  - Each enterprise: workers train locally inside their
    TEE → enterprise orchestrator aggregates → enterprise
    publishes its aggregated_update to the consortium as a
    signed GradientUpdate
  - Consortium aggregates the per-enterprise aggregates
    via FedAvg or FedMedian

The whole composition runs on the existing primitives:
  - Sprint 308 FederatedLearningOrchestrator (used at both
    levels — enterprise + consortium)
  - Sprint 308a worker signatures (used by consortium to
    verify enterprise submissions)
  - Sprint 308a DP policy (applied by consortium to the
    cross-org aggregate)
  - Sprint 308c encrypted-gradient transport (used between
    enterprise and consortium — the consortium aggregator
    only sees encrypted aggregates from individual
    enterprises until aggregation time)

New code this sprint: a single bridge helper that wraps a
local round's aggregated_update as a consortium-bound
signed GradientUpdate.
"""
from __future__ import annotations

import base64

import pytest

torch = pytest.importorskip("torch")

from prsm.enterprise.consortium_fl import (
    aggregated_round_to_gradient_update,
)
from prsm.enterprise.federated_learning import (
    AggregationStrategy,
    DPPolicy,
    FederatedLearningOrchestrator,
    WorkerKey,
    decode_gradient,
    encode_gradient,
    generate_transport_keypair,
    generate_worker_keypair,
    seal_gradient_for_orchestrator,
    sign_gradient_update,
    verify_gradient_update_signature,
)


# ── aggregated_round_to_gradient_update bridge ──────


def _worker_update(orch, job, node_id, priv, gradient,
                   samples=10, round_index=0):
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    from prsm.enterprise.federated_learning import (
        GradientUpdate,
    )
    import base64 as _b64
    u = GradientUpdate(
        job_id=job.job_id, round_index=round_index,
        worker_node_id=node_id,
        gradient_b64=_b64.b64encode(
            encode_gradient(gradient),
        ).decode(),
        sample_count=samples,
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=100.0,
    )
    return sign_gradient_update(
        u, worker_privkey_b64=priv,
    )


def _setup_enterprise_orchestrator(
    *, n_workers=2,
):
    """Build a fresh enterprise orchestrator with a small
    worker pool. Returns (orch, job, [(node_id, priv)])."""
    orch = FederatedLearningOrchestrator()
    workers = []
    for i in range(n_workers):
        priv, pub = generate_worker_keypair()
        node_id = f"w{i}"
        orch.register_worker_key(WorkerKey(node_id, pub))
        workers.append((node_id, priv))
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=[w[0] for w in workers],
        rounds_target=1,
        min_workers_per_round=n_workers,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
    )
    orch.issue_round(job.job_id)
    return orch, job, workers


def test_bridge_signs_consortium_update():
    """Enterprise A's aggregated round → signed consortium
    update. The signature verifies under the enterprise's
    own signing key."""
    enterprise_priv, enterprise_pub = (
        generate_worker_keypair()
    )
    orch, job, workers = _setup_enterprise_orchestrator(
        n_workers=2,
    )
    for (node_id, priv) in workers:
        orch.accept_gradient_update(_worker_update(
            orch, job, node_id, priv,
            [1.0, 2.0, 3.0],
        ))
    local_round = orch.aggregate_round(job.job_id, 0)

    consortium_update = aggregated_round_to_gradient_update(
        local_round=local_round,
        consortium_job_id="consort-j-1",
        consortium_round_index=0,
        enterprise_node_id="enterprise-A",
        enterprise_privkey_b64=enterprise_priv,
        sample_count=20,  # total samples across workers
    )
    assert consortium_update.job_id == "consort-j-1"
    assert consortium_update.round_index == 0
    assert consortium_update.worker_node_id == (
        "enterprise-A"
    )
    assert consortium_update.sample_count == 20
    assert verify_gradient_update_signature(
        consortium_update, enterprise_pub,
    )


def test_bridge_carries_aggregated_gradient():
    """The bridged update's gradient_b64 IS the enterprise's
    aggregated_update bytes."""
    enterprise_priv, _ = generate_worker_keypair()
    orch, job, workers = _setup_enterprise_orchestrator(
        n_workers=2,
    )
    orch.accept_gradient_update(_worker_update(
        orch, job, workers[0][0], workers[0][1],
        [10.0, 20.0],
    ))
    orch.accept_gradient_update(_worker_update(
        orch, job, workers[1][0], workers[1][1],
        [30.0, 40.0],
    ))
    local_round = orch.aggregate_round(job.job_id, 0)

    bridged = aggregated_round_to_gradient_update(
        local_round=local_round,
        consortium_job_id="c1",
        consortium_round_index=0,
        enterprise_node_id="enterprise-A",
        enterprise_privkey_b64=enterprise_priv,
        sample_count=20,
    )
    # Decode the bridged gradient — should equal the
    # enterprise's aggregated_update (FedAvg of the two
    # worker gradients with equal weights)
    bridged_grad = decode_gradient(
        base64.b64decode(bridged.gradient_b64),
    )
    assert bridged_grad == pytest.approx(
        [20.0, 30.0], abs=1e-6,
    )


def test_bridge_rejects_unaggregated_round():
    """A round that hasn't been aggregated yet has empty
    aggregated_update — bridging it would publish zero
    bytes, which is operator confusion. Refuse loud."""
    enterprise_priv, _ = generate_worker_keypair()
    orch, job, workers = _setup_enterprise_orchestrator(
        n_workers=2,
    )
    # Don't accept updates; don't aggregate
    from prsm.enterprise.federated_learning import (
        RoundStatus,
    )
    rnd = orch.get_round(job.job_id, 0)
    assert rnd.status == RoundStatus.ISSUED
    with pytest.raises(
        ValueError, match="AGGREGATED|aggregate",
    ):
        aggregated_round_to_gradient_update(
            local_round=rnd,
            consortium_job_id="c1",
            consortium_round_index=0,
            enterprise_node_id="enterprise-A",
            enterprise_privkey_b64=enterprise_priv,
            sample_count=20,
        )


def test_bridge_with_transport_seal():
    """When transport_pubkey_b64 is set, the bridged update
    is sealed to the consortium's transport pubkey. The
    consortium then unseals on aggregation."""
    enterprise_priv, _ = generate_worker_keypair()
    tx_priv, tx_pub = generate_transport_keypair()
    orch, job, workers = _setup_enterprise_orchestrator(
        n_workers=2,
    )
    for (node_id, priv) in workers:
        orch.accept_gradient_update(_worker_update(
            orch, job, node_id, priv, [1.0, 2.0, 3.0],
        ))
    local_round = orch.aggregate_round(job.job_id, 0)

    bridged = aggregated_round_to_gradient_update(
        local_round=local_round,
        consortium_job_id="c1",
        consortium_round_index=0,
        enterprise_node_id="enterprise-A",
        enterprise_privkey_b64=enterprise_priv,
        sample_count=20,
        transport_pubkey_b64=tx_pub,
    )
    assert bridged.gradient_envelope_b64 is not None
    # Consortium can unseal with the matching privkey
    from prsm.enterprise.federated_learning import (
        unseal_gradient_from_worker,
    )
    unsealed = unseal_gradient_from_worker(
        bridged.gradient_b64,
        bridged.gradient_envelope_b64,
        tx_priv,
    )
    assert decode_gradient(unsealed) == pytest.approx(
        [1.0, 2.0, 3.0], abs=1e-6,
    )


# ── End-to-end: 2 enterprises × 2 workers → consortium ──


def test_e2e_cross_org_two_enterprises_no_encryption():
    """Two enterprises each train 2 workers; each
    enterprise aggregates locally; consortium aggregates
    across enterprises. The output is the hierarchical
    cross-org aggregate."""
    # Enterprise A
    a_orch, a_job, a_workers = (
        _setup_enterprise_orchestrator(n_workers=2)
    )
    a_priv, a_pub = generate_worker_keypair()

    # Enterprise B
    b_orch, b_job, b_workers = (
        _setup_enterprise_orchestrator(n_workers=2)
    )
    b_priv, b_pub = generate_worker_keypair()

    # Consortium orchestrator
    c_orch = FederatedLearningOrchestrator()
    c_orch.register_worker_key(
        WorkerKey("enterprise-A", a_pub),
    )
    c_orch.register_worker_key(
        WorkerKey("enterprise-B", b_pub),
    )
    c_job = c_orch.propose_job(
        model_id="cross-org-x",
        dataset_cids=[],  # consortium doesn't hold data
        worker_pool=["enterprise-A", "enterprise-B"],
        rounds_target=1,
        min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
    )
    c_orch.issue_round(c_job.job_id)

    # Enterprise A's workers train
    for (node_id, priv) in a_workers:
        a_orch.accept_gradient_update(_worker_update(
            a_orch, a_job, node_id, priv, [1.0, 1.0, 1.0],
        ))
    a_round = a_orch.aggregate_round(a_job.job_id, 0)
    a_bridged = aggregated_round_to_gradient_update(
        local_round=a_round,
        consortium_job_id=c_job.job_id,
        consortium_round_index=0,
        enterprise_node_id="enterprise-A",
        enterprise_privkey_b64=a_priv,
        sample_count=20,
    )

    # Enterprise B's workers train
    for (node_id, priv) in b_workers:
        b_orch.accept_gradient_update(_worker_update(
            b_orch, b_job, node_id, priv,
            [3.0, 3.0, 3.0],
        ))
    b_round = b_orch.aggregate_round(b_job.job_id, 0)
    b_bridged = aggregated_round_to_gradient_update(
        local_round=b_round,
        consortium_job_id=c_job.job_id,
        consortium_round_index=0,
        enterprise_node_id="enterprise-B",
        enterprise_privkey_b64=b_priv,
        sample_count=20,
    )

    # Submit to consortium
    c_orch.accept_gradient_update(a_bridged)
    c_orch.accept_gradient_update(b_bridged)

    # Aggregate at consortium
    c_round = c_orch.aggregate_round(c_job.job_id, 0)
    cross_org = decode_gradient(c_round.aggregated_update)
    # Enterprise A aggregates to [1,1,1]; enterprise B to
    # [3,3,3]; equal sample_count → FedAvg = [2,2,2]
    assert cross_org == pytest.approx(
        [2.0, 2.0, 2.0], abs=1e-6,
    )


def test_e2e_cross_org_with_transport_encryption():
    """Same as above but with end-to-end gradient encryption
    between enterprise and consortium — the consortium
    only sees sealed bytes until unseal time."""
    tx_priv, tx_pub = generate_transport_keypair()

    a_orch, a_job, a_workers = (
        _setup_enterprise_orchestrator(n_workers=2)
    )
    a_priv, a_pub = generate_worker_keypair()
    b_orch, b_job, b_workers = (
        _setup_enterprise_orchestrator(n_workers=2)
    )
    b_priv, b_pub = generate_worker_keypair()

    c_orch = FederatedLearningOrchestrator()
    c_orch.register_worker_key(
        WorkerKey("enterprise-A", a_pub),
    )
    c_orch.register_worker_key(
        WorkerKey("enterprise-B", b_pub),
    )
    c_job = c_orch.propose_job(
        model_id="x",
        dataset_cids=[],
        worker_pool=["enterprise-A", "enterprise-B"],
        rounds_target=1,
        min_workers_per_round=2,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
        transport_pubkey_b64=tx_pub,
    )
    c_orch.issue_round(c_job.job_id)

    for (node_id, priv) in a_workers:
        a_orch.accept_gradient_update(_worker_update(
            a_orch, a_job, node_id, priv, [2.0, 4.0],
        ))
    a_round = a_orch.aggregate_round(a_job.job_id, 0)
    for (node_id, priv) in b_workers:
        b_orch.accept_gradient_update(_worker_update(
            b_orch, b_job, node_id, priv, [6.0, 8.0],
        ))
    b_round = b_orch.aggregate_round(b_job.job_id, 0)

    a_bridged = aggregated_round_to_gradient_update(
        local_round=a_round,
        consortium_job_id=c_job.job_id,
        consortium_round_index=0,
        enterprise_node_id="enterprise-A",
        enterprise_privkey_b64=a_priv,
        sample_count=20,
        transport_pubkey_b64=tx_pub,
    )
    b_bridged = aggregated_round_to_gradient_update(
        local_round=b_round,
        consortium_job_id=c_job.job_id,
        consortium_round_index=0,
        enterprise_node_id="enterprise-B",
        enterprise_privkey_b64=b_priv,
        sample_count=20,
        transport_pubkey_b64=tx_pub,
    )
    # Bridged updates are sealed
    assert a_bridged.gradient_envelope_b64 is not None
    assert b_bridged.gradient_envelope_b64 is not None

    c_orch.accept_gradient_update(a_bridged)
    c_orch.accept_gradient_update(b_bridged)

    # Consortium needs the transport privkey
    import os as _os
    _os.environ[
        "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY"
    ] = tx_priv
    try:
        c_round = c_orch.aggregate_round(c_job.job_id, 0)
        cross_org = decode_gradient(
            c_round.aggregated_update,
        )
        assert cross_org == pytest.approx(
            [4.0, 6.0], abs=1e-6,
        )
    finally:
        _os.environ.pop(
            "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
            None,
        )


# ── Byzantine resistance at consortium level ────────


def test_e2e_cross_org_fedmedian_byzantine_enterprise():
    """One enterprise reports a malicious aggregate
    (extreme values trying to swing the cross-org model).
    With FedMedian at the consortium level, the malicious
    enterprise can't move the consortium aggregate."""
    enterprises = []  # (orch, job, workers, priv, pub, agg_value)
    for i, (n_workers, agg) in enumerate([
        (2, 1.0),
        (2, 1.0),
        (2, 1_000_000.0),  # byzantine
    ]):
        orch, job, workers = (
            _setup_enterprise_orchestrator(
                n_workers=n_workers,
            )
        )
        priv, pub = generate_worker_keypair()
        enterprises.append(
            (orch, job, workers, priv, pub, agg, f"e-{i}"),
        )

    c_orch = FederatedLearningOrchestrator()
    for (orch, job, workers, priv, pub, agg, eid) in enterprises:
        c_orch.register_worker_key(WorkerKey(eid, pub))
    c_job = c_orch.propose_job(
        model_id="x", dataset_cids=[],
        worker_pool=[e[6] for e in enterprises],
        rounds_target=1, min_workers_per_round=3,
        aggregation=AggregationStrategy.FEDMEDIAN,
        require_signed_updates=True,
    )
    c_orch.issue_round(c_job.job_id)

    for (orch, job, workers, priv, pub, agg, eid) in enterprises:
        for (node_id, wpriv) in workers:
            orch.accept_gradient_update(_worker_update(
                orch, job, node_id, wpriv, [agg, agg],
            ))
        local_round = orch.aggregate_round(job.job_id, 0)
        bridged = aggregated_round_to_gradient_update(
            local_round=local_round,
            consortium_job_id=c_job.job_id,
            consortium_round_index=0,
            enterprise_node_id=eid,
            enterprise_privkey_b64=priv,
            sample_count=20,
        )
        c_orch.accept_gradient_update(bridged)

    c_round = c_orch.aggregate_round(c_job.job_id, 0)
    cross_org = decode_gradient(c_round.aggregated_update)
    # Median of [1, 1, 1_000_000] = 1; byzantine fails
    assert cross_org == pytest.approx(
        [1.0, 1.0], abs=1e-6,
    )


# ── Cross-org reproducibility ───────────────────────


def test_e2e_cross_org_deterministic_with_same_inputs():
    """Running the whole cross-org pipeline twice with
    identical inputs produces identical cross-org
    aggregates. Critical for audit + replay."""
    def run_once(*, seed):
        a_orch, a_job, a_workers = (
            _setup_enterprise_orchestrator(n_workers=2)
        )
        a_priv, a_pub = generate_worker_keypair()
        b_orch, b_job, b_workers = (
            _setup_enterprise_orchestrator(n_workers=2)
        )
        b_priv, b_pub = generate_worker_keypair()

        c_orch = FederatedLearningOrchestrator()
        c_orch.register_worker_key(
            WorkerKey("eA", a_pub),
        )
        c_orch.register_worker_key(
            WorkerKey("eB", b_pub),
        )
        c_job = c_orch.propose_job(
            model_id="x", dataset_cids=[],
            worker_pool=["eA", "eB"],
            rounds_target=1, min_workers_per_round=2,
            aggregation=AggregationStrategy.FEDAVG,
        )
        c_orch.issue_round(c_job.job_id)

        for (node_id, priv) in a_workers:
            a_orch.accept_gradient_update(_worker_update(
                a_orch, a_job, node_id, priv,
                [1.5, 2.5, 3.5],
            ))
        a_round = a_orch.aggregate_round(a_job.job_id, 0)
        a_b = aggregated_round_to_gradient_update(
            local_round=a_round,
            consortium_job_id=c_job.job_id,
            consortium_round_index=0,
            enterprise_node_id="eA",
            enterprise_privkey_b64=a_priv,
            sample_count=20,
        )
        for (node_id, priv) in b_workers:
            b_orch.accept_gradient_update(_worker_update(
                b_orch, b_job, node_id, priv,
                [7.5, 8.5, 9.5],
            ))
        b_round = b_orch.aggregate_round(b_job.job_id, 0)
        b_b = aggregated_round_to_gradient_update(
            local_round=b_round,
            consortium_job_id=c_job.job_id,
            consortium_round_index=0,
            enterprise_node_id="eB",
            enterprise_privkey_b64=b_priv,
            sample_count=20,
        )
        c_orch.accept_gradient_update(a_b)
        c_orch.accept_gradient_update(b_b)
        c_round = c_orch.aggregate_round(c_job.job_id, 0)
        return decode_gradient(c_round.aggregated_update)

    r1 = run_once(seed=1)
    r2 = run_once(seed=1)
    for a, b in zip(r1, r2):
        assert abs(a - b) < 1e-6
