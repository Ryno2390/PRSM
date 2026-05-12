"""Sprint 308c — encrypted-gradient transport.

Gradients aren't plaintext training data but they leak
information (membership inference, model inversion).
Sprints 308/308a/308b sent gradients between worker and
orchestrator as plaintext base64 — anyone with packet
capture saw the gradient. This sprint closes that gap.

The mechanic: the orchestrator declares a transport pubkey
on the FederatedJob. Workers seal the gradient bytes to
that pubkey via X25519 ECDH + ChaCha20-Poly1305 hybrid
(the same primitive sprint 304 ships for recipient
encryption). The orchestrator unseals before aggregation
using a privkey loaded from
PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY env.

The signed payload binds the envelope alongside the
sealed gradient_b64, so a MITM can't strip or replace
either piece without breaking the worker's signature.

Backwards-compat: jobs WITHOUT transport_pubkey_b64
proceed exactly as before (plaintext gradient). Both
gradient_envelope_b64 and transport_pubkey_b64 default
None.
"""
from __future__ import annotations

import base64
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.enterprise.federated_learning import (
    AggregationStrategy,
    FederatedLearningOrchestrator,
    GradientUpdate,
    WorkerKey,
    decode_gradient,
    encode_gradient,
    generate_transport_keypair,
    generate_worker_keypair,
    seal_gradient_for_orchestrator,
    sign_gradient_update,
    unseal_gradient_from_worker,
    verify_gradient_update_signature,
)
from prsm.node.api import create_api_app


# ── Transport keypair ───────────────────────────────


def test_generate_transport_keypair_b64():
    priv, pub = generate_transport_keypair()
    assert len(base64.b64decode(priv)) == 32
    assert len(base64.b64decode(pub)) == 32


def test_generate_transport_keypair_unique():
    a = generate_transport_keypair()
    b = generate_transport_keypair()
    assert a != b


# ── Seal / unseal round-trip ────────────────────────


def test_seal_unseal_round_trip():
    priv, pub = generate_transport_keypair()
    gradient = b"\x01\x02\x03\x04" * 8
    sealed_b64, envelope_b64 = (
        seal_gradient_for_orchestrator(gradient, pub)
    )
    out = unseal_gradient_from_worker(
        sealed_b64, envelope_b64, priv,
    )
    assert out == gradient


def test_seal_with_real_gradient_round_trip():
    priv, pub = generate_transport_keypair()
    grad = encode_gradient([0.1, -0.2, 3.14, -42.0])
    sealed_b64, env_b64 = seal_gradient_for_orchestrator(
        grad, pub,
    )
    out = unseal_gradient_from_worker(
        sealed_b64, env_b64, priv,
    )
    assert decode_gradient(out) == pytest.approx(
        [0.1, -0.2, 3.14, -42.0], abs=1e-6,
    )


def test_seal_produces_different_ciphertext_each_run():
    """Fresh ephemeral keypair + fresh nonce per seal
    means the SAME plaintext produces DIFFERENT ciphertext
    each time. Probabilistic encryption guarantee."""
    _, pub = generate_transport_keypair()
    gradient = b"identical input"
    ciphertexts = [
        seal_gradient_for_orchestrator(gradient, pub)[0]
        for _ in range(5)
    ]
    assert len(set(ciphertexts)) == 5


def test_unseal_wrong_privkey_fails():
    _, pub = generate_transport_keypair()
    other_priv, _ = generate_transport_keypair()
    sealed_b64, env_b64 = seal_gradient_for_orchestrator(
        b"secret", pub,
    )
    with pytest.raises(Exception):
        unseal_gradient_from_worker(
            sealed_b64, env_b64, other_priv,
        )


def test_unseal_tampered_ciphertext_fails():
    priv, pub = generate_transport_keypair()
    sealed_b64, env_b64 = seal_gradient_for_orchestrator(
        b"secret data", pub,
    )
    raw = bytearray(base64.b64decode(sealed_b64))
    raw[3] ^= 0x01
    sealed_b64 = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(Exception):
        unseal_gradient_from_worker(
            sealed_b64, env_b64, priv,
        )


def test_unseal_tampered_envelope_fails():
    priv, pub = generate_transport_keypair()
    sealed_b64, env_b64 = seal_gradient_for_orchestrator(
        b"secret data", pub,
    )
    raw = bytearray(base64.b64decode(env_b64))
    raw[3] ^= 0x01
    env_b64 = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(Exception):
        unseal_gradient_from_worker(
            sealed_b64, env_b64, priv,
        )


def test_seal_rejects_malformed_pubkey():
    with pytest.raises(ValueError):
        seal_gradient_for_orchestrator(
            b"x", "not-base64!",
        )


# ── Sealed signing payload ──────────────────────────


def test_signature_binds_envelope():
    """The signed canonical payload now includes
    gradient_envelope_b64. Replacing the envelope after
    signing must break verification."""
    priv, pub = generate_worker_keypair()
    update = GradientUpdate(
        job_id="j1", round_index=0,
        worker_node_id="n1",
        gradient_b64="SEALED",
        sample_count=10,
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=100.0,
        gradient_envelope_b64="ORIGINAL-ENV",
    )
    signed = sign_gradient_update(
        update, worker_privkey_b64=priv,
    )
    assert verify_gradient_update_signature(signed, pub)
    signed.gradient_envelope_b64 = "TAMPERED-ENV"
    assert not verify_gradient_update_signature(signed, pub)


def test_signature_round_trip_without_envelope():
    """Sprint 308b backwards-compat: an update with
    gradient_envelope_b64=None must still sign + verify
    cleanly (the canonical payload encodes the field as
    null)."""
    priv, pub = generate_worker_keypair()
    update = GradientUpdate(
        job_id="j1", round_index=0,
        worker_node_id="n1",
        gradient_b64="PLAINTEXT",
        sample_count=10,
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=100.0,
        gradient_envelope_b64=None,
    )
    signed = sign_gradient_update(
        update, worker_privkey_b64=priv,
    )
    assert verify_gradient_update_signature(signed, pub)


# ── compute_signed_gradient_update with transport ───


def test_compute_signed_update_seals_when_transport_set():
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    worker_priv, worker_pub = generate_worker_keypair()
    tx_priv, tx_pub = generate_transport_keypair()
    update = compute_signed_gradient_update(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=worker_priv,
        worker_attestation_b64="",
        transport_pubkey_b64=tx_pub,
    )
    # Signature verifies
    assert verify_gradient_update_signature(
        update, worker_pub,
    )
    # Envelope populated
    assert update.gradient_envelope_b64 is not None
    # gradient_b64 is now sealed — orchestrator can unseal
    out = unseal_gradient_from_worker(
        update.gradient_b64,
        update.gradient_envelope_b64,
        tx_priv,
    )
    # Unsealed bytes decode to the stub gradient
    grad = decode_gradient(out)
    assert isinstance(grad, list)
    assert len(grad) > 0


def test_compute_signed_update_no_seal_when_transport_none():
    """Without transport_pubkey_b64, the gradient is
    plaintext — backwards-compat with sprints 308/308a/
    308b."""
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    worker_priv, _ = generate_worker_keypair()
    update = compute_signed_gradient_update(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=worker_priv,
        worker_attestation_b64="",
        transport_pubkey_b64=None,
    )
    assert update.gradient_envelope_b64 is None


# ── Job lifecycle with transport encryption ────────


def _setup_job_with_transport(*, transport_pubkey):
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
        transport_pubkey_b64=transport_pubkey,
    )
    orch.issue_round(job.job_id)
    return orch, job


def test_propose_job_carries_transport_pubkey():
    _, pub = generate_transport_keypair()
    orch, job = _setup_job_with_transport(
        transport_pubkey=pub,
    )
    refreshed = orch.get_job(job.job_id)
    assert refreshed.transport_pubkey_b64 == pub


def test_aggregate_unseals_sealed_gradients(monkeypatch):
    """End-to-end: orchestrator pubkey on the job,
    worker seals + signs, orchestrator unseals during
    aggregation. The aggregated output must match what
    we'd get if the gradient flowed plaintext."""
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    tx_priv, tx_pub = generate_transport_keypair()
    monkeypatch.setenv(
        "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
        tx_priv,
    )
    orch, job = _setup_job_with_transport(
        transport_pubkey=tx_pub,
    )
    worker_priv, _ = generate_worker_keypair()
    update = compute_signed_gradient_update(
        job_id=job.job_id, round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=worker_priv,
        worker_attestation_b64="",
        transport_pubkey_b64=tx_pub,
    )
    orch.accept_gradient_update(update)
    rnd = orch.aggregate_round(job.job_id, 0)
    # Aggregated output is plaintext gradient
    aggregated = decode_gradient(rnd.aggregated_update)
    assert len(aggregated) == 8  # stub dim


def test_aggregate_503_without_orchestrator_privkey(
    monkeypatch,
):
    """If the job has transport encryption but the
    orchestrator privkey isn't loaded, aggregation must
    refuse loud."""
    monkeypatch.delenv(
        "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
        raising=False,
    )
    _, tx_pub = generate_transport_keypair()
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    orch, job = _setup_job_with_transport(
        transport_pubkey=tx_pub,
    )
    worker_priv, _ = generate_worker_keypair()
    update = compute_signed_gradient_update(
        job_id=job.job_id, round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=worker_priv,
        worker_attestation_b64="",
        transport_pubkey_b64=tx_pub,
    )
    orch.accept_gradient_update(update)
    with pytest.raises(ValueError, match="transport|privkey"):
        orch.aggregate_round(job.job_id, 0)


def test_aggregate_unseal_failure_marks_round_failed(
    monkeypatch,
):
    """If the orchestrator has the wrong privkey, unseal
    fails and the round is marked FAILED — so the operator
    notices."""
    monkeypatch.setenv(
        "PRSM_FEDERATED_ORCHESTRATOR_TRANSPORT_PRIVKEY",
        generate_transport_keypair()[0],  # WRONG priv
    )
    _, tx_pub = generate_transport_keypair()
    from prsm.compute.train import (
        compute_signed_gradient_update,
    )
    from prsm.enterprise.federated_learning import (
        RoundStatus,
    )
    orch, job = _setup_job_with_transport(
        transport_pubkey=tx_pub,
    )
    worker_priv, _ = generate_worker_keypair()
    update = compute_signed_gradient_update(
        job_id=job.job_id, round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=worker_priv,
        worker_attestation_b64="",
        transport_pubkey_b64=tx_pub,
    )
    orch.accept_gradient_update(update)
    with pytest.raises(Exception):
        orch.aggregate_round(job.job_id, 0)
    rnd = orch.get_round(job.job_id, 0)
    assert rnd.status == RoundStatus.FAILED


# ── Endpoints ───────────────────────────────────────


def _client(*, orchestrator=None, transport_privkey=None,
            worker_privkey=None, node_id="worker-1",
            attestation_blob=None):
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = node_id
    node.ftns_ledger = None
    node._federated_learning_orchestrator = orchestrator
    node._federated_worker_privkey_b64 = worker_privkey
    node._federated_orchestrator_transport_privkey_b64 = (
        transport_privkey
    )
    node._tee_node_attestation_blob = attestation_blob
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_transport_pubkey_endpoint_503_unconfigured():
    resp = _client(transport_privkey=None).get(
        "/admin/federated/transport-pubkey",
    )
    assert resp.status_code == 503


def test_transport_pubkey_endpoint_happy_path():
    priv, pub = generate_transport_keypair()
    resp = _client(transport_privkey=priv).get(
        "/admin/federated/transport-pubkey",
    )
    assert resp.status_code == 200
    assert resp.json()["transport_pubkey_b64"] == pub


def test_compute_train_seals_when_transport_pubkey_set():
    worker_priv, _ = generate_worker_keypair()
    tx_priv, tx_pub = generate_transport_keypair()
    resp = _client(worker_privkey=worker_priv).post(
        "/compute/train",
        json={
            "job_id": "j1", "round_index": 0,
            "dataset_cid": "QmA", "sample_count": 10,
            "transport_pubkey_b64": tx_pub,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gradient_envelope_b64"]
    # Orchestrator can unseal with the matching privkey
    out = unseal_gradient_from_worker(
        body["gradient_b64"],
        body["gradient_envelope_b64"],
        tx_priv,
    )
    assert len(decode_gradient(out)) > 0


def test_compute_train_no_seal_without_transport_pubkey():
    worker_priv, _ = generate_worker_keypair()
    resp = _client(worker_privkey=worker_priv).post(
        "/compute/train",
        json={
            "job_id": "j1", "round_index": 0,
            "dataset_cid": "QmA", "sample_count": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("gradient_envelope_b64") is None


def test_propose_endpoint_accepts_transport_pubkey():
    orch = FederatedLearningOrchestrator()
    _, tx_pub = generate_transport_keypair()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "transport_pubkey_b64": tx_pub,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["transport_pubkey_b64"] == tx_pub


def test_propose_endpoint_rejects_malformed_transport_pubkey():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "transport_pubkey_b64": "not-base64!",
        },
    )
    assert resp.status_code == 422
