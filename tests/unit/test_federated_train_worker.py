"""Sprint 308b — worker-side /compute/train shim.

The orchestrator dispatches a round (sprint 308) and
assigns workers. Each worker calls its own /compute/train
endpoint, which runs a pluggable training strategy on the
assigned dataset shard inside the TEE, signs the resulting
gradient with the worker's Ed25519 privkey (sprint 308a),
binds the worker's TEE attestation blob into the signed
payload (sprint 308b — new this sprint), and returns the
signed update to the caller. The caller (orchestrator,
enterprise, or whoever ran the round) then POSTs the
update to /admin/federated/job/{id}/update.

This factoring keeps the worker stateless w.r.t. the
orchestrator URL — distribution is the orchestrator's
concern, not the worker's.

Two layers tested here:
  1. The training shim (primitive)
  2. The /compute/train HTTP endpoint
"""
from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.compute.train import (
    TrainingStrategy,
    compute_signed_gradient_update,
    deterministic_stub_train_fn,
)
from prsm.enterprise.federated_learning import (
    decode_gradient,
    generate_worker_keypair,
    verify_gradient_update_signature,
)
from prsm.node.api import create_api_app


# ── Deterministic stub training function ────────────


def test_stub_returns_fixed_length_gradient():
    g = deterministic_stub_train_fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
    )
    # The stub is deterministic; default length is 8
    assert isinstance(g, list)
    assert len(g) == 8


def test_stub_is_deterministic_for_same_inputs():
    a = deterministic_stub_train_fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
    )
    b = deterministic_stub_train_fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
    )
    assert a == b


def test_stub_varies_with_inputs():
    a = deterministic_stub_train_fn(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
    )
    b = deterministic_stub_train_fn(
        job_id="j1", round_index=1,
        dataset_cid="QmA", sample_count=10,
    )
    c = deterministic_stub_train_fn(
        job_id="j1", round_index=0,
        dataset_cid="QmB", sample_count=10,
    )
    assert a != b
    assert a != c


# ── compute_signed_gradient_update ──────────────────


def test_compute_signed_update_produces_valid_signature():
    priv, pub = generate_worker_keypair()
    update = compute_signed_gradient_update(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=priv,
        worker_attestation_b64="",
    )
    assert verify_gradient_update_signature(update, pub)


def test_compute_signed_update_carries_attestation_in_sig():
    """Sprint 308b — worker_attestation_b64 is now part
    of the signed payload. Tampering with it must break
    verification."""
    priv, pub = generate_worker_keypair()
    blob_b64 = base64.b64encode(b"valid-attestation").decode()
    update = compute_signed_gradient_update(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=10,
        worker_node_id="n1",
        worker_privkey_b64=priv,
        worker_attestation_b64=blob_b64,
    )
    assert verify_gradient_update_signature(update, pub)
    # Tamper: replace attestation
    update.worker_attestation_b64 = base64.b64encode(
        b"forged-attestation",
    ).decode()
    assert not verify_gradient_update_signature(
        update, pub,
    )


def test_compute_signed_update_with_custom_strategy():
    """Pluggable training fn — caller supplies their own
    train fn instead of the stub."""
    priv, pub = generate_worker_keypair()

    def custom_train(*, job_id, round_index,
                     dataset_cid, sample_count):
        return [42.0, 42.0, 42.0]

    update = compute_signed_gradient_update(
        job_id="j1", round_index=0,
        dataset_cid="QmA", sample_count=5,
        worker_node_id="n1",
        worker_privkey_b64=priv,
        worker_attestation_b64="",
        train_fn=custom_train,
    )
    grad = decode_gradient(
        base64.b64decode(update.gradient_b64),
    )
    assert grad == pytest.approx(
        [42.0, 42.0, 42.0], abs=1e-6,
    )


def test_compute_signed_update_fields_set():
    priv, _ = generate_worker_keypair()
    update = compute_signed_gradient_update(
        job_id="j-xyz", round_index=3,
        dataset_cid="QmA", sample_count=50,
        worker_node_id="n7",
        worker_privkey_b64=priv,
        worker_attestation_b64="att",
    )
    assert update.job_id == "j-xyz"
    assert update.round_index == 3
    assert update.worker_node_id == "n7"
    assert update.sample_count == 50
    assert update.worker_attestation_b64 == "att"
    assert update.worker_signature_b64  # populated


# ── /compute/train endpoint ─────────────────────────


def _client(*, worker_privkey=None, attestation_blob=None,
            node_id="worker-1"):
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = node_id
    node.ftns_ledger = None
    node._tee_node_attestation_blob = attestation_blob
    node._federated_worker_privkey_b64 = worker_privkey
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_train_endpoint_503_without_privkey():
    """Worker hasn't been configured with a signing key —
    refuse loud so the operator notices."""
    resp = _client(worker_privkey=None).post(
        "/compute/train",
        json={
            "job_id": "j1",
            "round_index": 0,
            "dataset_cid": "QmA",
            "sample_count": 10,
        },
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert (
        "privkey" in detail.lower()
        or "key" in detail.lower()
    )


def test_train_endpoint_happy_path_with_stub():
    priv, pub = generate_worker_keypair()
    resp = _client(worker_privkey=priv).post(
        "/compute/train",
        json={
            "job_id": "j1",
            "round_index": 0,
            "dataset_cid": "QmA",
            "sample_count": 10,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "j1"
    assert body["round_index"] == 0
    assert body["worker_node_id"] == "worker-1"
    assert body["sample_count"] == 10
    assert body["gradient_b64"]
    assert body["worker_signature_b64"]

    # Reconstruct as GradientUpdate and verify signature
    from prsm.enterprise.federated_learning import (
        GradientUpdate,
    )
    update = GradientUpdate.from_dict(body)
    assert verify_gradient_update_signature(update, pub)


def test_train_endpoint_carries_node_attestation():
    """If the worker has an attestation blob wired, it
    flows through into the signed update so the
    orchestrator can audit the TEE that produced this
    gradient."""
    priv, pub = generate_worker_keypair()
    blob = b"my-tee-quote-bytes"
    blob_b64 = base64.b64encode(blob).decode()
    resp = _client(
        worker_privkey=priv, attestation_blob=blob,
    ).post(
        "/compute/train",
        json={
            "job_id": "j1", "round_index": 0,
            "dataset_cid": "QmA", "sample_count": 10,
        },
    )
    body = resp.json()
    assert body["worker_attestation_b64"] == blob_b64

    # Signature binds the attestation
    from prsm.enterprise.federated_learning import (
        GradientUpdate,
    )
    update = GradientUpdate.from_dict(body)
    assert verify_gradient_update_signature(update, pub)
    # Tamper-detection sanity: changing attestation breaks
    # the signature
    update.worker_attestation_b64 = "forged"
    assert not verify_gradient_update_signature(
        update, pub,
    )


def test_train_endpoint_422_missing_job_id():
    priv, _ = generate_worker_keypair()
    resp = _client(worker_privkey=priv).post(
        "/compute/train",
        json={
            "round_index": 0,
            "dataset_cid": "QmA",
            "sample_count": 10,
        },
    )
    assert resp.status_code == 422


def test_train_endpoint_422_negative_round_index():
    priv, _ = generate_worker_keypair()
    resp = _client(worker_privkey=priv).post(
        "/compute/train",
        json={
            "job_id": "j1",
            "round_index": -1,
            "dataset_cid": "QmA",
            "sample_count": 10,
        },
    )
    assert resp.status_code == 422


def test_train_endpoint_422_negative_sample_count():
    priv, _ = generate_worker_keypair()
    resp = _client(worker_privkey=priv).post(
        "/compute/train",
        json={
            "job_id": "j1", "round_index": 0,
            "dataset_cid": "QmA",
            "sample_count": -5,
        },
    )
    assert resp.status_code == 422


def test_train_endpoint_503_invalid_privkey_format():
    """Operator misconfigured the privkey env (wrong
    length, bad base64). Refuse with 503 so they notice."""
    resp = _client(worker_privkey="not-base64!").post(
        "/compute/train",
        json={
            "job_id": "j1", "round_index": 0,
            "dataset_cid": "QmA", "sample_count": 10,
        },
    )
    assert resp.status_code == 503


# ── End-to-end: train then submit ───────────────────


def test_e2e_train_then_orchestrator_accepts():
    """Whole loop: worker trains → caller takes signed
    update → orchestrator accepts. Verifies the wire
    format matches what /admin/federated/job/.../update
    expects."""
    from prsm.enterprise.federated_learning import (
        AggregationStrategy, FederatedLearningOrchestrator,
        GradientUpdate, WorkerKey,
    )

    # Set up orchestrator with the worker key registered
    orch = FederatedLearningOrchestrator()
    priv, pub = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("worker-1", pub))
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["worker-1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
        require_signed_updates=True,
    )
    orch.issue_round(job.job_id)

    # Worker trains
    train_resp = _client(
        worker_privkey=priv, node_id="worker-1",
    ).post(
        "/compute/train",
        json={
            "job_id": job.job_id, "round_index": 0,
            "dataset_cid": "QmA", "sample_count": 10,
        },
    )
    assert train_resp.status_code == 200

    # Caller submits to orchestrator
    update = GradientUpdate.from_dict(train_resp.json())
    orch.accept_gradient_update(update)
    rnd = orch.get_round(job.job_id, 0)
    assert len(rnd.gradient_updates_received) == 1


# ── MCP tool ────────────────────────────────────────


def test_mcp_tool_registered():
    from prsm.mcp_server import TOOL_HANDLERS
    assert "prsm_federated_train" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_job_id():
    from prsm.mcp_server import handle_prsm_federated_train
    r = await handle_prsm_federated_train({})
    assert "job_id" in r.lower()


@pytest.mark.asyncio
async def test_mcp_train_round_calls_compute_endpoint():
    from unittest.mock import AsyncMock, patch
    from prsm.mcp_server import handle_prsm_federated_train
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1", "round_index": 0,
            "worker_node_id": "w1",
            "gradient_b64": "AAAA",
            "sample_count": 10,
            "worker_attestation_b64": "",
            "worker_signature_b64": "BBBB",
            "timestamp": 100.0,
        }),
    ) as mock_call:
        r = await handle_prsm_federated_train({
            "job_id": "j-1",
            "round_index": 0,
            "dataset_cid": "QmA",
            "sample_count": 10,
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/compute/train"
    assert "j-1" in r
    assert "w1" in r
