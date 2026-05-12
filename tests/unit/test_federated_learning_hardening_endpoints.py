"""Sprint 308a — hardening operator surface.

POST /admin/federated/worker-key  — register a worker
                                    Ed25519 signing pubkey
GET  /admin/federated/worker-key  — list registered keys
POST /admin/federated/job         — extended with
                                    `require_signed_updates`
                                    and `dp_policy`

prsm_federated_learning MCP tool gains actions:
register_worker_key | list_worker_keys.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.enterprise.federated_learning import (
    AggregationStrategy, DPPolicy,
    FederatedLearningOrchestrator, WorkerKey,
    encode_gradient, generate_worker_keypair,
    sign_gradient_update,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_federated_learning,
)
from prsm.node.api import create_api_app


def _client(orchestrator=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._federated_learning_orchestrator = orchestrator
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── /admin/federated/worker-key ─────────────────────


def test_register_worker_key_503_unwired():
    _, pub = generate_worker_keypair()
    resp = _client(orchestrator=None).post(
        "/admin/federated/worker-key",
        json={"node_id": "n1", "signing_pubkey_b64": pub},
    )
    assert resp.status_code == 503


def test_register_worker_key_happy_path():
    orch = FederatedLearningOrchestrator()
    _, pub = generate_worker_keypair()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/worker-key",
        json={"node_id": "n1", "signing_pubkey_b64": pub},
    )
    assert resp.status_code == 200
    assert orch.get_worker_key("n1") is not None


def test_register_worker_key_422_bad_pubkey():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/worker-key",
        json={
            "node_id": "n1",
            "signing_pubkey_b64": "not-base64!",
        },
    )
    assert resp.status_code == 422


def test_list_worker_keys():
    orch = FederatedLearningOrchestrator()
    _, pub = generate_worker_keypair()
    orch.register_worker_key(WorkerKey("n1", pub))
    body = _client(orchestrator=orch).get(
        "/admin/federated/worker-key",
    ).json()
    assert len(body["worker_keys"]) == 1
    assert body["worker_keys"][0]["node_id"] == "n1"


# ── propose-job extended schema ─────────────────────


def test_propose_with_require_signed_updates_and_dp_policy():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "require_signed_updates": True,
            "dp_policy": {
                "epsilon": 1.0, "delta": 1e-5,
                "clip_norm": 1.0,
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["require_signed_updates"] is True
    assert body["dp_policy"]["epsilon"] == 1.0


def test_propose_422_invalid_dp_policy():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "dp_policy": {
                "epsilon": -1.0,  # invalid
                "delta": 1e-5, "clip_norm": 1.0,
            },
        },
    )
    assert resp.status_code == 422


# ── End-to-end signed-update flow via HTTP ─────────


def test_e2e_signed_update_via_http():
    orch = FederatedLearningOrchestrator()
    priv, pub = generate_worker_keypair()
    c = _client(orchestrator=orch)
    # Register worker key
    c.post(
        "/admin/federated/worker-key",
        json={"node_id": "n1", "signing_pubkey_b64": pub},
    )
    # Propose signed-required job
    job_resp = c.post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "require_signed_updates": True,
        },
    )
    job_id = job_resp.json()["job_id"]
    c.post(
        f"/admin/federated/job/{job_id}/issue-round",
    )
    # Sign + submit
    from prsm.enterprise.federated_learning import (
        GradientUpdate,
    )
    u = GradientUpdate(
        job_id=job_id, round_index=0,
        worker_node_id="n1",
        gradient_b64=base64.b64encode(
            encode_gradient([1.0, 2.0]),
        ).decode(),
        sample_count=10,
        worker_attestation_b64="",
        worker_signature_b64="",
        timestamp=100.0,
    )
    u = sign_gradient_update(u, worker_privkey_b64=priv)
    r = c.post(
        f"/admin/federated/job/{job_id}/update",
        json={
            "round_index": u.round_index,
            "worker_node_id": u.worker_node_id,
            "gradient_b64": u.gradient_b64,
            "sample_count": u.sample_count,
            "worker_attestation_b64": (
                u.worker_attestation_b64
            ),
            "worker_signature_b64": u.worker_signature_b64,
            "timestamp": u.timestamp,
        },
    )
    assert r.status_code == 200


def test_e2e_unsigned_update_refused_when_required():
    orch = FederatedLearningOrchestrator()
    _, pub = generate_worker_keypair()
    c = _client(orchestrator=orch)
    c.post(
        "/admin/federated/worker-key",
        json={"node_id": "n1", "signing_pubkey_b64": pub},
    )
    job_resp = c.post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "require_signed_updates": True,
        },
    )
    job_id = job_resp.json()["job_id"]
    c.post(
        f"/admin/federated/job/{job_id}/issue-round",
    )
    # No signature on the update
    r = c.post(
        f"/admin/federated/job/{job_id}/update",
        json={
            "round_index": 0, "worker_node_id": "n1",
            "gradient_b64": base64.b64encode(
                encode_gradient([1.0]),
            ).decode(),
            "sample_count": 10,
            "worker_attestation_b64": "",
            "worker_signature_b64": "",  # MISSING
            "timestamp": 100.0,
        },
    )
    assert r.status_code == 422


# ── MCP ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_register_worker_key():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "node_id": "n1",
            "signing_pubkey_b64": "AAAA",
        }),
    ) as mock_call:
        r = await handle_prsm_federated_learning({
            "action": "register_worker_key",
            "node_id": "n1",
            "signing_pubkey_b64": "AAAA",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/federated/worker-key"
    assert "n1" in r


@pytest.mark.asyncio
async def test_mcp_list_worker_keys():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "worker_keys": [{
                "node_id": "n1",
                "signing_pubkey_b64": "AAAAAAAA",
            }],
        }),
    ):
        r = await handle_prsm_federated_learning({
            "action": "list_worker_keys",
        })
    assert "n1" in r


@pytest.mark.asyncio
async def test_mcp_register_worker_key_requires_fields():
    r = await handle_prsm_federated_learning({
        "action": "register_worker_key",
    })
    assert (
        "node_id" in r.lower()
        or "signing_pubkey" in r.lower()
    )
