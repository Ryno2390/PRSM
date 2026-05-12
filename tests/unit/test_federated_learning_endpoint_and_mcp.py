"""Sprint 308 — federated-learning HTTP + MCP.

POST /admin/federated/job            — propose a new job
GET  /admin/federated/job            — list jobs
GET  /admin/federated/job/{id}       — job detail
POST /admin/federated/job/{id}/issue-round    — issue next round
POST /admin/federated/job/{id}/update — accept a gradient update
POST /admin/federated/job/{id}/aggregate/{round} — aggregate round
GET  /admin/federated/job/{id}/round/{round}   — round detail

prsm_federated_learning MCP tool.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.enterprise.federated_learning import (
    AggregationStrategy,
    FederatedLearningOrchestrator,
    JobStatus,
    encode_gradient,
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


def _b64g(values):
    return base64.b64encode(
        encode_gradient(values),
    ).decode()


# ── propose ────────────────────────────────────────


def test_propose_503_unwired():
    resp = _client(orchestrator=None).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": [],
            "worker_pool": [], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
        },
    )
    assert resp.status_code == 503


def test_propose_happy_path():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "mock-llama-3-8b",
            "dataset_cids": ["QmA", "QmB"],
            "worker_pool": ["n1", "n2", "n3"],
            "rounds_target": 5,
            "min_workers_per_round": 2,
            "aggregation": "fedavg",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "proposed"
    assert body["job_id"]


def test_propose_422_invalid_aggregation():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 1,
            "aggregation": "unknown-strategy",
        },
    )
    assert resp.status_code == 422


def test_propose_422_min_workers_over_pool():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job",
        json={
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"], "rounds_target": 1,
            "min_workers_per_round": 5,
            "aggregation": "fedavg",
        },
    )
    assert resp.status_code == 422


# ── list / get ─────────────────────────────────────


def test_list_filter_by_status():
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
    body = _client(orchestrator=orch).get(
        "/admin/federated/job?status=proposed",
    ).json()
    ids = [j["job_id"] for j in body["jobs"]]
    assert ids == [j1.job_id]


def test_get_job_404_unknown():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).get(
        "/admin/federated/job/no-such",
    )
    assert resp.status_code == 404


def test_get_job_happy_path():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    body = _client(orchestrator=orch).get(
        f"/admin/federated/job/{job.job_id}",
    ).json()
    assert body["job_id"] == job.job_id


# ── issue-round / update / aggregate ────────────────


def test_issue_round_happy_path():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    resp = _client(orchestrator=orch).post(
        f"/admin/federated/job/{job.job_id}/issue-round",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "issued"
    assert body["round_index"] == 0


def test_update_then_aggregate_happy_path():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    c = _client(orchestrator=orch)
    r = c.post(
        f"/admin/federated/job/{job.job_id}/update",
        json={
            "round_index": 0,
            "worker_node_id": "n1",
            "gradient_b64": _b64g([1.0, 2.0, 3.0]),
            "sample_count": 50,
            "worker_attestation_b64": "",
            "worker_signature_b64": "",
            "timestamp": 100.0,
        },
    )
    assert r.status_code == 200
    r2 = c.post(
        f"/admin/federated/job/{job.job_id}"
        f"/aggregate/0",
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "aggregated"
    assert body["aggregated_update_b64"]


def test_update_422_unknown_worker():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=1,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    resp = _client(orchestrator=orch).post(
        f"/admin/federated/job/{job.job_id}/update",
        json={
            "round_index": 0,
            "worker_node_id": "intruder",
            "gradient_b64": _b64g([1.0]),
            "sample_count": 10,
            "worker_attestation_b64": "",
            "worker_signature_b64": "",
            "timestamp": 100.0,
        },
    )
    assert resp.status_code == 422


def test_aggregate_404_unknown_job():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).post(
        "/admin/federated/job/no-such/aggregate/0",
    )
    assert resp.status_code == 404


# ── round detail ────────────────────────────────────


def test_round_detail_404_unknown():
    orch = FederatedLearningOrchestrator()
    resp = _client(orchestrator=orch).get(
        "/admin/federated/job/no-such/round/0",
    )
    assert resp.status_code == 404


def test_round_detail_happy_path():
    orch = FederatedLearningOrchestrator()
    job = orch.propose_job(
        model_id="x", dataset_cids=["QmA"],
        worker_pool=["n1"], rounds_target=2,
        min_workers_per_round=1,
        aggregation=AggregationStrategy.FEDAVG,
    )
    orch.issue_round(job.job_id)
    body = _client(orchestrator=orch).get(
        f"/admin/federated/job/{job.job_id}/round/0",
    ).json()
    assert body["round_index"] == 0
    assert body["status"] == "issued"


# ── MCP ────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_federated_learning" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_federated_learning({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_federated_learning(
        {"action": "boom"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_propose():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1", "status": "proposed",
            "current_round": 0, "rounds_target": 5,
            "model_id": "x", "dataset_cids": ["QmA"],
            "worker_pool": ["n1"],
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "started_at": 100.0, "completed_at": None,
        }),
    ) as mock_call:
        r = await handle_prsm_federated_learning({
            "action": "propose",
            "model_id": "x",
            "dataset_cids": ["QmA"],
            "worker_pool": ["n1"],
            "rounds_target": 5,
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/federated/job"
    assert "j-1" in r


@pytest.mark.asyncio
async def test_mcp_list():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "jobs": [{
                "job_id": "j-abcdef0123",
                "status": "running",
                "current_round": 2,
                "rounds_target": 5,
                "model_id": "x",
                "dataset_cids": [],
                "worker_pool": [],
                "min_workers_per_round": 1,
                "aggregation": "fedavg",
                "started_at": 0.0,
                "completed_at": None,
            }],
        }),
    ):
        r = await handle_prsm_federated_learning({
            "action": "list",
        })
    assert "j-abcdef" in r
    assert "running" in r.lower()
    assert "2/5" in r or "2" in r


@pytest.mark.asyncio
async def test_mcp_issue_round():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1",
            "round_index": 0,
            "status": "issued",
            "worker_assignments": [
                {"node_id": "n1", "dataset_cid": "QmA",
                 "assigned_at": 100.0},
            ],
            "gradient_updates_received": [],
            "aggregated_update_b64": "",
            "issued_at": 100.0,
            "completed_at": None,
        }),
    ) as mock_call:
        r = await handle_prsm_federated_learning({
            "action": "issue_round",
            "job_id": "j-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/federated/job/j-1/issue-round"
    )
    assert "issued" in r.lower()
    assert "n1" in r


@pytest.mark.asyncio
async def test_mcp_aggregate():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1",
            "round_index": 0,
            "status": "aggregated",
            "worker_assignments": [],
            "gradient_updates_received": [],
            "aggregated_update_b64": "AAAA",
            "issued_at": 100.0,
            "completed_at": 200.0,
        }),
    ) as mock_call:
        r = await handle_prsm_federated_learning({
            "action": "aggregate",
            "job_id": "j-1",
            "round_index": 0,
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/federated/job/j-1/aggregate/0"
    )
    assert "aggregated" in r.lower()


@pytest.mark.asyncio
async def test_mcp_lookup():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1",
            "status": "running",
            "current_round": 1, "rounds_target": 3,
            "model_id": "x",
            "dataset_cids": ["QmA"],
            "worker_pool": ["n1"],
            "min_workers_per_round": 1,
            "aggregation": "fedavg",
            "started_at": 0.0, "completed_at": None,
        }),
    ) as mock_call:
        r = await handle_prsm_federated_learning({
            "action": "lookup", "job_id": "j-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/federated/job/j-1"
    assert "j-1" in r
    assert "1/3" in r or "running" in r.lower()


@pytest.mark.asyncio
async def test_mcp_propose_requires_fields():
    r = await handle_prsm_federated_learning({
        "action": "propose",
    })
    assert (
        "model_id" in r.lower()
        or "missing" in r.lower()
        or "required" in r.lower()
    )


@pytest.mark.asyncio
async def test_mcp_issue_round_requires_job_id():
    r = await handle_prsm_federated_learning({
        "action": "issue_round",
    })
    assert "job_id" in r.lower()
