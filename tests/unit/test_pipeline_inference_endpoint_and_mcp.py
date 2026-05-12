"""Sprint 312 — pipeline inference HTTP + MCP surface.

POST /admin/inference/pipeline/job            — propose
GET  /admin/inference/pipeline/job            — list
GET  /admin/inference/pipeline/job/{id}       — detail
POST /admin/inference/pipeline/job/{id}/execute — run +
                                                 receipt
GET  /admin/inference/pipeline/job/{id}/round — round +
                                                receipt
                                                detail

prsm_pipeline_inference MCP tool — propose | list | lookup
                                  | execute | get_round.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.pipeline_orchestrator import (
    PipelineInferenceOrchestrator,
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
from prsm.enterprise.federated_learning import (
    generate_worker_keypair,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_pipeline_inference,
)
from prsm.node.api import create_api_app


def _client(orchestrator=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._pipeline_inference_orchestrator = orchestrator
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _orch():
    priv, pub = generate_worker_keypair()
    return (
        PipelineInferenceOrchestrator(
            orchestrator_privkey_b64=priv,
        ),
        priv, pub,
    )


# ── propose ────────────────────────────────────────


def test_propose_503_unwired():
    resp = _client(orchestrator=None).post(
        "/admin/inference/pipeline/job",
        json={
            "model_id": "m1",
            "partition": {
                "total_layers": 4,
                "stage_layer_ranges": [[0, 1], [2, 3]],
                "stage_node_ids": ["n0", "n1"],
            },
        },
    )
    assert resp.status_code == 503


def test_propose_happy_path():
    orch, _, _ = _orch()
    resp = _client(orchestrator=orch).post(
        "/admin/inference/pipeline/job",
        json={
            "model_id": "m1",
            "partition": {
                "total_layers": 4,
                "stage_layer_ranges": [[0, 1], [2, 3]],
                "stage_node_ids": ["n0", "n1"],
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "proposed"
    assert body["job_id"]


def test_propose_422_invalid_partition():
    orch, _, _ = _orch()
    resp = _client(orchestrator=orch).post(
        "/admin/inference/pipeline/job",
        json={
            "model_id": "m1",
            "partition": {
                "total_layers": 4,
                "stage_layer_ranges": [[0, 1]],  # gap
                "stage_node_ids": ["n0"],
            },
        },
    )
    assert resp.status_code == 422


# ── list / get ────────────────────────────────────


def test_list_returns_jobs():
    orch, _, _ = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    orch.propose_job(
        model_id="m1", partition=partition,
    )
    body = _client(orchestrator=orch).get(
        "/admin/inference/pipeline/job",
    ).json()
    assert len(body["jobs"]) == 1


def test_get_job_404_unknown():
    orch, _, _ = _orch()
    resp = _client(orchestrator=orch).get(
        "/admin/inference/pipeline/job/nope",
    )
    assert resp.status_code == 404


def test_get_job_happy_path():
    orch, _, _ = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    body = _client(orchestrator=orch).get(
        f"/admin/inference/pipeline/job/{job.job_id}",
    ).json()
    assert body["job_id"] == job.job_id


# ── execute ───────────────────────────────────────


def test_execute_happy_path():
    """Execute via HTTP — v1 uses default stub stage
    runners since the API can't accept Python callables."""
    orch, _, pub = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    resp = _client(orchestrator=orch).post(
        f"/admin/inference/pipeline/job/{job.job_id}"
        f"/execute",
        json={
            "prompt_b64": base64.b64encode(
                b"hello world",
            ).decode(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["receipt"] is not None
    # Reconstruct receipt + verify with orchestrator pubkey
    from prsm.compute.inference.pipeline_receipt import (
        PipelineInferenceReceipt,
    )
    receipt = PipelineInferenceReceipt.from_dict(
        body["receipt"],
    )
    result = verify_pipeline_receipt(
        receipt, orchestrator_pubkey_b64=pub,
    )
    assert result.ok


def test_execute_422_missing_prompt():
    orch, _, _ = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    resp = _client(orchestrator=orch).post(
        f"/admin/inference/pipeline/job/{job.job_id}"
        f"/execute",
        json={},
    )
    assert resp.status_code == 422


def test_execute_404_unknown_job():
    orch, _, _ = _orch()
    resp = _client(orchestrator=orch).post(
        "/admin/inference/pipeline/job/nope/execute",
        json={"prompt_b64": base64.b64encode(b"x").decode()},
    )
    assert resp.status_code == 404


def test_execute_422_bad_base64():
    orch, _, _ = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    resp = _client(orchestrator=orch).post(
        f"/admin/inference/pipeline/job/{job.job_id}"
        f"/execute",
        json={"prompt_b64": "not-base64!"},
    )
    assert resp.status_code == 422


# ── round detail ───────────────────────────────────


def test_get_round_404_no_execution_yet():
    orch, _, _ = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    resp = _client(orchestrator=orch).get(
        f"/admin/inference/pipeline/job/{job.job_id}"
        f"/round",
    )
    assert resp.status_code == 404


def test_get_round_happy_path():
    orch, _, _ = _orch()
    partition = even_layer_partition(
        total_layers=4, node_ids=["n0", "n1"],
    )
    job = orch.propose_job(
        model_id="m1", partition=partition,
    )
    orch.execute(
        job.job_id, prompt=b"x",
        stage_runners=[
            deterministic_stub_stage_runner(),
            deterministic_stub_stage_runner(),
        ],
    )
    body = _client(orchestrator=orch).get(
        f"/admin/inference/pipeline/job/{job.job_id}"
        f"/round",
    ).json()
    assert body["status"] == "completed"
    assert body["receipt"]


# ── MCP ────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_pipeline_inference" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_pipeline_inference({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_pipeline_inference({
        "action": "explode",
    })
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_propose():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1", "model_id": "m1",
            "status": "proposed",
            "partition": {
                "total_layers": 4,
                "stage_layer_ranges": [[0, 1], [2, 3]],
                "stage_node_ids": ["n0", "n1"],
            },
            "created_at": 0.0,
        }),
    ) as mock_call:
        r = await handle_prsm_pipeline_inference({
            "action": "propose",
            "model_id": "m1",
            "total_layers": 4,
            "node_ids": ["n0", "n1"],
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/inference/pipeline/job"
    assert "j-1" in r


@pytest.mark.asyncio
async def test_mcp_execute():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1", "round_id": "r-1",
            "status": "completed",
            "receipt": {
                "version": "v1",
                "prompt_hash": "a" * 64,
                "output_hash": "b" * 64,
                "partition_hash": "c" * 64,
                "stage_receipts": [],
                "orchestrator_signature_b64": "sig",
            },
            "started_at": 0.0, "completed_at": 1.0,
            "error": None,
        }),
    ) as mock_call:
        r = await handle_prsm_pipeline_inference({
            "action": "execute",
            "job_id": "j-1",
            "prompt_b64": base64.b64encode(
                b"hello",
            ).decode(),
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/inference/pipeline/job/j-1/execute"
    )
    assert "completed" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "jobs": [{
                "job_id": "j-abcdefgh",
                "model_id": "m1",
                "status": "completed",
                "partition": {
                    "total_layers": 4,
                    "stage_layer_ranges": [
                        [0, 1], [2, 3],
                    ],
                    "stage_node_ids": ["n0", "n1"],
                },
                "created_at": 0.0,
            }],
        }),
    ):
        r = await handle_prsm_pipeline_inference({
            "action": "list",
        })
    assert "j-abcdef" in r
    assert "m1" in r


@pytest.mark.asyncio
async def test_mcp_lookup():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1", "model_id": "m1",
            "status": "completed",
            "partition": {
                "total_layers": 4,
                "stage_layer_ranges": [[0, 1], [2, 3]],
                "stage_node_ids": ["n0", "n1"],
            },
            "created_at": 0.0,
        }),
    ) as mock_call:
        r = await handle_prsm_pipeline_inference({
            "action": "lookup",
            "job_id": "j-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/inference/pipeline/job/j-1"
    assert "j-1" in r


@pytest.mark.asyncio
async def test_mcp_get_round():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "job_id": "j-1", "round_id": "r-1",
            "status": "completed",
            "receipt": {
                "version": "v1",
                "prompt_hash": "a" * 64,
                "output_hash": "b" * 64,
                "partition_hash": "c" * 64,
                "stage_receipts": [],
                "orchestrator_signature_b64": "sig",
            },
            "started_at": 0.0, "completed_at": 1.0,
            "error": None,
        }),
    ) as mock_call:
        r = await handle_prsm_pipeline_inference({
            "action": "get_round",
            "job_id": "j-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/inference/pipeline/job/j-1/round"
    )
    assert "completed" in r.lower()
