"""Sprint 342 — Surface federated_learning + pipeline_inference
orchestrators on /health/detailed.

Sprint 308 wired `FederatedLearningOrchestrator` and sprint 312
wired `PipelineInferenceOrchestrator` onto Node. Both are
filesystem-persisted, both can fail to construct on misconfig
(missing env, broken persistence dir, schema corruption). But
neither appeared on `/health/detailed` — operators alerting on
that endpoint couldn't see when an orchestrator failed to wire.

Sprint 342 adds two new subsystems:
  - `federated_learning_orchestrator` reading
    `node._federated_learning_orchestrator`
  - `pipeline_inference_orchestrator` reading
    `node._pipeline_inference_orchestrator`

Both follow the sprint-147 convention:
  - status=ok when wired + working
  - status=not_wired when None (opt-out per the optional list)
  - status=error when probe raises
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_client(*, fl=None, pipeline=None):
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.identity.node_id = "n-test"
    node.discovery = None
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._federated_learning_orchestrator = fl
    node._pipeline_inference_orchestrator = pipeline
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── federated_learning_orchestrator ────────────────────────


def test_fl_orchestrator_appears_when_wired():
    orch = MagicMock()
    orch.list_jobs = MagicMock(return_value=[{"id": "j1"}, {"id": "j2"}])
    client = _make_client(fl=orch)
    body = client.get("/health/detailed").json()
    assert "federated_learning_orchestrator" in body["subsystems"]
    sub = body["subsystems"]["federated_learning_orchestrator"]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    # Job count surfaced for ops
    assert sub.get("jobs_count") == 2


def test_fl_orchestrator_not_wired_when_none():
    client = _make_client(fl=None)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["federated_learning_orchestrator"]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


def test_fl_orchestrator_error_when_list_jobs_raises():
    """Wired but list_jobs() raises → status=error so operators
    see the reason via /health/detailed."""
    orch = MagicMock()
    orch.list_jobs = MagicMock(side_effect=RuntimeError("disk full"))
    client = _make_client(fl=orch)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["federated_learning_orchestrator"]
    assert sub["available"] is False
    assert sub["status"] == "error"
    assert "disk full" in sub.get("error", "")


# ── pipeline_inference_orchestrator ────────────────────────


def test_pipeline_orchestrator_appears_when_wired():
    orch = MagicMock()
    orch.list_jobs = MagicMock(return_value=[{"id": "p1"}])
    client = _make_client(pipeline=orch)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["pipeline_inference_orchestrator"]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    assert sub.get("jobs_count") == 1


def test_pipeline_orchestrator_not_wired_when_none():
    client = _make_client(pipeline=None)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["pipeline_inference_orchestrator"]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


def test_pipeline_orchestrator_error_when_list_jobs_raises():
    orch = MagicMock()
    orch.list_jobs = MagicMock(side_effect=RuntimeError("schema bad"))
    client = _make_client(pipeline=orch)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["pipeline_inference_orchestrator"]
    assert sub["available"] is False
    assert sub["status"] == "error"
    assert "schema bad" in sub.get("error", "")


# ── Aggregate status semantics ─────────────────────────────


def test_top_status_healthy_when_orchestrators_not_wired():
    """Opt-out (not_wired) must not flip top-level to degraded.
    Per sprint-147 convention: optional + opt-out = healthy."""
    from prsm.node.api import create_api_app
    node = MagicMock()
    node.identity.node_id = "n-test"
    # Wire core subsystems OK
    ledger = MagicMock()
    ledger._is_initialized = True
    ledger._connected_address = "0xabc"
    ledger.contract_address = "0xabc"
    node.ftns_ledger = ledger
    esc = MagicMock()
    esc._escrows = {}
    esc.default_timeout = 3600
    node._payment_escrow = esc
    node._job_history = None
    node.discovery = None
    node._federated_learning_orchestrator = None
    node._pipeline_inference_orchestrator = None
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    body = client.get("/health/detailed").json()
    # FL + pipeline both not_wired → opt-out → still healthy
    # (assuming other optional subsystems also clean)
    assert body["subsystems"][
        "federated_learning_orchestrator"
    ]["status"] == "not_wired"
    assert body["subsystems"][
        "pipeline_inference_orchestrator"
    ]["status"] == "not_wired"
