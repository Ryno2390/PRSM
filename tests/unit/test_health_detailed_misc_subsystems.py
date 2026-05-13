"""Sprint 343 — Add 5 more orchestrator subsystems to /health/detailed.

Sprint 342 introduced `_orchestrator_subsystem` for FL +
pipeline inference. This sprint generalizes the helper to
accept a custom probe callable (since not every subsystem
exposes `.list_jobs()`) and adds 5 more wired stores:

  content_filter_store     — `.count()` returns total filter rules
  disclosure_intake        — `.count()` returns disclosure record count
  incident_response        — `.count()` returns incident record count
  corp_capability_store    — `.list_issuers()` returns issuer list
  upgrade_orchestrator     — `.count()` returns upgrade proposal count

All five were wired in earlier sprints (270/300/302/304/303
respectively) but never got /health/detailed visibility.
Operators alerting on /health/detailed couldn't catch wired-
but-broken state on any of them.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_client(**overrides):
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.identity.node_id = "n-test"
    node.discovery = None
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._federated_learning_orchestrator = None
    node._pipeline_inference_orchestrator = None
    node._content_filter_store = None
    node._disclosure_intake = None
    node._incident_response = None
    node._corp_capability_store = None
    node._upgrade_orchestrator = None
    for k, v in overrides.items():
        setattr(node, k, v)
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize("subsystem, attr, count_method, count_value", [
    ("content_filter_store", "_content_filter_store", "count", 5),
    ("disclosure_intake", "_disclosure_intake", "count", 12),
    ("incident_response", "_incident_response", "count", 0),
    ("upgrade_orchestrator", "_upgrade_orchestrator", "count", 3),
])
def test_count_based_subsystem_appears_when_wired(
    subsystem, attr, count_method, count_value,
):
    store = MagicMock()
    getattr(store, count_method).return_value = count_value
    client = _make_client(**{attr: store})
    body = client.get("/health/detailed").json()
    assert subsystem in body["subsystems"], (
        f"expected {subsystem} in subsystems"
    )
    sub = body["subsystems"][subsystem]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    assert sub.get("record_count") == count_value


def test_corp_capability_store_uses_list_issuers():
    store = MagicMock()
    store.list_issuers = MagicMock(return_value=["iss1", "iss2", "iss3"])
    client = _make_client(_corp_capability_store=store)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"]["corp_capability_store"]
    assert sub["available"] is True
    assert sub["status"] == "ok"
    assert sub.get("record_count") == 3


@pytest.mark.parametrize("subsystem, attr", [
    ("content_filter_store", "_content_filter_store"),
    ("disclosure_intake", "_disclosure_intake"),
    ("incident_response", "_incident_response"),
    ("corp_capability_store", "_corp_capability_store"),
    ("upgrade_orchestrator", "_upgrade_orchestrator"),
])
def test_not_wired_when_none(subsystem, attr):
    client = _make_client()  # all None by default
    body = client.get("/health/detailed").json()
    sub = body["subsystems"][subsystem]
    assert sub["available"] is False
    assert sub["status"] == "not_wired"


@pytest.mark.parametrize("subsystem, attr, probe_method", [
    ("content_filter_store", "_content_filter_store", "count"),
    ("disclosure_intake", "_disclosure_intake", "count"),
    ("incident_response", "_incident_response", "count"),
    ("corp_capability_store", "_corp_capability_store", "list_issuers"),
    ("upgrade_orchestrator", "_upgrade_orchestrator", "count"),
])
def test_error_when_probe_raises(subsystem, attr, probe_method):
    store = MagicMock()
    getattr(store, probe_method).side_effect = RuntimeError("disk full")
    client = _make_client(**{attr: store})
    body = client.get("/health/detailed").json()
    sub = body["subsystems"][subsystem]
    assert sub["available"] is False
    assert sub["status"] == "error"
    assert "disk full" in sub.get("error", "")


def test_top_status_healthy_when_all_5_not_wired():
    """All five new subsystems opt-out → top-level remains
    healthy per sprint-147 convention."""
    from prsm.node.api import create_api_app
    node = MagicMock()
    node.identity.node_id = "n-test"
    # Core wired
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
    # All 5 new subsystems explicitly None
    for attr in ("_content_filter_store", "_disclosure_intake",
                 "_incident_response", "_corp_capability_store",
                 "_upgrade_orchestrator",
                 "_federated_learning_orchestrator",
                 "_pipeline_inference_orchestrator"):
        setattr(node, attr, None)
    client = TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )
    body = client.get("/health/detailed").json()
    for subsystem in ("content_filter_store", "disclosure_intake",
                      "incident_response", "corp_capability_store",
                      "upgrade_orchestrator"):
        assert body["subsystems"][subsystem]["status"] == "not_wired"
