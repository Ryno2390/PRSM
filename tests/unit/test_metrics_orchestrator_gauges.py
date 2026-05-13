"""Sprint 345 — Prometheus /metrics gauges for the §7/§14
orchestrators + stores added in sprints 342/343.

Sprint 342 wired FL + pipeline-inference orchestrators onto
/health/detailed. Sprint 343 added 5 §14/§7 stores. Sprint
344 surfaced their counts on prsm_node_health. But the
canonical ops dashboard surface (/metrics for Grafana /
Prometheus / Datadog) was missing them — ops couldn't alert
on FL job backlog, pipeline backlog, content-filter rule
count, disclosure record count, incident count, $CORP issuer
count, or upgrade proposal count.

Sprint 345 closes the gap. Exposes 7 gauges (one per
orchestrator/store), fail-soft per the rest of /metrics:
each probe is wrapped so a single subsystem error omits its
gauge but doesn't 500 the endpoint.

  prsm_fl_jobs_count           — federated_learning_orchestrator
  prsm_pipeline_jobs_count     — pipeline_inference_orchestrator
  prsm_content_filter_count    — content_filter_store
  prsm_disclosure_count        — disclosure_intake
  prsm_incident_count          — incident_response
  prsm_corp_issuer_count       — corp_capability_store
  prsm_upgrade_count           — upgrade_orchestrator

When a subsystem is not_wired (attr is None or missing), its
gauge is omitted — Prometheus then sees the absence as
"feature disabled" rather than zero, which avoids misleading
"all zeros" dashboards on opt-out deploys.
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


@pytest.mark.parametrize("attr, method, value, metric", [
    ("_federated_learning_orchestrator", "list_jobs",
     [{}, {}, {}], "prsm_fl_jobs_count 3"),
    ("_pipeline_inference_orchestrator", "list_jobs",
     [{}], "prsm_pipeline_jobs_count 1"),
    ("_content_filter_store", "count",
     5, "prsm_content_filter_count 5"),
    ("_disclosure_intake", "count",
     12, "prsm_disclosure_count 12"),
    ("_incident_response", "count",
     0, "prsm_incident_count 0"),
    ("_upgrade_orchestrator", "count",
     7, "prsm_upgrade_count 7"),
])
def test_gauge_emitted_when_wired(attr, method, value, metric):
    store = MagicMock()
    if method == "list_jobs":
        store.list_jobs = MagicMock(return_value=value)
    else:
        getattr(store, method).return_value = value
    client = _make_client(**{attr: store})
    body = client.get("/metrics").text
    assert metric in body, (
        f"expected '{metric}' in metrics output; got:\n{body[:500]}"
    )


def test_corp_capability_gauge_uses_issuer_count():
    store = MagicMock()
    store.list_issuers = MagicMock(return_value=["a", "b"])
    client = _make_client(_corp_capability_store=store)
    body = client.get("/metrics").text
    assert "prsm_corp_issuer_count 2" in body


@pytest.mark.parametrize("attr, metric_name", [
    ("_federated_learning_orchestrator", "prsm_fl_jobs_count"),
    ("_pipeline_inference_orchestrator", "prsm_pipeline_jobs_count"),
    ("_content_filter_store", "prsm_content_filter_count"),
    ("_disclosure_intake", "prsm_disclosure_count"),
    ("_incident_response", "prsm_incident_count"),
    ("_corp_capability_store", "prsm_corp_issuer_count"),
    ("_upgrade_orchestrator", "prsm_upgrade_count"),
])
def test_gauge_omitted_when_not_wired(attr, metric_name):
    """When subsystem is None, gauge is OMITTED (not zero).
    Prometheus reads absence as 'feature disabled' which is
    semantically clearer than 'wired but reporting zero'."""
    client = _make_client()  # all None by default
    body = client.get("/metrics").text
    assert metric_name not in body
    # /metrics still returns 200 + canonical node_up gauge
    assert "prsm_node_up 1" in body


@pytest.mark.parametrize("attr, method", [
    ("_federated_learning_orchestrator", "list_jobs"),
    ("_content_filter_store", "count"),
    ("_upgrade_orchestrator", "count"),
    ("_corp_capability_store", "list_issuers"),
])
def test_metrics_endpoint_fail_soft_when_probe_raises(
    attr, method,
):
    """A subsystem whose probe raises must NOT 500 /metrics —
    the gauge is omitted, other metrics still emit."""
    store = MagicMock()
    getattr(store, method).side_effect = RuntimeError("disk full")
    client = _make_client(**{attr: store})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    # node_up still present
    assert "prsm_node_up 1" in resp.text
