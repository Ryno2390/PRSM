"""Sprint 275 — operator-side marketplace reputation visibility.

ReputationTracker has been quietly informing the query
orchestrator's marketplace candidate pool since Phase 3 Task 6,
but operators had no surface for inspecting it. These endpoints
close the observability gap so operators running their node
can see which providers are being trusted (or excluded by slash
history) and verify the marketplace dispatcher is making
defensible choices.

GET  /marketplace/reputation              — list all
GET  /marketplace/reputation/{prov_id}    — single detail
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.marketplace.reputation import ReputationTracker
from prsm.node.api import create_api_app


def _client(tracker=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.reputation_tracker = tracker
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _seed(tracker, provider_id, n_success=15, n_fail=0):
    """Push at least MIN_SAMPLES_FOR_SCORE (10) so score != NEUTRAL."""
    for _ in range(n_success):
        tracker.record_success(provider_id, latency_ms=120.0)
    for _ in range(n_fail):
        tracker.record_failure(provider_id)


# ── GET /marketplace/reputation ──────────────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/marketplace/reputation")
    assert resp.status_code == 503


def test_list_empty_tracker():
    t = ReputationTracker()
    resp = _client(t).get("/marketplace/reputation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["providers"] == []
    assert body["count"] == 0


def test_list_populated_sorted_by_score_desc():
    t = ReputationTracker()
    _seed(t, "good-provider", n_success=20, n_fail=0)
    _seed(t, "bad-provider", n_success=2, n_fail=20)
    resp = _client(t).get("/marketplace/reputation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    ids = [p["provider_id"] for p in body["providers"]]
    # Sorted by score desc: good > bad
    assert ids[0] == "good-provider"
    assert ids[1] == "bad-provider"
    # Each row has score + counts + latency
    assert body["providers"][0]["score"] > 0.9
    assert body["providers"][0]["successes"] == 20
    assert body["providers"][0]["failures"] == 0
    assert body["providers"][0]["latency_p50_ms"] is not None


def test_list_includes_slash_count():
    t = ReputationTracker()
    _seed(t, "slashed-provider", n_success=20, n_fail=0)
    t.record_slash(
        "slashed-provider", batch_id="batch1",
        slash_amount_wei=1000, reason="DOUBLE_SPEND",
    )
    resp = _client(t).get("/marketplace/reputation")
    body = resp.json()
    p = body["providers"][0]
    assert p["slashed_count"] == 1
    assert p["has_been_slashed"] is True


def test_list_limit_validation():
    t = ReputationTracker()
    resp = _client(t).get("/marketplace/reputation?limit=0")
    assert resp.status_code == 422
    resp = _client(t).get("/marketplace/reputation?limit=10001")
    assert resp.status_code == 422


def test_list_limit_caps_results():
    t = ReputationTracker()
    for i in range(5):
        _seed(t, f"p{i}", n_success=20, n_fail=0)
    resp = _client(t).get("/marketplace/reputation?limit=2")
    body = resp.json()
    assert len(body["providers"]) == 2
    assert body["count"] == 5  # total unaffected by limit


# ── GET /marketplace/reputation/{provider_id} ────────────


def test_get_one_503_when_unwired():
    resp = _client(None).get("/marketplace/reputation/abc")
    assert resp.status_code == 503


def test_get_one_unknown_provider_returns_neutral():
    """ReputationTracker returns NEUTRAL_SCORE for unknown
    provider — the endpoint mirrors that contract so callers
    can see the cold-start behavior reliably."""
    t = ReputationTracker()
    resp = _client(t).get("/marketplace/reputation/unknown")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider_id"] == "unknown"
    assert body["score"] == 0.5  # NEUTRAL_SCORE
    assert body["known"] is False
    assert body["successes"] == 0
    assert body["failures"] == 0
    assert body["slash_events"] == []


def test_get_one_known_provider_full_detail():
    t = ReputationTracker()
    _seed(t, "p1", n_success=20, n_fail=2)
    t.record_preemption("p1")
    t.record_slash(
        "p1", batch_id="batch1",
        slash_amount_wei=5000, reason="INVALID_SIGNATURE",
        tx_hash="0xdeadbeef",
    )
    resp = _client(t).get("/marketplace/reputation/p1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider_id"] == "p1"
    assert body["known"] is True
    assert body["successes"] == 20
    assert body["failures"] == 2
    assert body["preempted"] == 1
    assert body["slashed_count"] == 1
    assert body["has_been_slashed"] is True
    assert len(body["slash_events"]) == 1
    slash = body["slash_events"][0]
    assert slash["batch_id"] == "batch1"
    assert slash["reason"] == "INVALID_SIGNATURE"
    assert slash["slash_amount_wei"] == 5000
    assert slash["tx_hash"] == "0xdeadbeef"
    assert body["latency_p50_ms"] is not None
    assert body["latency_p95_ms"] is not None
    assert body["first_seen_unix"] > 0
    assert body["last_seen_unix"] > 0
