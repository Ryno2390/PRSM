"""Sprint 272 — Foundation takedown-notice intake endpoints.

Per Vision §14 / R9-SCOPING-1 §8 separation: Foundation
records notices (information distribution), operators
voluntarily act on them via sprint-269 ContentFilterStore.
These endpoints provide the read/write surface for the
notice ring.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.takedown_notice_log import TakedownNoticeRing


def _client(ring=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._takedown_notice_ring = ring
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── POST /admin/takedown-notice ──────────────────────────


def test_record_503_when_unwired():
    resp = _client(None).post(
        "/admin/takedown-notice",
        json={
            "target_cid": "bafy1", "sender": "x@y.z",
            "jurisdiction": "US-DMCA", "basis": "DMCA §512(c)",
        },
    )
    assert resp.status_code == 503


def test_record_happy_path():
    r = TakedownNoticeRing()
    resp = _client(r).post(
        "/admin/takedown-notice",
        json={
            "target_cid": "bafy1", "sender": "legal@ex.com",
            "jurisdiction": "US-DMCA", "basis": "DMCA §512(c)",
            "notice_text": "full body",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_cid"] == "bafy1"
    assert body["status"] == "received"
    assert body["notice_id"]
    assert r.count() == 1


def test_record_422_missing_required_field():
    r = TakedownNoticeRing()
    resp = _client(r).post(
        "/admin/takedown-notice",
        json={
            "target_cid": "bafy1", "sender": "",
            "jurisdiction": "US-DMCA", "basis": "DMCA §512(c)",
        },
    )
    assert resp.status_code == 422
    assert "missing required field" in resp.json()["detail"].lower()


def test_record_422_all_fields_missing():
    r = TakedownNoticeRing()
    resp = _client(r).post("/admin/takedown-notice", json={})
    assert resp.status_code == 422


def test_record_caps_oversized_notice_text():
    r = TakedownNoticeRing()
    huge = "x" * 100_000
    resp = _client(r).post(
        "/admin/takedown-notice",
        json={
            "target_cid": "bafy1", "sender": "x",
            "jurisdiction": "y", "basis": "z",
            "notice_text": huge,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["notice_text"]) == 8192


# ── GET /admin/takedown-notices ──────────────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/admin/takedown-notices")
    assert resp.status_code == 503


def test_list_empty():
    r = TakedownNoticeRing()
    resp = _client(r).get("/admin/takedown-notices")
    assert resp.status_code == 200
    body = resp.json()
    assert body["notices"] == []
    assert body["total"] == 0


def test_list_populated_newest_first():
    r = TakedownNoticeRing()
    e1 = r.record(
        target_cid="c1", sender="s", jurisdiction="j", basis="b",
        timestamp=100.0,
    )
    e2 = r.record(
        target_cid="c2", sender="s", jurisdiction="j", basis="b",
        timestamp=200.0,
    )
    resp = _client(r).get("/admin/takedown-notices")
    body = resp.json()
    assert body["total"] == 2
    ids = [n["notice_id"] for n in body["notices"]]
    assert ids == [e2.notice_id, e1.notice_id]


def test_list_target_cid_filter():
    r = TakedownNoticeRing()
    r.record(
        target_cid="target-a", sender="s",
        jurisdiction="j", basis="b",
    )
    r.record(
        target_cid="target-b", sender="s",
        jurisdiction="j", basis="b",
    )
    resp = _client(r).get(
        "/admin/takedown-notices?target_cid=target-a",
    )
    body = resp.json()
    assert len(body["notices"]) == 1
    assert body["notices"][0]["target_cid"] == "target-a"


def test_list_invalid_limit_422():
    r = TakedownNoticeRing()
    resp = _client(r).get("/admin/takedown-notices?limit=0")
    assert resp.status_code == 422
    resp = _client(r).get("/admin/takedown-notices?limit=1001")
    assert resp.status_code == 422


def test_list_invalid_offset_422():
    r = TakedownNoticeRing()
    resp = _client(r).get("/admin/takedown-notices?offset=-1")
    assert resp.status_code == 422


def test_list_invalid_status_422():
    r = TakedownNoticeRing()
    resp = _client(r).get("/admin/takedown-notices?status=bogus")
    assert resp.status_code == 422


# ── GET /admin/takedown-notices/{notice_id} ──────────────


def test_get_one_503_when_unwired():
    resp = _client(None).get("/admin/takedown-notices/abc")
    assert resp.status_code == 503


def test_get_one_happy_path():
    r = TakedownNoticeRing()
    e = r.record(
        target_cid="bafy1", sender="s", jurisdiction="j", basis="b",
    )
    resp = _client(r).get(f"/admin/takedown-notices/{e.notice_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["notice_id"] == e.notice_id
    assert body["target_cid"] == "bafy1"


def test_get_one_404_when_missing():
    r = TakedownNoticeRing()
    resp = _client(r).get("/admin/takedown-notices/nonexistent")
    assert resp.status_code == 404
