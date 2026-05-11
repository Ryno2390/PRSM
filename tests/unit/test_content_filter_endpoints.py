"""Sprint 269 — content-filter admin CRUD endpoints + /content/
retrieve enforcement."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_filter_store import ContentFilterStore


def _client(filter_store=None, content_provider=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._content_filter_store = filter_store
    node.content_provider = content_provider
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── GET /admin/content-filter ────────────────────────────


def test_get_filter_503_when_unwired():
    resp = _client(None).get("/admin/content-filter")
    assert resp.status_code == 503


def test_get_filter_empty():
    s = ContentFilterStore()
    resp = _client(s).get("/admin/content-filter")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count_cids"] == 0
    assert body["action_on_match"] == "refuse"


def test_get_filter_populated():
    s = ContentFilterStore()
    s.add_cids(["bafy1", "bafy2"])
    s.add_tags(["safety"])
    resp = _client(s).get("/admin/content-filter")
    body = resp.json()
    assert body["count_cids"] == 2
    assert body["count_tags"] == 1


# ── POST /admin/content-filter/cids ──────────────────────


def test_add_cids_happy_path():
    s = ContentFilterStore()
    resp = _client(s).post(
        "/admin/content-filter/cids",
        json={"cids": ["bafy1", "bafy2", "bafy3"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == 3
    assert body["total"] == 3


def test_add_cids_idempotent():
    s = ContentFilterStore()
    s.add_cids(["bafy1"])
    resp = _client(s).post(
        "/admin/content-filter/cids",
        json={"cids": ["bafy1", "bafy2"]},  # bafy1 already there
    )
    body = resp.json()
    assert body["added"] == 1
    assert body["total"] == 2


def test_add_cids_422_when_not_list():
    resp = _client(ContentFilterStore()).post(
        "/admin/content-filter/cids",
        json={"cids": "bafy1"},  # not a list
    )
    assert resp.status_code == 422


def test_add_cids_422_when_too_many():
    resp = _client(ContentFilterStore()).post(
        "/admin/content-filter/cids",
        json={"cids": [f"c-{i}" for i in range(1001)]},
    )
    assert resp.status_code == 422


def test_add_cids_503_when_store_unwired():
    resp = _client(None).post(
        "/admin/content-filter/cids",
        json={"cids": ["bafy1"]},
    )
    assert resp.status_code == 503


# ── DELETE /admin/content-filter/cids/{cid} ──────────────


def test_remove_cid_happy_path():
    s = ContentFilterStore()
    s.add_cids(["bafy1", "bafy2"])
    resp = _client(s).delete("/admin/content-filter/cids/bafy1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["removed"] == "bafy1"
    assert body["total"] == 1


def test_remove_cid_404_when_missing():
    s = ContentFilterStore()
    resp = _client(s).delete(
        "/admin/content-filter/cids/never-existed",
    )
    assert resp.status_code == 404


# ── POST /admin/content-filter/tags ──────────────────────


def test_add_tags_happy_path():
    s = ContentFilterStore()
    resp = _client(s).post(
        "/admin/content-filter/tags",
        json={"tags": ["Safety-Flagged"]},
    )
    body = resp.json()
    assert body["added"] == 1
    assert body["total"] == 1
    # Verify lowercased on store
    assert s.to_dict()["blocked_model_tags"] == ["safety-flagged"]


# ── POST /admin/content-filter/action ────────────────────


def test_set_action_happy_path():
    s = ContentFilterStore()
    resp = _client(s).post(
        "/admin/content-filter/action",
        json={"action": "log_and_refuse"},
    )
    body = resp.json()
    assert body["action_on_match"] == "log_and_refuse"
    assert s.to_dict()["action_on_match"] == "log_and_refuse"


def test_set_action_422_on_invalid_action():
    resp = _client(ContentFilterStore()).post(
        "/admin/content-filter/action",
        json={"action": "explode"},
    )
    assert resp.status_code == 422


# ── /content/retrieve enforcement ────────────────────────


def test_retrieve_refuses_blocked_cid():
    """A CID in the operator's blocklist gets 451 Unavailable
    For Legal Reasons (RFC 7725) BEFORE any compute cost or
    network fetch."""
    s = ContentFilterStore()
    s.add_cids(["bafy-blocked"])
    cp = MagicMock()
    cp.get_stats = MagicMock(return_value={"providers_tried": 0})
    cp.request_content = AsyncMock(return_value=b"unreachable")
    resp = _client(s, cp).get("/content/retrieve/bafy-blocked")
    assert resp.status_code == 451
    assert "blocked" in resp.json()["detail"]
    # CRITICAL: content_provider.request_content was NEVER called
    cp.request_content.assert_not_called()


def test_retrieve_allows_unblocked_cid():
    """A CID NOT in the blocklist passes through to the
    provider as usual."""
    s = ContentFilterStore()
    s.add_cids(["bafy-other"])
    cp = MagicMock()
    cp.get_stats = MagicMock(return_value={"providers_tried": 0})
    cp.request_content = AsyncMock(return_value=None)  # not found
    resp = _client(s, cp).get("/content/retrieve/bafy-allowed")
    # Not 451 (filter didn't block); reaches the provider
    assert resp.status_code != 451
    cp.request_content.assert_awaited_once()


def test_retrieve_passes_through_when_filter_unwired():
    """If filter store isn't initialized, retrieve behaves
    as pre-269 — no filter check."""
    cp = MagicMock()
    cp.get_stats = MagicMock(return_value={"providers_tried": 0})
    cp.request_content = AsyncMock(return_value=None)
    resp = _client(None, cp).get("/content/retrieve/bafy-any")
    assert resp.status_code != 451
    cp.request_content.assert_awaited_once()
