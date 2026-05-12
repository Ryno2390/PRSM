"""Sprint 287 — creator reputation HTTP + MCP surface.

Operator-facing read surface for the CreatorReputationTracker:
  GET /marketplace/creator-reputation             paginated list
  GET /marketplace/creator-reputation/{creator_id} single detail
  POST /marketplace/creator-reputation/access    record an access

The write endpoint (POST /access) is operator-internal: it's
called when ContentStore.retrieve_with_artifacts resolves a
piece of content (operator records the access for THEIR
local view). Per the per-node-not-federated contract (no
gossip), each operator builds their own creator reputation
view from their own observed traffic.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.marketplace.creator_reputation import (
    CreatorReputationTracker,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_creator_reputation,
)
from prsm.node.api import create_api_app


def _client(tracker=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._creator_reputation_tracker = tracker
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _seed_busy(tracker, creator_id="alice", n=20):
    """Push above MIN_SAMPLES_FOR_SCORE."""
    for i in range(n):
        tracker.record_access(
            creator_id=creator_id,
            purchaser_id=f"p{i}",
            content_id=f"c{i % 5}",
        )


# ── GET /marketplace/creator-reputation ──────────────────


def test_list_503_when_unwired():
    resp = _client(None).get("/marketplace/creator-reputation")
    assert resp.status_code == 503


def test_list_empty_tracker():
    t = CreatorReputationTracker()
    resp = _client(t).get("/marketplace/creator-reputation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["creators"] == []
    assert body["count"] == 0


def test_list_populated_sorted_by_score_desc():
    t = CreatorReputationTracker()
    # alice: high reach, all repeats
    for i in range(50):
        t.record_access(
            creator_id="alice", purchaser_id=f"a{i}",
            content_id="c0",
        )
        t.record_access(
            creator_id="alice", purchaser_id=f"a{i}",
            content_id="c1",
        )
    # bob: high reach, zero repeats (spam pattern)
    for i in range(50):
        t.record_access(
            creator_id="bob", purchaser_id=f"b{i}",
            content_id=f"c{i}",
        )
    resp = _client(t).get("/marketplace/creator-reputation")
    body = resp.json()
    ids = [c["creator_id"] for c in body["creators"]]
    # alice > bob (alice has repeats; bob doesn't)
    assert ids[0] == "alice"
    assert ids[1] == "bob"
    assert (
        body["creators"][0]["score"]
        > body["creators"][1]["score"]
    )


def test_list_invalid_limit_422():
    t = CreatorReputationTracker()
    resp = _client(t).get(
        "/marketplace/creator-reputation?limit=0",
    )
    assert resp.status_code == 422


def test_list_limit_caps_results():
    t = CreatorReputationTracker()
    for who in ["a", "b", "c", "d"]:
        _seed_busy(t, creator_id=who, n=12)
    resp = _client(t).get(
        "/marketplace/creator-reputation?limit=2",
    )
    body = resp.json()
    assert len(body["creators"]) == 2
    assert body["count"] == 4


# ── GET /marketplace/creator-reputation/{creator_id} ─────


def test_get_one_503_when_unwired():
    resp = _client(None).get(
        "/marketplace/creator-reputation/alice",
    )
    assert resp.status_code == 503


def test_get_one_unknown_returns_neutral():
    t = CreatorReputationTracker()
    resp = _client(t).get(
        "/marketplace/creator-reputation/nobody",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["creator_id"] == "nobody"
    assert body["known"] is False
    assert body["score"] == 0.5  # NEUTRAL_SCORE


def test_get_one_known_returns_detail():
    t = CreatorReputationTracker()
    _seed_busy(t, creator_id="alice", n=15)
    resp = _client(t).get(
        "/marketplace/creator-reputation/alice",
    )
    body = resp.json()
    assert body["creator_id"] == "alice"
    assert body["known"] is True
    assert body["total_accesses"] == 15
    assert body["distinct_purchasers"] == 15
    assert body["first_seen_unix"] > 0
    assert body["last_seen_unix"] > 0


# ── POST /marketplace/creator-reputation/access ──────────


def test_record_access_503_when_unwired():
    resp = _client(None).post(
        "/marketplace/creator-reputation/access",
        json={
            "creator_id": "alice",
            "purchaser_id": "bob",
            "content_id": "c1",
        },
    )
    assert resp.status_code == 503


def test_record_access_happy_path():
    t = CreatorReputationTracker()
    resp = _client(t).post(
        "/marketplace/creator-reputation/access",
        json={
            "creator_id": "alice",
            "purchaser_id": "bob",
            "content_id": "c1",
        },
    )
    assert resp.status_code == 200
    assert t.access_count("alice") == 1


def test_record_access_422_missing_field():
    t = CreatorReputationTracker()
    resp = _client(t).post(
        "/marketplace/creator-reputation/access",
        json={"creator_id": "alice"},
    )
    assert resp.status_code == 422


def test_record_access_422_empty_field():
    t = CreatorReputationTracker()
    resp = _client(t).post(
        "/marketplace/creator-reputation/access",
        json={
            "creator_id": "",
            "purchaser_id": "bob",
            "content_id": "c1",
        },
    )
    assert resp.status_code == 422


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_creator_reputation" in TOOL_HANDLERS


class TestMcpValidation:
    @pytest.mark.asyncio
    async def test_missing_action(self):
        r = await handle_prsm_creator_reputation({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        r = await handle_prsm_creator_reputation(
            {"action": "explode"},
        )
        assert "must be" in r.lower()


class TestMcpList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creators": [], "count": 0, "limit": 100,
            }),
        ) as mock_call:
            r = await handle_prsm_creator_reputation(
                {"action": "list"},
            )
        args = mock_call.await_args[0]
        assert args[0] == "GET"
        assert args[1].startswith("/marketplace/creator-reputation")
        assert "0" in r

    @pytest.mark.asyncio
    async def test_list_renders_rows(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creators": [{
                    "creator_id": "alice",
                    "score": 0.85,
                    "total_accesses": 100,
                    "distinct_purchasers": 50,
                    "repeat_purchaser_count": 30,
                    "first_seen_unix": 100,
                    "last_seen_unix": 200,
                }],
                "count": 1, "limit": 100,
            }),
        ):
            r = await handle_prsm_creator_reputation(
                {"action": "list"},
            )
        assert "alice" in r
        assert "0.850" in r or "0.85" in r


class TestMcpLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_creator_id(self):
        r = await handle_prsm_creator_reputation(
            {"action": "lookup"},
        )
        assert "creator_id" in r

    @pytest.mark.asyncio
    async def test_lookup_renders_detail(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creator_id": "alice",
                "known": True,
                "score": 0.85,
                "total_accesses": 100,
                "distinct_purchasers": 50,
                "repeat_purchaser_count": 30,
                "first_seen_unix": 100,
                "last_seen_unix": 200,
            }),
        ) as mock_call:
            r = await handle_prsm_creator_reputation(
                {"action": "lookup", "creator_id": "alice"},
            )
        args = mock_call.await_args[0]
        assert args[1] == (
            "/marketplace/creator-reputation/alice"
        )
        assert "alice" in r
        assert "100" in r
        assert "50" in r

    @pytest.mark.asyncio
    async def test_lookup_unknown_cold_start_marker(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "creator_id": "nobody",
                "known": False,
                "score": 0.5,
                "total_accesses": 0,
                "distinct_purchasers": 0,
                "repeat_purchaser_count": 0,
                "first_seen_unix": 0,
                "last_seen_unix": 0,
            }),
        ):
            r = await handle_prsm_creator_reputation({
                "action": "lookup", "creator_id": "nobody",
            })
        assert "cold-start" in r.lower() or "0.5" in r
