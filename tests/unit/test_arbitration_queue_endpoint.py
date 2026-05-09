"""GET /content/arbitration/queue — operator-side view of
pending content-attribution disputes.

Closes the operator-UX gap from PRSM-PROV-1 Item 6 (shipped
2026-05-08): the FilesystemArbitrationQueue persists disputes
at ``~/.prsm/arbitration_queue/`` but operators have no way
to see them without scanning the directory by hand.

Backs the ``prsm_arbitration_status`` MCP tool.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.data.dedup.arbitration import DisputedAttributionRecord


def _record(*, new_cid="bafyaa", parent_cid="bafyaa-parent",
            similarity=0.85, kind="image", flagged_at=1_700_000_000,
            proposal_id=None):
    return DisputedAttributionRecord(
        new_cid=new_cid,
        new_creator="0x" + "11" * 20,
        candidate_parent_cid=parent_cid,
        candidate_parent_creator="0x" + "22" * 20,
        similarity=similarity,
        fingerprint_kind=kind,
        flagged_at=flagged_at,
        proposal_id=proposal_id,
    )


def _node(*, arbitration_queue=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node._arbitration_queue = arbitration_queue
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestArbitrationQueueAvailability:
    def test_503_when_queue_not_wired(self):
        node = _node()
        resp = _client(node).get("/content/arbitration/queue")
        assert resp.status_code == 503


class TestArbitrationQueueHappyPath:
    def test_empty_queue_returns_empty_list(self):
        queue = MagicMock()
        queue.list_pending = AsyncMock(return_value=[])
        node = _node(arbitration_queue=queue)
        resp = _client(node).get("/content/arbitration/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pending"] == []
        assert body["total"] == 0

    def test_pending_records_serialized(self):
        records = [
            _record(new_cid="bafy-a", similarity=0.82),
            _record(new_cid="bafy-b", similarity=0.91, kind="audio"),
        ]
        queue = MagicMock()
        queue.list_pending = AsyncMock(return_value=records)
        node = _node(arbitration_queue=queue)
        resp = _client(node).get("/content/arbitration/queue")
        body = resp.json()
        assert body["total"] == 2
        cids = {r["new_cid"] for r in body["pending"]}
        assert cids == {"bafy-a", "bafy-b"}

    def test_record_includes_load_bearing_fields(self):
        rec = _record(
            new_cid="bafy-x", parent_cid="bafy-y",
            similarity=0.88, kind="text",
            flagged_at=1_700_000_500, proposal_id="prop-7",
        )
        queue = MagicMock()
        queue.list_pending = AsyncMock(return_value=[rec])
        node = _node(arbitration_queue=queue)
        resp = _client(node).get("/content/arbitration/queue")
        entry = resp.json()["pending"][0]
        assert entry["new_cid"] == "bafy-x"
        assert entry["candidate_parent_cid"] == "bafy-y"
        assert entry["similarity"] == 0.88
        assert entry["fingerprint_kind"] == "text"
        assert entry["flagged_at"] == 1_700_000_500
        assert entry["proposal_id"] == "prop-7"


class TestArbitrationQueueFailure:
    def test_502_when_list_pending_raises(self):
        queue = MagicMock()
        queue.list_pending = AsyncMock(
            side_effect=RuntimeError("disk read error"),
        )
        node = _node(arbitration_queue=queue)
        resp = _client(node).get("/content/arbitration/queue")
        assert resp.status_code == 502
