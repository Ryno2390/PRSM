"""GET /content/mine + prsm_my_content MCP.

Content publishers need a way to see what they've uploaded.
Surface ContentUploader.uploaded_content dict via paginated
endpoint. Each entry includes royalty_rate, access_count,
total_royalties, provenance_tx_hash so publishers can verify
on-chain registration + accruals.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.content_uploader import UploadedContent
from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_my_content,
)


def _node(*, uploaded=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None

    if uploaded is None:
        node.content_uploader = None
    else:
        cu = MagicMock()
        cu.uploaded_content = uploaded
        node.content_uploader = cu
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _make_record(cid, **overrides):
    defaults = {
        "content_id": cid,
        "filename": f"{cid}.dat",
        "size_bytes": 1000,
        "content_hash": "0xCAFEBABE",
        "creator_id": "test-creator",
        "royalty_rate": 0.05,
        "access_count": 0,
        "total_royalties": 0.0,
    }
    defaults.update(overrides)
    return UploadedContent(**defaults)


# ── Endpoint ───────────────────────────────────────────────────


class TestEndpoint:
    def test_503_when_uploader_not_wired(self):
        resp = _client(_node()).get("/content/mine")
        assert resp.status_code == 503

    def test_returns_uploaded_content_list(self):
        records = {
            "cid1": _make_record("cid1", filename="a.txt"),
            "cid2": _make_record("cid2", filename="b.txt"),
        }
        resp = _client(_node(uploaded=records)).get("/content/mine")
        body = resp.json()
        assert body["total"] == 2
        assert len(body["entries"]) == 2
        # Most-recent-first by created_at
        cids = [e["content_id"] for e in body["entries"]]
        assert set(cids) == {"cid1", "cid2"}

    def test_includes_royalty_fields(self):
        records = {
            "cid1": _make_record(
                "cid1",
                royalty_rate=0.10,
                access_count=42,
                total_royalties=1.5,
                provenance_tx_hash="0xPROVTX",
            ),
        }
        resp = _client(_node(uploaded=records)).get("/content/mine")
        entry = resp.json()["entries"][0]
        assert entry["royalty_rate"] == 0.10
        assert entry["access_count"] == 42
        assert entry["total_royalties"] == 1.5
        assert entry["provenance_tx_hash"] == "0xPROVTX"

    def test_pagination(self):
        records = {}
        for i in range(5):
            records[f"cid{i}"] = _make_record(
                f"cid{i}",
                # stagger created_at so ordering is determinate
                created_at=1700000000 + i,
            )
        resp = _client(_node(uploaded=records)).get(
            "/content/mine?limit=2&offset=1"
        )
        body = resp.json()
        assert body["total"] == 5
        assert len(body["entries"]) == 2
        # Most-recent-first; offset=1 skips most-recent (cid4)
        assert body["entries"][0]["content_id"] == "cid3"

    def test_invalid_limit_returns_422(self):
        resp = _client(_node(uploaded={})).get("/content/mine?limit=0")
        assert resp.status_code == 422

    def test_empty_returns_zero_total(self):
        resp = _client(_node(uploaded={})).get("/content/mine")
        body = resp.json()
        assert body["total"] == 0
        assert body["entries"] == []


# ── MCP ───────────────────────────────────────────────────────


class TestMcp:
    def test_handler_registered(self):
        assert "prsm_my_content" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_my_content" in names

    @pytest.mark.asyncio
    async def test_renders_content_list(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [
                    {
                        "content_id": "cid1",
                        "filename": "a.txt",
                        "size_bytes": 1024,
                        "royalty_rate": 0.05,
                        "access_count": 12,
                        "total_royalties": 0.6,
                        "provenance_tx_hash": "0xPROV",
                        "created_at": 1700000000,
                    },
                ],
                "total": 1, "offset": 0, "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_my_content({})
        assert "cid1" in result
        assert "a.txt" in result
        assert "12" in result  # access_count
        assert "0.6" in result  # total_royalties

    @pytest.mark.asyncio
    async def test_empty_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {"entries": [], "total": 0, "offset": 0, "limit": 20}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_my_content({})
        assert "No uploaded content" in result

    @pytest.mark.asyncio
    async def test_503_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "ContentUploader not initialized."}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_my_content({})
        assert "not configured" in result.lower()
