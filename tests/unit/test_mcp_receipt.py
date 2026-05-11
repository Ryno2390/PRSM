"""Sprint 242 — GET /compute/receipt/{job_id} endpoint + prsm_receipt MCP."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_receipt


# ── Endpoint ──────────────────────────────────────────────


def _endpoint_client(store=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._receipt_store = store
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_endpoint_returns_stored_receipt():
    from prsm.node.receipt_store import ReceiptStore
    store = ReceiptStore()
    store.put("job-1", {"job_id": "job-1", "model_id": "m1"})
    resp = _endpoint_client(store).get("/compute/receipt/job-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-1"
    assert body["model_id"] == "m1"


def test_endpoint_404_when_missing():
    from prsm.node.receipt_store import ReceiptStore
    resp = _endpoint_client(ReceiptStore()).get(
        "/compute/receipt/missing",
    )
    assert resp.status_code == 404


def test_endpoint_503_when_store_unwired():
    resp = _endpoint_client(None).get("/compute/receipt/job-1")
    assert resp.status_code == 503


# ── MCP wrapper ──────────────────────────────────────────


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_receipt" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_job_id_rejected(self):
        result = await handle_prsm_receipt({})
        assert "job_id" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_renders(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "job_id": "job-1",
                "request_id": "req-1",
                "model_id": "mock-llama-3-8b",
                "privacy_tier": "standard",
                "content_tier": "A",
                "tee_type": "software",
                "cost_ftns": "0.10",
                "settler_node_id": "settler-7",
            }),
        ) as mock_call:
            result = await handle_prsm_receipt({"job_id": "job-1"})
        args, _ = mock_call.await_args
        assert args[1] == "/compute/receipt/job-1"
        assert "job-1" in result
        assert "mock-llama-3-8b" in result
        assert "0.10" in result
        assert "verify_receipt" in result


class TestNotFound:
    @pytest.mark.asyncio
    async def test_404_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "No receipt for job_id='x'"}),
        ):
            result = await handle_prsm_receipt({"job_id": "x"})
        assert "no receipt" in result.lower()


class TestUnwired:
    @pytest.mark.asyncio
    async def test_503_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Receipt store not initialized.",
            }),
        ):
            result = await handle_prsm_receipt({"job_id": "x"})
        assert "PRSM_RECEIPT_STORE_DIR" in result or "not wired" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_receipt({"job_id": "x"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
