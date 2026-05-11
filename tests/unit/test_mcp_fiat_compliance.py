"""Sprint 282 — prsm_fiat_compliance MCP tool.

Operator-facing query surface for the fiat compliance ring.
action selector: list | summary | lookup. No write paths via
MCP — recording is automatic from quote/execute handlers.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_fiat_compliance,
)


def test_tool_registered():
    assert "prsm_fiat_compliance" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_fiat_compliance({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_fiat_compliance(
            {"action": "explode"},
        )
        assert "must be" in r.lower()


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [], "count": 0,
                "limit": 100, "offset": 0,
            }),
        ) as mock_call:
            r = await handle_prsm_fiat_compliance(
                {"action": "list"},
            )
        args = mock_call.await_args[0]
        assert args[0] == "GET"
        assert args[1].startswith("/admin/fiat-compliance")
        assert "0" in r

    @pytest.mark.asyncio
    async def test_list_renders_entries(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [{
                    "entry_id": "abc-123-def",
                    "timestamp": 100.0,
                    "kind": "onramp_quote",
                    "user_id": "alice",
                    "usd_amount": 100.0,
                    "ftns_amount": 100.0,
                    "status": "PENDING_COMMISSION",
                    "kyc_status": "VERIFIED",
                    "tx_hash": None,
                    "address": "0xabc",
                    "jurisdiction": "US",
                    "metadata": {},
                }],
                "count": 1, "limit": 100, "offset": 0,
            }),
        ):
            r = await handle_prsm_fiat_compliance(
                {"action": "list"},
            )
        assert "alice" in r
        assert "onramp_quote" in r
        assert "100" in r

    @pytest.mark.asyncio
    async def test_list_passes_kind_filter(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [], "count": 0,
                "limit": 100, "offset": 0,
            }),
        ) as mock_call:
            await handle_prsm_fiat_compliance({
                "action": "list",
                "kind": "onramp_quote",
            })
        path = mock_call.await_args[0][1]
        assert "kind=onramp_quote" in path

    @pytest.mark.asyncio
    async def test_list_passes_user_filter(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entries": [], "count": 0,
                "limit": 100, "offset": 0,
            }),
        ) as mock_call:
            await handle_prsm_fiat_compliance({
                "action": "list",
                "user_id": "alice",
            })
        path = mock_call.await_args[0][1]
        assert "user_id=alice" in path

    @pytest.mark.asyncio
    async def test_list_503_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": (
                    "Fiat compliance ring not initialized."
                ),
            }),
        ):
            r = await handle_prsm_fiat_compliance(
                {"action": "list"},
            )
        assert "not wired" in r.lower() or "not initialized" in r.lower()


class TestSummary:
    @pytest.mark.asyncio
    async def test_summary_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "by_kind": {}, "total_entries": 0,
            }),
        ) as mock_call:
            r = await handle_prsm_fiat_compliance(
                {"action": "summary"},
            )
        args = mock_call.await_args[0]
        assert args[1] == "/admin/fiat-compliance/summary"
        assert "0" in r

    @pytest.mark.asyncio
    async def test_summary_renders_buckets(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "by_kind": {
                    "onramp_quote": {
                        "count": 5, "total_usd": 500.0,
                    },
                    "offramp_quote": {
                        "count": 3, "total_usd": 150.0,
                    },
                    "kyc_initiate": {
                        "count": 2, "total_usd": 0.0,
                    },
                },
                "total_entries": 10,
            }),
        ):
            r = await handle_prsm_fiat_compliance(
                {"action": "summary"},
            )
        assert "onramp_quote" in r
        assert "500" in r
        assert "offramp_quote" in r
        assert "150" in r
        assert "kyc_initiate" in r


class TestLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_entry_id(self):
        r = await handle_prsm_fiat_compliance(
            {"action": "lookup"},
        )
        assert "entry_id" in r

    @pytest.mark.asyncio
    async def test_lookup_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "entry_id": "abc-123",
                "timestamp": 100.0,
                "kind": "onramp_quote",
                "user_id": "alice",
                "usd_amount": 100.0,
                "ftns_amount": 100.0,
                "status": "PENDING_COMMISSION",
                "kyc_status": "VERIFIED",
                "tx_hash": None,
                "vendor_ref": None,
                "address": "0xabc",
                "jurisdiction": "US",
                "metadata": {"payment_method_alias": "primary"},
            }),
        ) as mock_call:
            r = await handle_prsm_fiat_compliance({
                "action": "lookup",
                "entry_id": "abc-123",
            })
        args = mock_call.await_args[0]
        assert args[1] == "/admin/fiat-compliance/abc-123"
        assert "abc-123" in r
        assert "alice" in r
        assert "VERIFIED" in r

    @pytest.mark.asyncio
    async def test_lookup_404_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "no entry with id='missing'",
            }),
        ):
            r = await handle_prsm_fiat_compliance({
                "action": "lookup",
                "entry_id": "missing",
            })
        assert "no entry" in r.lower()
