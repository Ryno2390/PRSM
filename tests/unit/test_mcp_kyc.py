"""Sprint 280 — prsm_kyc MCP tool.

LLM-facing surface for KYC adapter. action selector:
initiate | lookup | list | status. Vendor webhook routing
intentionally NOT exposed via MCP (server-to-server only).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_kyc


def test_tool_registered():
    assert "prsm_kyc" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_kyc({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_kyc({"action": "explode"})
        assert "must be" in r.lower()


class TestInitiate:
    @pytest.mark.asyncio
    async def test_initiate_requires_user_id(self):
        r = await handle_prsm_kyc({
            "action": "initiate", "email": "a@x.io",
        })
        assert "user_id" in r

    @pytest.mark.asyncio
    async def test_initiate_requires_email(self):
        r = await handle_prsm_kyc({
            "action": "initiate", "user_id": "alice",
        })
        assert "email" in r

    @pytest.mark.asyncio
    async def test_initiate_pending_commission(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice", "email": "a@x.io",
                "vendor": None, "vendor_ref": None,
                "session_url": None,
                "level": "basic",
                "status": "PENDING_COMMISSION",
                "created_at": 100.0, "verified_at": 0,
            }),
        ) as mock_call:
            r = await handle_prsm_kyc({
                "action": "initiate", "user_id": "alice",
                "email": "a@x.io",
            })
        args = mock_call.await_args[0]
        assert args[0] == "POST"
        assert args[1] == "/wallet/kyc/initiate"
        assert "PENDING_COMMISSION" in r

    @pytest.mark.asyncio
    async def test_initiate_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice", "email": "a@x.io",
                "vendor": "persona",
                "vendor_ref": "persona-alice",
                "session_url": "https://persona.example/v/alice",
                "level": "basic", "status": "INITIATED",
                "created_at": 100.0, "verified_at": 0,
            }),
        ):
            r = await handle_prsm_kyc({
                "action": "initiate", "user_id": "alice",
                "email": "a@x.io",
            })
        assert "INITIATED" in r
        assert "persona.example/v/alice" in r

    @pytest.mark.asyncio
    async def test_initiate_passes_level(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice", "email": "a@x.io",
                "vendor": "persona",
                "vendor_ref": "ref", "session_url": "url",
                "level": "enhanced", "status": "INITIATED",
                "created_at": 100.0, "verified_at": 0,
            }),
        ) as mock_call:
            await handle_prsm_kyc({
                "action": "initiate", "user_id": "alice",
                "email": "a@x.io", "level": "enhanced",
            })
        body = mock_call.await_args[0][2]
        assert body["level"] == "enhanced"


class TestLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_user_id(self):
        r = await handle_prsm_kyc({"action": "lookup"})
        assert "user_id" in r

    @pytest.mark.asyncio
    async def test_lookup_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice", "email": "a@x.io",
                "vendor": "persona", "vendor_ref": "ref",
                "session_url": "url", "level": "basic",
                "status": "VERIFIED",
                "created_at": 100.0, "verified_at": 200.0,
            }),
        ) as mock_call:
            r = await handle_prsm_kyc({
                "action": "lookup", "user_id": "alice",
            })
        args = mock_call.await_args[0]
        assert args[1] == "/wallet/kyc/alice"
        assert "VERIFIED" in r

    @pytest.mark.asyncio
    async def test_lookup_404_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "no KYC record for user_id='ghost'",
            }),
        ):
            r = await handle_prsm_kyc({
                "action": "lookup", "user_id": "ghost",
            })
        assert "no kyc" in r.lower()


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "records": [], "count": 0, "limit": 100,
            }),
        ):
            r = await handle_prsm_kyc({"action": "list"})
        assert "0" in r

    @pytest.mark.asyncio
    async def test_list_populated(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "records": [{
                    "user_id": "alice", "email": "a@x.io",
                    "vendor": "persona", "vendor_ref": "r1",
                    "session_url": "url", "level": "basic",
                    "status": "VERIFIED",
                    "created_at": 100.0, "verified_at": 150.0,
                }],
                "count": 1, "limit": 100,
            }),
        ):
            r = await handle_prsm_kyc({"action": "list"})
        assert "alice" in r
        assert "VERIFIED" in r


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_uncommissioned(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "commissioned": False,
                "vendor": None,
                "supported_vendors": [
                    "persona", "onfido", "plaid",
                ],
                "record_count": 0,
            }),
        ) as mock_call:
            r = await handle_prsm_kyc({"action": "status"})
        args = mock_call.await_args[0]
        assert args[1] == "/wallet/kyc/status"
        assert "PENDING_COMMISSION" in r or "False" in r
        # Supported-vendors list surfaces so operator knows
        # what KYC_VENDOR values are valid.
        assert "persona" in r.lower()

    @pytest.mark.asyncio
    async def test_status_commissioned(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "commissioned": True,
                "vendor": "persona",
                "supported_vendors": [
                    "persona", "onfido", "plaid",
                ],
                "record_count": 5,
            }),
        ):
            r = await handle_prsm_kyc({"action": "status"})
        assert "commissioned" in r.lower()
        assert "persona" in r.lower()
        assert "5" in r
