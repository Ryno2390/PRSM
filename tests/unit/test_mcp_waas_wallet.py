"""Sprint 276 — prsm_waas_wallet MCP tool.

Wraps the sprint-276 WaaS endpoints behind a single action-
selector tool (provision | lookup | list | status). Per the
Coinbase SDK seamlessness principle, this is the LLM-facing
surface that lets a user say "give me a wallet" and have one
provisioned without ever seeing crypto plumbing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_waas_wallet,
)


def test_tool_registered():
    assert "prsm_waas_wallet" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_waas_wallet({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_waas_wallet(
            {"action": "explode"},
        )
        assert "must be" in r.lower()


class TestProvision:
    @pytest.mark.asyncio
    async def test_provision_requires_user_id(self):
        r = await handle_prsm_waas_wallet({
            "action": "provision",
            "email": "a@x.io",
        })
        assert "user_id" in r

    @pytest.mark.asyncio
    async def test_provision_requires_email(self):
        r = await handle_prsm_waas_wallet({
            "action": "provision",
            "user_id": "alice",
        })
        assert "email" in r

    @pytest.mark.asyncio
    async def test_provision_pending_commission_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice",
                "email": "a@x.io",
                "wallet_id": None,
                "address": None,
                "network": "base-mainnet",
                "status": "PENDING_COMMISSION",
                "created_at": 100.0,
            }),
        ) as mock_call:
            r = await handle_prsm_waas_wallet({
                "action": "provision",
                "user_id": "alice",
                "email": "a@x.io",
            })
        args = mock_call.await_args[0]
        assert args[0] == "POST"
        assert args[1] == "/wallet/waas/provision"
        assert "PENDING_COMMISSION" in r
        assert "alice" in r

    @pytest.mark.asyncio
    async def test_provision_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice",
                "email": "a@x.io",
                "wallet_id": "w-alice",
                "address": "0xabc",
                "network": "base-mainnet",
                "status": "PROVISIONED",
                "created_at": 100.0,
            }),
        ):
            r = await handle_prsm_waas_wallet({
                "action": "provision",
                "user_id": "alice",
                "email": "a@x.io",
            })
        assert "alice" in r
        assert "w-alice" in r
        assert "0xabc" in r
        assert "PROVISIONED" in r


class TestLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_user_id(self):
        r = await handle_prsm_waas_wallet({"action": "lookup"})
        assert "user_id" in r

    @pytest.mark.asyncio
    async def test_lookup_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "alice",
                "email": "a@x.io",
                "wallet_id": "w-alice",
                "address": "0xabc",
                "network": "base-mainnet",
                "status": "PROVISIONED",
                "created_at": 100.0,
            }),
        ) as mock_call:
            r = await handle_prsm_waas_wallet({
                "action": "lookup",
                "user_id": "alice",
            })
        args = mock_call.await_args[0]
        assert args[0] == "GET"
        assert args[1] == "/wallet/waas/alice"
        assert "0xabc" in r

    @pytest.mark.asyncio
    async def test_lookup_missing_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "no wallet for user_id='nobody'",
            }),
        ):
            r = await handle_prsm_waas_wallet({
                "action": "lookup",
                "user_id": "nobody",
            })
        assert "no wallet" in r.lower()


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "wallets": [], "count": 0, "limit": 100,
            }),
        ):
            r = await handle_prsm_waas_wallet({"action": "list"})
        assert "0" in r

    @pytest.mark.asyncio
    async def test_list_renders_rows(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "wallets": [{
                    "user_id": "alice",
                    "email": "a@x.io",
                    "wallet_id": "w-alice",
                    "address": "0xabc",
                    "network": "base-mainnet",
                    "status": "PROVISIONED",
                    "created_at": 100.0,
                }],
                "count": 1, "limit": 100,
            }),
        ):
            r = await handle_prsm_waas_wallet({"action": "list"})
        assert "alice" in r
        assert "0xabc" in r


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_uncommissioned(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "commissioned": False,
                "network": "base-mainnet",
                "wallet_count": 0,
            }),
        ) as mock_call:
            r = await handle_prsm_waas_wallet(
                {"action": "status"},
            )
        args = mock_call.await_args[0]
        assert args[1] == "/wallet/waas/status"
        assert "PENDING_COMMISSION" in r or "False" in r
        assert "base-mainnet" in r

    @pytest.mark.asyncio
    async def test_status_commissioned(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "commissioned": True,
                "network": "base-mainnet",
                "wallet_count": 5,
            }),
        ):
            r = await handle_prsm_waas_wallet(
                {"action": "status"},
            )
        assert "commissioned" in r.lower()
        assert "5" in r
