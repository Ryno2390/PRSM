"""Sprint 234 — prsm_settler_admin MCP tool.

Four settler write endpoints had no MCP coverage:
  - POST /settler/register      — register as a settler
  - POST /settler/unbond        — initiate unbonding
  - POST /settler/batch/sign    — sign a pending batch
  - POST /settler/slash/propose — propose a slash

Consolidates into single tool with `action` selector. All four
are sensitive write ops — auth enforced server-side.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_settler_admin


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_settler_admin" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        result = await handle_prsm_settler_admin({})
        assert "action" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        result = await handle_prsm_settler_admin({"action": "bogus"})
        assert "must be" in result.lower()

    @pytest.mark.asyncio
    async def test_register_requires_args(self):
        result = await handle_prsm_settler_admin({"action": "register"})
        assert "settler_id" in result.lower() or "address" in result.lower() or "bond_amount" in result.lower()

    @pytest.mark.asyncio
    async def test_register_inf_bond_rejected(self):
        result = await handle_prsm_settler_admin({
            "action": "register",
            "settler_id": "s-1",
            "address": "0xabc",
            "bond_amount": float("inf"),
        })
        assert "bond_amount" in result.lower() or "finite" in result.lower()

    @pytest.mark.asyncio
    async def test_slash_inf_amount_rejected(self):
        result = await handle_prsm_settler_admin({
            "action": "slash",
            "settler_id": "s-1",
            "slash_amount": float("inf"),
            "reason": "bad behavior",
            "proposer_id": "p-1",
        })
        assert "slash_amount" in result.lower() or "finite" in result.lower()


class TestRouting:
    @pytest.mark.asyncio
    async def test_register_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "settler_id": "s-1",
                "status": "active",
            }),
        ) as mock_call:
            await handle_prsm_settler_admin({
                "action": "register",
                "settler_id": "s-1",
                "address": "0xabc",
                "bond_amount": 10000.0,
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1].startswith("/settler/register")
        assert "settler_id=s-1" in args[1]
        assert "bond_amount=10000" in args[1]

    @pytest.mark.asyncio
    async def test_unbond_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "settler_id": "s-1",
                "status": "unbonding",
            }),
        ) as mock_call:
            await handle_prsm_settler_admin({
                "action": "unbond",
                "settler_id": "s-1",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert "/settler/unbond" in args[1]

    @pytest.mark.asyncio
    async def test_sign_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "batch_id": "b-1",
                "settler_id": "s-1",
                "signature_count": 2,
                "threshold": 3,
                "approved": False,
            }),
        ) as mock_call:
            await handle_prsm_settler_admin({
                "action": "sign",
                "batch_id": "b-1",
                "settler_id": "s-1",
                "signature": "0xsig",
            })
        args, _ = mock_call.await_args
        assert "/settler/batch/sign" in args[1]

    @pytest.mark.asyncio
    async def test_slash_routes(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"proposal_id": "p-1"}),
        ) as mock_call:
            await handle_prsm_settler_admin({
                "action": "slash",
                "settler_id": "s-1",
                "slash_amount": 100.0,
                "reason": "double-sign",
                "proposer_id": "p-1",
            })
        args, _ = mock_call.await_args
        assert "/settler/slash/propose" in args[1]


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_settler_admin({
                "action": "unbond",
                "settler_id": "s-1",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
