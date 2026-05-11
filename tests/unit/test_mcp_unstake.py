"""Sprint 217 — prsm_unstake MCP tool.

POST /staking/unstake creates an unstake request (7-day default
unlock period). No MCP wrapper existed. Closes a gap visible to
any staker: they can stake via prsm_stake but couldn't unstake
without curl.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_unstake


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_unstake" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_unstake"] is handle_prsm_unstake


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_stake_id_rejected(self):
        result = await handle_prsm_unstake({})
        assert "stake_id" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_stake_id_rejected(self):
        result = await handle_prsm_unstake({"stake_id": "  "})
        assert "stake_id" in result.lower()

    @pytest.mark.asyncio
    async def test_negative_amount_rejected_locally(self):
        result = await handle_prsm_unstake({
            "stake_id": "s-1", "amount": -1,
        })
        assert "amount" in result.lower()

    @pytest.mark.asyncio
    async def test_inf_amount_rejected_locally(self):
        result = await handle_prsm_unstake({
            "stake_id": "s-1", "amount": float("inf"),
        })
        assert "amount" in result.lower()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_unstake(self):
        """amount omitted = unstake full stake."""
        mock_resp = {
            "request_id": "u-1",
            "stake_id": "s-1",
            "user_id": "user-a",
            "amount": 500.0,
            "requested_at": "2026-05-11T12:00:00",
            "available_at": "2026-05-18T12:00:00",
            "status": "pending",
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_unstake({"stake_id": "s-1"})
        args, kwargs = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/staking/unstake"
        # Body sent through.
        body = args[2] if len(args) > 2 else kwargs.get("data") or {}
        assert body.get("stake_id") == "s-1"
        assert "amount" not in body or body["amount"] is None
        assert "u-1" in result
        assert "available_at" in result.lower() or "2026-05-18" in result

    @pytest.mark.asyncio
    async def test_partial_unstake(self):
        mock_resp = {
            "request_id": "u-2",
            "stake_id": "s-1",
            "user_id": "user-a",
            "amount": 250.0,
            "requested_at": "2026-05-11T12:00:00",
            "available_at": "2026-05-18T12:00:00",
            "status": "pending",
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            await handle_prsm_unstake({
                "stake_id": "s-1", "amount": 250.0,
            })
        args, kwargs = mock_call.await_args
        body = args[2] if len(args) > 2 else kwargs.get("data") or {}
        assert body.get("amount") == 250.0


class TestNotFoundPath:
    @pytest.mark.asyncio
    async def test_404_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "stake not found"}),
        ):
            result = await handle_prsm_unstake({"stake_id": "missing"})
        assert "not found" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_unstake({"stake_id": "s-1"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
