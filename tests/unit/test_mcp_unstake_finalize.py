"""Sprint 219 — prsm_unstake_finalize MCP tool.

Two write-side endpoints close the unstake lifecycle:
  - POST /staking/withdraw/{request_id} — withdraw after unlock
  - POST /staking/cancel-unstake/{request_id} — cancel before
    unlock (restores tokens to active staking)

Consolidates into a single MCP tool with `action` selector
(withdraw|cancel) so the AI side-panel doesn't sprawl into two
near-identical wrappers for the same request_id lifecycle.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_unstake_finalize


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_unstake_finalize" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_request_id_rejected(self):
        result = await handle_prsm_unstake_finalize({
            "action": "withdraw",
        })
        assert "request_id" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        result = await handle_prsm_unstake_finalize({
            "request_id": "u-1",
        })
        assert "action" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        result = await handle_prsm_unstake_finalize({
            "request_id": "u-1", "action": "bogus",
        })
        assert "must be" in result.lower()


class TestWithdraw:
    @pytest.mark.asyncio
    async def test_routes_to_withdraw_endpoint(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "request_id": "u-1",
                "success": True,
                "amount_withdrawn": 500.0,
            }),
        ) as mock_call:
            result = await handle_prsm_unstake_finalize({
                "request_id": "u-1", "action": "withdraw",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/staking/withdraw/u-1"
        assert "500" in result


class TestCancel:
    @pytest.mark.asyncio
    async def test_routes_to_cancel_endpoint(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "request_id": "u-1",
                "cancelled": True,
                "reason": "changed mind",
            }),
        ) as mock_call:
            result = await handle_prsm_unstake_finalize({
                "request_id": "u-1", "action": "cancel",
                "reason": "changed mind",
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert "/staking/cancel-unstake/u-1" in args[1]
        assert "reason=changed%20mind" in args[1] or "reason=changed+mind" in args[1] or "changed mind" in args[1]


class TestNotFound:
    @pytest.mark.asyncio
    async def test_404_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "request not found"}),
        ):
            result = await handle_prsm_unstake_finalize({
                "request_id": "missing", "action": "withdraw",
            })
        assert "not found" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_unstake_finalize({
                "request_id": "u-1", "action": "withdraw",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
