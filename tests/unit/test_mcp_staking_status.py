"""Sprint 215 — prsm_staking_status MCP tool.

GET /staking/status returns the user's complete staking dashboard
(active stakes, pending unstake requests, total staked, rewards
earned/claimed). No MCP wrapper existed, so stakers tracking
their positions had to curl manually.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_staking_status


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_staking_status" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_staking_status"] is handle_prsm_staking_status


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_full_status(self):
        mock_resp = {
            "user_id": "user-a",
            "total_staked": 1000.0,
            "active_stakes": [
                {
                    "stake_id": "s-1",
                    "amount": 500.0,
                    "stake_type": "governance",
                    "status": "active",
                    "staked_at": "2026-05-01T00:00:00",
                    "rewards_earned": 10.5,
                    "rewards_claimed": 5.0,
                },
                {
                    "stake_id": "s-2",
                    "amount": 500.0,
                    "stake_type": "validation",
                    "status": "active",
                    "staked_at": "2026-05-05T00:00:00",
                    "rewards_earned": 2.5,
                    "rewards_claimed": 0.0,
                },
            ],
            "pending_unstake_requests": [
                {
                    "request_id": "u-1",
                    "stake_id": "s-3",
                    "amount": 200.0,
                    "requested_at": "2026-05-10T00:00:00",
                    "available_at": "2026-06-09T00:00:00",
                    "status": "pending",
                },
            ],
            "total_rewards_earned": 13.0,
            "total_rewards_claimed": 5.0,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_staking_status({})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/staking/status"
        assert "1000" in result
        assert "s-1" in result
        assert "s-2" in result
        assert "governance" in result
        assert "validation" in result
        assert "u-1" in result
        assert "13" in result

    @pytest.mark.asyncio
    async def test_empty_status_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "user-a",
                "total_staked": 0,
                "active_stakes": [],
                "pending_unstake_requests": [],
                "total_rewards_earned": 0,
                "total_rewards_claimed": 0,
            }),
        ):
            result = await handle_prsm_staking_status({})
        assert "no active stakes" in result.lower() or "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_node_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_staking_status({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
