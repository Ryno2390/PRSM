"""Sprint 218 — prsm_claim_rewards MCP tool.

POST /staking/claim-rewards had no MCP wrapper. Stakers
accumulating rewards could see them via prsm_staking_status
(sprint 215) but couldn't claim via MCP — only curl.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_claim_rewards


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_claim_rewards" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_claim_rewards"] is handle_prsm_claim_rewards


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_claim_all(self):
        """No stake_id = claim across all stakes."""
        mock_resp = {
            "user_id": "user-a",
            "total_rewards_claimed": 13.5,
            "stakes_processed": 3,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            result = await handle_prsm_claim_rewards({})
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1].startswith("/staking/claim-rewards")
        assert "stake_id" not in args[1]
        assert "13.5" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_claim_specific_stake(self):
        mock_resp = {
            "user_id": "user-a",
            "total_rewards_claimed": 5.0,
            "stakes_processed": 1,
        }
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value=mock_resp),
        ) as mock_call:
            await handle_prsm_claim_rewards({"stake_id": "s-7"})
        args, _ = mock_call.await_args
        assert "stake_id=s-7" in args[1]


class TestZeroRewards:
    @pytest.mark.asyncio
    async def test_zero_rewards_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "user_id": "user-a",
                "total_rewards_claimed": 0.0,
                "stakes_processed": 0,
            }),
        ):
            result = await handle_prsm_claim_rewards({})
        assert "0" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_claim_rewards({})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
