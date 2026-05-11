"""Sprint 229 — prsm_stake_lookup MCP tool.

GET /staking/stakes/{stake_id} + GET /staking/unstake-requests/
{request_id} had no MCP wrappers. Consolidates into single
`prsm_stake_lookup` with `kind` selector (stake|unstake_request)
since both follow the same id-lookup pattern.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_stake_lookup


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_stake_lookup" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_id_rejected(self):
        result = await handle_prsm_stake_lookup({"kind": "stake"})
        assert "id" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_kind_rejected(self):
        result = await handle_prsm_stake_lookup({
            "kind": "bogus", "id": "x",
        })
        assert "must be" in result.lower()


class TestRouting:
    @pytest.mark.asyncio
    async def test_stake_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"stake_id": "s-1", "amount": 100}),
        ) as mock_call:
            await handle_prsm_stake_lookup({
                "kind": "stake", "id": "s-1",
            })
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/staking/stakes/s-1"

    @pytest.mark.asyncio
    async def test_unstake_request_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"request_id": "u-1", "amount": 50}),
        ) as mock_call:
            await handle_prsm_stake_lookup({
                "kind": "unstake_request", "id": "u-1",
            })
        args, _ = mock_call.await_args
        assert args[1] == "/staking/unstake-requests/u-1"


class TestNotFound:
    @pytest.mark.asyncio
    async def test_404_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"detail": "Stake not found"}),
        ):
            result = await handle_prsm_stake_lookup({
                "kind": "stake", "id": "missing",
            })
        assert "not found" in result.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_stake_lookup({
                "kind": "stake", "id": "s-1",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
