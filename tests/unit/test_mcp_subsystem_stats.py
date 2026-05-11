"""Sprint 216 — prsm_subsystem_stats MCP tool.

Three operator-side stats endpoints had no MCP coverage:
  - GET /settler/stats   — Phase 6 settler-network health
  - GET /storage/stats   — local storage provider stats
  - GET /compute/stats   — local compute provider stats

Consolidates into a single MCP tool with `subsystem` selector
(settler|storage|compute) so the AI side-panel doesn't sprawl
into 3 nearly-identical tools for trivial wrappers.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_subsystem_stats


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_subsystem_stats" in TOOL_HANDLERS
        assert TOOL_HANDLERS["prsm_subsystem_stats"] is handle_prsm_subsystem_stats


class TestRouting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("subsystem,path", [
        ("settler", "/settler/stats"),
        ("storage", "/storage/stats"),
        ("compute", "/compute/stats"),
    ])
    async def test_routes_to_correct_endpoint(self, subsystem, path):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"available": True}),
        ) as mock_call:
            await handle_prsm_subsystem_stats({"subsystem": subsystem})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == path


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_subsystem_rejected(self):
        result = await handle_prsm_subsystem_stats({})
        assert "subsystem" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_subsystem_rejected(self):
        result = await handle_prsm_subsystem_stats({"subsystem": "bogus"})
        assert "must be one of" in result.lower()


class TestRender:
    @pytest.mark.asyncio
    async def test_renders_dict_keys(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "pledged_gb": 100, "used_gb": 25,
                "pinned_count": 17, "available": True,
            }),
        ):
            result = await handle_prsm_subsystem_stats({
                "subsystem": "storage",
            })
        assert "pledged_gb" in result
        assert "100" in result
        assert "25" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_node_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_subsystem_stats({
                "subsystem": "storage",
            })
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
