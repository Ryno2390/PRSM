"""Sprint 232 — prsm_node_resources MCP tool.

GET /node/resources (current config + computed effective values)
+ PUT /node/resources (update at runtime) had no MCP coverage.
Operators wanting to view or tune their node's resource allocation
had to curl. Consolidates into single tool with `action` selector
(get|update).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import TOOL_HANDLERS, handle_prsm_node_resources


class TestRegistration:
    def test_tool_in_handlers(self):
        assert "prsm_node_resources" in TOOL_HANDLERS


class TestRouting:
    @pytest.mark.asyncio
    async def test_get_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "cpu_allocation_pct": 50,
                "memory_allocation_pct": 50,
                "storage_gb": 100.0,
                "effective_cpu_cores": 4.0,
                "effective_memory_gb": 8.0,
                "storage_available_gb": 95.0,
            }),
        ) as mock_call:
            await handle_prsm_node_resources({"action": "get"})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/node/resources"

    @pytest.mark.asyncio
    async def test_update_routes_correctly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "cpu_allocation_pct": 75,
                "effective_cpu_cores": 6.0,
            }),
        ) as mock_call:
            await handle_prsm_node_resources({
                "action": "update", "cpu_allocation_pct": 75,
            })
        args, kwargs = mock_call.await_args
        assert args[0] == "PUT"
        assert args[1] == "/node/resources"
        body = args[2] if len(args) > 2 else kwargs.get("data") or {}
        assert body.get("cpu_allocation_pct") == 75


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        result = await handle_prsm_node_resources({})
        assert "action" in result.lower()

    @pytest.mark.asyncio
    async def test_update_requires_at_least_one_field(self):
        result = await handle_prsm_node_resources({"action": "update"})
        assert "at least one" in result.lower() or "no fields" in result.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        result = await handle_prsm_node_resources({"action": "bogus"})
        assert "must be" in result.lower()


class TestRender:
    @pytest.mark.asyncio
    async def test_get_renders(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "cpu_allocation_pct": 50,
                "effective_cpu_cores": 4.0,
                "storage_available_gb": 95.0,
            }),
        ):
            result = await handle_prsm_node_resources({"action": "get"})
        assert "cpu_allocation_pct" in result
        assert "50" in result
        assert "4.0" in result or "4" in result


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_node_resources({"action": "get"})
        assert isinstance(result, str)
        assert "running" in result.lower() or "failed" in result.lower()
