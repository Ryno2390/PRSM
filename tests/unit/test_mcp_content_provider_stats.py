"""Sprint 268 — prsm_content_provider_stats MCP wrapper."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_content_provider_stats,
)


def test_tool_in_handlers():
    assert "prsm_content_provider_stats" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_renders_with_nested_discovery():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "local_content_count": 42,
            "pending_requests": 3,
            "discovery": {
                "queries_sent": 100,
                "responses_received": 75,
            },
            "total_fetches": 200,
        }),
    ):
        result = await handle_prsm_content_provider_stats({})
    assert "local_content_count" in result
    assert "42" in result
    assert "discovery:" in result
    assert "queries_sent" in result
    assert "100" in result


@pytest.mark.asyncio
async def test_503_friendly():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "Content provider not initialized.",
        }),
    ):
        result = await handle_prsm_content_provider_stats({})
    assert "not wired" in result.lower()


@pytest.mark.asyncio
async def test_network_error():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(side_effect=RuntimeError("conn refused")),
    ):
        result = await handle_prsm_content_provider_stats({})
    assert "running" in result.lower() or "failed" in result.lower()
