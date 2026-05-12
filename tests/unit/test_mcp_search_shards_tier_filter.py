"""Sprint 289 — prsm_search_shards tier filter passthrough.

The MCP tool forwards optional min_tier + exclude_new kwargs
to /content/search and renders creator_tier in the result
table.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import handle_prsm_search_shards


@pytest.mark.asyncio
async def test_search_passes_min_tier_to_endpoint():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={"results": [], "count": 0}),
    ) as mock_call:
        await handle_prsm_search_shards({
            "query": "foo", "min_tier": "medium",
        })
    path = mock_call.await_args[0][1]
    assert "min_tier=medium" in path


@pytest.mark.asyncio
async def test_search_passes_exclude_new():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={"results": [], "count": 0}),
    ) as mock_call:
        await handle_prsm_search_shards({
            "query": "foo", "exclude_new": True,
        })
    path = mock_call.await_args[0][1]
    assert "exclude_new=true" in path


@pytest.mark.asyncio
async def test_search_no_tier_args_no_extras():
    """Defaults: omit tier params (preserves pre-sprint-289
    behavior + minimal query string)."""
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={"results": [], "count": 0}),
    ) as mock_call:
        await handle_prsm_search_shards({"query": "foo"})
    path = mock_call.await_args[0][1]
    assert "min_tier" not in path
    assert "exclude_new" not in path


@pytest.mark.asyncio
async def test_search_renders_creator_tier_in_rows():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "results": [{
                "cid": "bafy-abc",
                "filename": "data.bin",
                "size_bytes": 1024,
                "creator_id": "0xcreator-long-address-here",
                "creator_tier": "high",
                "providers": ["p1"],
            }],
            "count": 1,
        }),
    ):
        r = await handle_prsm_search_shards({"query": "foo"})
    assert "high" in r
    assert "bafy-abc" in r


@pytest.mark.asyncio
async def test_search_invalid_min_tier_rejected_client_side():
    """Defense in depth: client-side reject bogus tier so we
    don't waste an RPC roundtrip."""
    r = await handle_prsm_search_shards({
        "query": "foo", "min_tier": "explode",
    })
    assert "must be" in r.lower() or "invalid" in r.lower()
