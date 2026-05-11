"""Sprint 267 — prsm_pinned_stats + prsm_provider_reputations
MCP wrappers."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS,
    handle_prsm_pinned_stats,
    handle_prsm_provider_reputations,
)


def test_tools_registered():
    assert "prsm_pinned_stats" in TOOL_HANDLERS
    assert "prsm_provider_reputations" in TOOL_HANDLERS


class TestPinnedStats:
    @pytest.mark.asyncio
    async def test_renders_list(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "pinned": [
                    {
                        "cid": "cid-a",
                        "size_bytes": 1024,
                        "requester_id": "user-1",
                        "last_verified": 200.0,
                        "successful_challenges": 5,
                        "failed_challenges": 0,
                    },
                    {
                        "cid": "cid-b",
                        "size_bytes": 4096,
                        "requester_id": "user-2",
                        "last_verified": None,
                        "successful_challenges": 0,
                        "failed_challenges": 0,
                    },
                ],
                "count": 2,
            }),
        ):
            result = await handle_prsm_pinned_stats({})
        assert "cid-a" in result
        assert "cid-b" in result
        assert "verified=200.0" in result
        assert "verified=NEVER" in result

    @pytest.mark.asyncio
    async def test_empty_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"pinned": [], "count": 0}),
        ):
            result = await handle_prsm_pinned_stats({})
        assert "no pinned" in result.lower()

    @pytest.mark.asyncio
    async def test_unwired_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Storage provider not initialized.",
            }),
        ):
            result = await handle_prsm_pinned_stats({})
        assert "not wired" in result.lower()


class TestProviderReputations:
    @pytest.mark.asyncio
    async def test_sorted_by_reputation(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "providers": {
                    "low": {
                        "reputation": 0.30,
                        "total_challenges": 10,
                        "successful_proofs": 3,
                        "failed_proofs": 7,
                        "expired_challenges": 0,
                    },
                    "high": {
                        "reputation": 0.95,
                        "total_challenges": 100,
                        "successful_proofs": 95,
                        "failed_proofs": 3,
                        "expired_challenges": 2,
                    },
                    "mid": {
                        "reputation": 0.60,
                        "total_challenges": 50,
                        "successful_proofs": 30,
                        "failed_proofs": 18,
                        "expired_challenges": 2,
                    },
                },
                "count": 3,
            }),
        ):
            result = await handle_prsm_provider_reputations({})
        # Highest first
        h_idx = result.find("high")
        m_idx = result.find("mid")
        l_idx = result.find("low")
        assert h_idx < m_idx < l_idx
        assert "0.950" in result

    @pytest.mark.asyncio
    async def test_empty_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"providers": {}, "count": 0}),
        ):
            result = await handle_prsm_provider_reputations({})
        assert "no provider" in result.lower()

    @pytest.mark.asyncio
    async def test_network_error(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            result = await handle_prsm_provider_reputations({})
        assert "running" in result.lower() or "failed" in result.lower()
