"""Sprint 275 — prsm_marketplace_reputation MCP tool.

Wraps the sprint-275 reputation read endpoints behind a
single action-selector tool (list | lookup). Provides
operators with on-demand visibility into the marketplace
candidate-pool's reputation state.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_marketplace_reputation,
)


def test_tool_registered():
    assert "prsm_marketplace_reputation" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_marketplace_reputation({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_marketplace_reputation(
            {"action": "explode"},
        )
        assert "must be" in r.lower()


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "providers": [], "count": 0, "limit": 100,
            }),
        ) as mock_call:
            r = await handle_prsm_marketplace_reputation(
                {"action": "list"},
            )
        args = mock_call.await_args[0]
        assert args[0] == "GET"
        assert args[1].startswith("/marketplace/reputation")
        assert "0 known" in r or "0 of 0" in r or "count: 0" in r

    @pytest.mark.asyncio
    async def test_list_renders_provider_rows(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "providers": [
                    {
                        "provider_id": "good-prov",
                        "known": True,
                        "score": 0.95,
                        "successes": 20,
                        "failures": 1,
                        "preempted": 0,
                        "slashed_count": 0,
                        "has_been_slashed": False,
                        "latency_p50_ms": 100.0,
                        "latency_p95_ms": 250.0,
                        "first_seen_unix": 100,
                        "last_seen_unix": 200,
                    },
                    {
                        "provider_id": "slashed-prov",
                        "known": True,
                        "score": 0.10,
                        "successes": 5,
                        "failures": 0,
                        "preempted": 0,
                        "slashed_count": 2,
                        "has_been_slashed": True,
                        "latency_p50_ms": 110.0,
                        "latency_p95_ms": 300.0,
                        "first_seen_unix": 100,
                        "last_seen_unix": 200,
                    },
                ],
                "count": 2, "limit": 100,
            }),
        ):
            r = await handle_prsm_marketplace_reputation(
                {"action": "list"},
            )
        assert "good-prov" in r
        assert "slashed-prov" in r
        assert "0.95" in r or "95" in r
        # Slash marker visible somewhere
        assert "slash" in r.lower() or "⚠" in r

    @pytest.mark.asyncio
    async def test_list_passes_limit(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "providers": [], "count": 0, "limit": 50,
            }),
        ) as mock_call:
            await handle_prsm_marketplace_reputation({
                "action": "list", "limit": 50,
            })
        path = mock_call.await_args[0][1]
        assert "limit=50" in path

    @pytest.mark.asyncio
    async def test_list_not_initialized_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Reputation tracker not initialized.",
            }),
        ):
            r = await handle_prsm_marketplace_reputation(
                {"action": "list"},
            )
        assert "not wired" in r.lower() or "not initialized" in r.lower()


class TestLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_provider_id(self):
        r = await handle_prsm_marketplace_reputation(
            {"action": "lookup"},
        )
        assert "provider_id" in r

    @pytest.mark.asyncio
    async def test_lookup_unknown_provider(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "provider_id": "unknown",
                "known": False,
                "score": 0.5,
                "successes": 0, "failures": 0, "preempted": 0,
                "slashed_count": 0, "has_been_slashed": False,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "first_seen_unix": 0,
                "last_seen_unix": 0,
                "slash_events": [],
            }),
        ) as mock_call:
            r = await handle_prsm_marketplace_reputation(
                {"action": "lookup", "provider_id": "unknown"},
            )
        args = mock_call.await_args[0]
        assert args[1] == "/marketplace/reputation/unknown"
        assert "unknown" in r
        # Cold-start neutral score signaled
        assert "0.5" in r or "neutral" in r.lower() or "0.500" in r

    @pytest.mark.asyncio
    async def test_lookup_renders_slash_events(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "provider_id": "p1",
                "known": True,
                "score": 0.20,
                "successes": 5, "failures": 1, "preempted": 0,
                "slashed_count": 1, "has_been_slashed": True,
                "latency_p50_ms": 100.0,
                "latency_p95_ms": 250.0,
                "first_seen_unix": 100,
                "last_seen_unix": 200,
                "slash_events": [{
                    "batch_id": "batch1",
                    "slash_amount_wei": 5000,
                    "reason": "DOUBLE_SPEND",
                    "recorded_unix": 150,
                    "tx_hash": "0xdeadbeef",
                }],
            }),
        ):
            r = await handle_prsm_marketplace_reputation(
                {"action": "lookup", "provider_id": "p1"},
            )
        assert "p1" in r
        assert "batch1" in r
        assert "DOUBLE_SPEND" in r
        assert "0xdeadbeef" in r
