"""prsm_royalty_claim MCP tool handler.

Closes the loop on the offramp-quote claim_required path: the
backend endpoint at POST /wallet/royalty/claim is wrapped by
this MCP handler so operators can claim accumulated FTNS
royalties directly from their AI side-panel.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_royalty_claim,
)


# ──────────────────────────────────────────────────────────────────────
# Tool registration
# ──────────────────────────────────────────────────────────────────────


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_royalty_claim" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_royalty_claim" in names

    def test_dry_run_arg_in_schema(self):
        tool = next(t for t in TOOLS if t.name == "prsm_royalty_claim")
        assert "dry_run" in tool.inputSchema["properties"]
        assert tool.inputSchema["properties"]["dry_run"]["default"] is True


# ──────────────────────────────────────────────────────────────────────
# Handler — happy path rendering
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClaimHandler:
    @pytest.mark.asyncio
    async def test_dry_run_renders_artifact(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "POST"
            assert path == "/wallet/royalty/claim"
            assert data["dry_run"] is True
            return {
                "status": "DRY_RUN",
                "claimable_ftns": 5.0,
                "amount_claimed_ftns": 0.0,
                "tx_hash": None,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_royalty_claim({})
        assert "5.000000 FTNS" in result
        assert "DRY_RUN" in result
        assert "dry_run=false" in result.lower() or \
            "dry_run\": false" in result.lower()

    @pytest.mark.asyncio
    async def test_executed_renders_tx_hash(self):
        async def fake_call_node_api(method, path, data=None):
            assert data["dry_run"] is False
            return {
                "status": "EXECUTED",
                "claimable_ftns": 3.0,
                "amount_claimed_ftns": 3.0,
                "tx_hash": "0x" + "ab" * 32,
                "transfer_status": "OK",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_royalty_claim({"dry_run": False})
        assert "3.000000 FTNS" in result
        assert "EXECUTED" in result
        assert "0x" + "ab" * 32 in result

    @pytest.mark.asyncio
    async def test_skipped_zero_renders_no_op_message(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "SKIPPED_ZERO",
                "claimable_ftns": 0.0,
                "amount_claimed_ftns": 0.0,
                "tx_hash": None,
                "note": "No claimable balance.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_royalty_claim({"dry_run": False})
        assert "SKIPPED_ZERO" in result
        assert "0.000000 FTNS" in result


# ──────────────────────────────────────────────────────────────────────
# Handler — error paths
# ──────────────────────────────────────────────────────────────────────


class TestRoyaltyClaimErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_royalty_claim({})
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_503_distributor_not_wired(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "detail": "RoyaltyDistributor client not wired on this node.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_royalty_claim({})
        assert "not configured" in result.lower() or \
            "not wired" in result.lower()
