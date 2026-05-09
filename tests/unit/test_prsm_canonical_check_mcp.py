"""prsm_canonical_check MCP tool handler.

Post-migration verification: filter /health/detailed for
canonical-match fields, render pass/fail summary.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_canonical_check,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_canonical_check" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_canonical_check" in names


class TestAllMatch:
    @pytest.mark.asyncio
    async def test_renders_all_pin_match_summary(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "subsystems": {
                    "ftns_ledger": {
                        "available": True,
                        "wired_address": "0x5276…",
                        "canonical_address": "0x5276…",
                        "canonical_match": True,
                    },
                    "royalty_distributor": {
                        "available": True,
                        "wired_address": "0xfEa9…",
                        "canonical_address": "0xfEa9…",
                        "canonical_match": True,
                    },
                    "payment_escrow": {
                        "available": True, "status": "ok",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_canonical_check({})
        assert "ALL 2 PIN(S) MATCH" in result
        assert "ftns_ledger" in result
        assert "royalty_distributor" in result


class TestMismatch:
    @pytest.mark.asyncio
    async def test_renders_mismatch_with_action_hint(self):
        """Post-A-08 scenario: operator pinned to v1 RoyaltyDistributor."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "subsystems": {
                    "ftns_ledger": {
                        "available": True,
                        "wired_address": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
                        "canonical_address": "0x5276a3756C85f2E9e46f6D34386167a209aa16e5",
                        "canonical_match": True,
                    },
                    "royalty_distributor": {
                        "available": True,
                        "wired_address": "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2",  # v1
                        "canonical_address": "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e",  # v2
                        "canonical_match": False,
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_canonical_check({})
        assert "MISMATCH" in result
        assert "1 match" in result
        # Both wired and canonical addresses surfaced.
        assert "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2" in result
        assert "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e" in result
        # Action hint present.
        assert "operator action" in result.lower()
        assert "PRSM_" in result

    @pytest.mark.asyncio
    async def test_multiple_mismatches_listed(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "subsystems": {
                    "ftns_ledger": {
                        "available": True,
                        "wired_address": "0xWRONG_FTNS",
                        "canonical_address": "0xCORRECT_FTNS",
                        "canonical_match": False,
                    },
                    "royalty_distributor": {
                        "available": True,
                        "wired_address": "0xWRONG_ROYALTY",
                        "canonical_address": "0xCORRECT_ROYALTY",
                        "canonical_match": False,
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_canonical_check({})
        assert "2 MISMATCH(ES)" in result
        assert "ftns_ledger" in result
        assert "royalty_distributor" in result


class TestSkipped:
    @pytest.mark.asyncio
    async def test_subsystems_without_canonical_listed_as_skipped(self):
        """Subsystems that don't have canonical-match implemented
        (e.g., job_history, payment_escrow) appear in the
        Skipped section if available."""
        async def fake_call_node_api(method, path, data=None):
            return {
                "subsystems": {
                    "ftns_ledger": {
                        "available": True,
                        "wired_address": "0x...",
                        "canonical_address": "0x...",
                        "canonical_match": True,
                    },
                    "payment_escrow": {
                        "available": True, "status": "ok",
                    },
                    "job_history": {
                        "available": True, "status": "ok",
                    },
                },
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_canonical_check({})
        assert "Skipped" in result
        assert "payment_escrow" in result
        assert "job_history" in result


class TestErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_canonical_check({})
        assert "cannot reach" in result.lower()
