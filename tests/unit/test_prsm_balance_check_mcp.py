"""prsm_balance_check MCP tool handler.

V1 scope per Vision §13 Phase 5: closes the explicit "not yet
created — currently just a stand-in" gap. Reads on-chain FTNS
balance + USD equivalent via the node API endpoint
``/balance/onchain`` (shipped same-sprint at
prsm/node/api.py:404+).
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest

from prsm.mcp_server import handle_prsm_balance_check, TOOL_HANDLERS, TOOLS


# ──────────────────────────────────────────────────────────────────────
# Tool registration
# ──────────────────────────────────────────────────────────────────────


class TestToolRegistration:
    def test_handler_registered_in_tool_dispatch(self):
        assert "prsm_balance_check" in TOOL_HANDLERS

    def test_tool_definition_present_in_TOOLS(self):
        names = [t.name for t in TOOLS]
        assert "prsm_balance_check" in names

    def test_tool_schema_accepts_optional_address(self):
        tool = next(t for t in TOOLS if t.name == "prsm_balance_check")
        assert "address" in tool.inputSchema["properties"]
        # address is optional — schema lists it but does not require it.
        assert "address" not in tool.inputSchema.get("required", [])


# ──────────────────────────────────────────────────────────────────────
# Handler — happy path
# ──────────────────────────────────────────────────────────────────────


class TestBalanceCheckHandler:
    @pytest.mark.asyncio
    async def test_returns_formatted_balance(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert path == "/balance/onchain"
            return {
                "address": "0x" + "11" * 20,
                "balance_wei": 42500000000000000000,
                "balance_ftns": 42.5,
                "usd_rate": 1.0,
                "usd_equivalent": 42.5,
                "source": "onchain",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})
        # Output is formatted text — content checks rather than exact
        # match to allow cosmetic edits.
        assert "42.5" in result
        assert "FTNS" in result
        assert "$42.50" in result or "42.50 USD" in result or "$ 42.50" in result

    @pytest.mark.asyncio
    async def test_passes_address_query_param(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {
                "address": "0x" + "ab" * 20,
                "balance_wei": 1_000_000_000_000_000_000,
                "balance_ftns": 1.0,
                "usd_rate": 1.0,
                "usd_equivalent": 1.0,
                "source": "onchain",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_balance_check({"address": "0x" + "ab" * 20})
        assert "address=0x" + "ab" * 20 in captured["path"]

    @pytest.mark.asyncio
    async def test_renders_address_in_output(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "address": "0xABCDabcdABCDabcdABCDabcdABCDabcdABCDabcd",
                "balance_wei": 0,
                "balance_ftns": 0.0,
                "usd_rate": 1.0,
                "usd_equivalent": 0.0,
                "source": "onchain",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})
        assert "0xABCDabcd" in result  # at least the prefix shows

    @pytest.mark.asyncio
    async def test_renders_usd_rate(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "address": "0x" + "11" * 20,
                "balance_wei": 10**19,
                "balance_ftns": 10.0,
                "usd_rate": 2.5,
                "usd_equivalent": 25.0,
                "source": "onchain",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})
        # USD rate visible so users understand what FTNS→USD conversion
        # was applied.
        assert "2.5" in result


# ──────────────────────────────────────────────────────────────────────
# Handler — error paths
# ──────────────────────────────────────────────────────────────────────


class TestBalanceCheckHandlerErrors:
    @pytest.mark.asyncio
    async def test_handles_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_balance_check({})
        # Must return user-facing error string, not raise.
        assert "cannot reach" in result.lower() or "error" in result.lower()
        assert "node" in result.lower()

    @pytest.mark.asyncio
    async def test_handles_503_response(self):
        # Endpoint returns its 503 detail as a dict from
        # _call_node_api when the underlying node ledger is missing.
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "On-chain ftns_ledger not initialized"}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})
        # No balance fields = degrade gracefully.
        assert (
            "not initialized" in result.lower()
            or "not available" in result.lower()
            or "not configured" in result.lower()
        )


# ──────────────────────────────────────────────────────────────────────
# Aggregate-source rendering (v2 endpoint response shape)
# ──────────────────────────────────────────────────────────────────────


def _v2_response(
    *,
    balance_ftns: float = 10.0,
    claimable_ftns: float = 0.0,
    escrowed_ftns: float = 0.0,
    claimable_available: bool = False,
    escrowed_available: bool = False,
    usd_rate: float = 1.0,
):
    """Build a v2-shape /balance/onchain response."""
    total_ftns = balance_ftns + claimable_ftns + escrowed_ftns
    return {
        "address": "0x" + "11" * 20,
        "balance_wei": int(balance_ftns * 10**18),
        "balance_ftns": balance_ftns,
        "usd_rate": usd_rate,
        "usd_equivalent": balance_ftns * usd_rate,
        "source": "onchain",
        "claimable_royalties_ftns": claimable_ftns,
        "escrowed_ftns": escrowed_ftns,
        "total_ftns": total_ftns,
        "total_usd_equivalent": total_ftns * usd_rate,
        "sources": {
            "onchain": {"available": True, "ftns": balance_ftns},
            "claimable_royalties": {
                "available": claimable_available,
                "ftns": claimable_ftns,
            },
            "escrowed": {
                "available": escrowed_available,
                "ftns": escrowed_ftns,
            },
        },
    }


class TestAggregateRendering:
    @pytest.mark.asyncio
    async def test_renders_aggregate_when_extras_present(self):
        """v2 response with claimable + escrowed wired → multi-line
        breakdown format with per-source lines + total + aggregate
        source label."""
        response = _v2_response(
            balance_ftns=10.0,
            claimable_ftns=2.0,
            escrowed_ftns=3.0,
            claimable_available=True,
            escrowed_available=True,
            usd_rate=1.5,
        )

        async def fake_call_node_api(method, path, data=None):
            return response
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})

        # Aggregate header (distinct from legacy "PRSM Wallet Balance").
        assert "PRSM Wallet Balance (aggregate)" in result
        # Per-source lines.
        assert "On-chain:" in result and "10.000000 FTNS" in result
        assert "Claimable royalties:" in result and "2.000000 FTNS" in result
        assert "Escrowed (pending):" in result and "3.000000 FTNS" in result
        # Total = 15 FTNS; total USD = 15 * 1.5 = 22.50.
        assert "15.000000 FTNS" in result
        assert "$22.50" in result
        # Aggregate source label (not "onchain").
        assert "Source:" in result and "aggregate" in result

    @pytest.mark.asyncio
    async def test_aggregate_marks_sources_unavailable(self):
        """When a source is wired but reports unavailable (e.g., RPC
        flaked), render with `(unavailable)` marker instead of
        treating zero as authoritative."""
        response = _v2_response(
            balance_ftns=10.0,
            claimable_ftns=0.0,
            escrowed_ftns=0.0,
            claimable_available=False,
            escrowed_available=False,
        )
        # Force aggregate path via non-zero contribution from one
        # source — but then mark both as unavailable to verify the
        # marker logic.
        response["claimable_royalties_ftns"] = 5.0
        response["sources"]["claimable_royalties"]["ftns"] = 5.0
        response["total_ftns"] = 15.0

        async def fake_call_node_api(method, path, data=None):
            return response
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})

        # claimable_royalties is unavailable → marker present on its line.
        # escrowed is unavailable → marker present on its line.
        assert "(unavailable)" in result
        # Aggregate header still rendered (extras present via non-zero).
        assert "(aggregate)" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_when_no_extras(self):
        """v2 response with all extras zero AND all unavailable →
        render legacy single-line format (don't pollute the user
        with empty source breakdowns)."""
        response = _v2_response(
            balance_ftns=42.5,
            claimable_ftns=0.0,
            escrowed_ftns=0.0,
            claimable_available=False,
            escrowed_available=False,
        )

        async def fake_call_node_api(method, path, data=None):
            return response
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_balance_check({})

        # Legacy format: header is "PRSM Wallet Balance" — no
        # "(aggregate)" suffix.
        assert "PRSM Wallet Balance" in result
        assert "(aggregate)" not in result
        # No multi-line per-source breakdown.
        assert "Claimable royalties:" not in result
        assert "Escrowed (pending):" not in result
        # Legacy rendering still emits "Source: onchain".
        assert "onchain" in result.lower()
