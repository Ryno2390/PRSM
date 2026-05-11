"""Sprint 277 — prsm_gasless_transfer MCP tool.

LLM-facing surface for gasless FTNS transfers. action selector:
quote | execute | status. quote = dry_run; execute = real
sponsored submission. Both gated on PaymasterClient commission.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_gasless_transfer,
)


def test_tool_registered():
    assert "prsm_gasless_transfer" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_gasless_transfer({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_gasless_transfer(
            {"action": "explode"},
        )
        assert "must be" in r.lower()


class TestQuote:
    @pytest.mark.asyncio
    async def test_requires_from_user_id(self):
        r = await handle_prsm_gasless_transfer({
            "action": "quote",
            "to_address": "0xabc",
            "ftns_amount": "10",
        })
        assert "from_user_id" in r

    @pytest.mark.asyncio
    async def test_requires_to_address(self):
        r = await handle_prsm_gasless_transfer({
            "action": "quote",
            "from_user_id": "alice",
            "ftns_amount": "10",
        })
        assert "to_address" in r

    @pytest.mark.asyncio
    async def test_requires_ftns_amount(self):
        r = await handle_prsm_gasless_transfer({
            "action": "quote",
            "from_user_id": "alice",
            "to_address": "0xabc",
        })
        assert "ftns_amount" in r

    @pytest.mark.asyncio
    async def test_quote_sends_dry_run_true(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "ESTIMATED",
                "gas_estimate_wei": 100_000,
                "tx_hash": None,
                "sponsor_amount_wei": None,
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
                "sender_address": "0xsender",
            }),
        ) as mock_call:
            r = await handle_prsm_gasless_transfer({
                "action": "quote",
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
            })
        args = mock_call.await_args[0]
        assert args[0] == "POST"
        assert args[1] == "/wallet/transfer/gasless"
        assert args[2]["dry_run"] is True
        assert "ESTIMATED" in r
        assert "100" in r  # gas estimate

    @pytest.mark.asyncio
    async def test_quote_pending_commission(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "PENDING_COMMISSION",
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
                "tx_hash": None,
            }),
        ):
            r = await handle_prsm_gasless_transfer({
                "action": "quote",
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
            })
        assert "PENDING_COMMISSION" in r


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_sends_dry_run_false(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "SUBMITTED",
                "tx_hash": "0xsuccess",
                "user_op_hash": "0xdead",
                "sponsor_amount_wei": 5_000_000_000_000_000,
                "gas_estimate_wei": 100_000,
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
                "sender_address": "0xsender",
            }),
        ) as mock_call:
            r = await handle_prsm_gasless_transfer({
                "action": "execute",
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
            })
        args = mock_call.await_args[0]
        assert args[2]["dry_run"] is False
        assert "SUBMITTED" in r
        assert "0xsuccess" in r

    @pytest.mark.asyncio
    async def test_execute_failed_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "status": "FAILED",
                "error": "bundler down",
                "tx_hash": None,
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
            }),
        ):
            r = await handle_prsm_gasless_transfer({
                "action": "execute",
                "from_user_id": "alice",
                "to_address": "0xabc",
                "ftns_amount": "10",
            })
        assert "FAILED" in r
        assert "bundler down" in r

    @pytest.mark.asyncio
    async def test_execute_404_unknown_sender(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "no WaaS wallet for from_user_id='ghost'",
            }),
        ):
            r = await handle_prsm_gasless_transfer({
                "action": "execute",
                "from_user_id": "ghost",
                "to_address": "0xabc",
                "ftns_amount": "10",
            })
        assert "no waas wallet" in r.lower()


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_uncommissioned(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "commissioned": False,
                "sponsorships": 0,
                "total_sponsored_wei": 0,
                "endpoint": None,
                "policy_id": None,
            }),
        ) as mock_call:
            r = await handle_prsm_gasless_transfer(
                {"action": "status"},
            )
        args = mock_call.await_args[0]
        assert args[1] == "/wallet/paymaster/status"
        assert "PENDING_COMMISSION" in r or "False" in r

    @pytest.mark.asyncio
    async def test_status_commissioned_with_spend(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "commissioned": True,
                "sponsorships": 3,
                "total_sponsored_wei": 15_000_000_000_000_000,
                "endpoint": "https://paymaster.example",
                "policy_id": "policy-abc",
            }),
        ):
            r = await handle_prsm_gasless_transfer(
                {"action": "status"},
            )
        assert "3" in r
        assert "commissioned" in r.lower()
