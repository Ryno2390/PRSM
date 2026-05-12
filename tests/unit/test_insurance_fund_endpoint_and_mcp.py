"""Sprint 299 — insurance fund HTTP + MCP surface.

GET  /admin/insurance-fund/status            — public; per Vision §14
                                                "public, on-chain verification"
POST /admin/insurance-fund/compose-recovery  — composer-only Safe payload
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.insurance_fund_tracker import (
    ERC20_TRANSFER_SELECTOR,
    InsuranceFundTracker,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_insurance_fund,
)
from prsm.node.api import create_api_app


class _FakeBackend:
    def __init__(self, balances=None):
        self.balances = balances or {}

    def balance_of(self, address):
        return self.balances.get(address, 0)


def _client(tracker=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._insurance_fund_tracker = tracker
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _tracker_at_target():
    return InsuranceFundTracker(
        fund_address="0x" + "ff" * 20,
        treasury_address="0x" + "ee" * 20,
        ftns_token_address="0x" + "11" * 20,
        chain_id=8453,
        backend=_FakeBackend(
            balances={
                "0x" + "ff" * 20: 5_000_000 * (10 ** 18),
                "0x" + "ee" * 20: 100_000_000 * (10 ** 18),
            },
        ),
    )


# ── HTTP: status ─────────────────────────────────────────


def test_status_503_when_unwired():
    resp = _client(None).get(
        "/admin/insurance-fund/status",
    )
    assert resp.status_code == 503


def test_status_returns_full_dict():
    resp = _client(_tracker_at_target()).get(
        "/admin/insurance-fund/status",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fund_balance_wei"] == 5_000_000 * (10 ** 18)
    assert body["reserve_ratio_bps"] == 500
    assert body["target_met"] is True
    assert body["target_bps"] == 500


def test_status_below_target():
    tracker = InsuranceFundTracker(
        fund_address="0xfund",
        treasury_address="0xtreasury",
        backend=_FakeBackend(
            balances={
                "0xfund": 2_000_000 * (10 ** 18),
                "0xtreasury": 100_000_000 * (10 ** 18),
            },
        ),
    )
    resp = _client(tracker).get(
        "/admin/insurance-fund/status",
    )
    body = resp.json()
    assert body["reserve_ratio_bps"] == 200
    assert body["target_met"] is False


# ── HTTP: compose-recovery ───────────────────────────────


def test_compose_503_when_unwired():
    resp = _client(None).post(
        "/admin/insurance-fund/compose-recovery",
        json={
            "recipient": "0x" + "ab" * 20,
            "amount_wei": 100,
            "reason": "test",
        },
    )
    assert resp.status_code == 503


def test_compose_happy_path():
    resp = _client(_tracker_at_target()).post(
        "/admin/insurance-fund/compose-recovery",
        json={
            "recipient": "0x" + "ab" * 20,
            "amount_wei": 1_000_000 * (10 ** 18),
            "reason": "Post-mortem BSR exploit 2026-05-12",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "recovery_transfer"
    assert body["data"].startswith(ERC20_TRANSFER_SELECTOR)
    assert body["recipient"] == "0x" + "ab" * 20
    assert "BSR exploit" in body["reason"]
    assert body["chain_id"] == 8453


def test_compose_422_missing_recipient():
    resp = _client(_tracker_at_target()).post(
        "/admin/insurance-fund/compose-recovery",
        json={"amount_wei": 100, "reason": "x"},
    )
    assert resp.status_code == 422


def test_compose_422_missing_reason():
    resp = _client(_tracker_at_target()).post(
        "/admin/insurance-fund/compose-recovery",
        json={
            "recipient": "0x" + "ab" * 20,
            "amount_wei": 100,
        },
    )
    assert resp.status_code == 422


def test_compose_422_invalid_amount():
    resp = _client(_tracker_at_target()).post(
        "/admin/insurance-fund/compose-recovery",
        json={
            "recipient": "0x" + "ab" * 20,
            "amount_wei": 0,
            "reason": "x",
        },
    )
    assert resp.status_code == 422


def test_compose_422_invalid_recipient_format():
    resp = _client(_tracker_at_target()).post(
        "/admin/insurance-fund/compose-recovery",
        json={
            "recipient": "not-an-address",
            "amount_wei": 100,
            "reason": "x",
        },
    )
    assert resp.status_code == 422


def test_compose_includes_warning_and_instructions():
    resp = _client(_tracker_at_target()).post(
        "/admin/insurance-fund/compose-recovery",
        json={
            "recipient": "0x" + "ab" * 20,
            "amount_wei": 100,
            "reason": "x",
        },
    )
    body = resp.json()
    assert "warning" in body and len(body["warning"]) > 0
    assert "instructions" in body
    assert "destructive" in body["warning"].lower()


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_insurance_fund" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_insurance_fund({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_insurance_fund(
        {"action": "explode"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_status_renders_at_target():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "fund_address": "0xfund",
            "treasury_address": "0xtreasury",
            "fund_balance_wei": 5_000_000 * (10 ** 18),
            "treasury_balance_wei": 100_000_000 * (10 ** 18),
            "reserve_ratio_bps": 500,
            "target_bps": 500,
            "target_met": True,
            "commissioned": True,
            "error": None,
        }),
    ) as mock_call:
        r = await handle_prsm_insurance_fund(
            {"action": "status"},
        )
    args = mock_call.await_args[0]
    assert args[1] == "/admin/insurance-fund/status"
    # Target met marker
    assert "✅" in r or "met" in r.lower()
    # Percentages render — 500 bps = 5.0%
    assert "5.0" in r or "5%" in r


@pytest.mark.asyncio
async def test_mcp_status_renders_below_target():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "fund_address": "0xfund",
            "treasury_address": "0xtreasury",
            "fund_balance_wei": 2_000_000 * (10 ** 18),
            "treasury_balance_wei": 100_000_000 * (10 ** 18),
            "reserve_ratio_bps": 200,
            "target_bps": 500,
            "target_met": False,
            "commissioned": True,
            "error": None,
        }),
    ):
        r = await handle_prsm_insurance_fund(
            {"action": "status"},
        )
    # Below-target warning marker
    assert "⚠" in r or "below" in r.lower()
    assert "2.0" in r or "2%" in r


@pytest.mark.asyncio
async def test_mcp_status_uncommissioned():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "fund_address": None,
            "treasury_address": "0xtreasury",
            "fund_balance_wei": None,
            "treasury_balance_wei": None,
            "reserve_ratio_bps": None,
            "target_bps": 500,
            "target_met": False,
            "commissioned": False,
            "error": None,
        }),
    ):
        r = await handle_prsm_insurance_fund(
            {"action": "status"},
        )
    assert "not configured" in r.lower() or "uncommissioned" in r.lower()


@pytest.mark.asyncio
async def test_mcp_compose_recovery_happy_path():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "action": "recovery_transfer",
            "to": "0x" + "11" * 20,
            "data": ERC20_TRANSFER_SELECTOR + "0" * 128,
            "value": "0",
            "from_fund": "0xfund",
            "recipient": "0x" + "ab" * 20,
            "amount_wei": str(1_000_000 * (10 ** 18)),
            "reason": "BSR exploit recovery",
            "chain_id": 8453,
            "warning": "DESTRUCTIVE: this transfer ...",
            "explorer_url": (
                "https://basescan.org/address/0x"
                + "11" * 20
            ),
            "instructions": "1) Open the Foundation Safe...",
        }),
    ) as mock_call:
        r = await handle_prsm_insurance_fund({
            "action": "compose_recovery",
            "recipient": "0x" + "ab" * 20,
            "amount_wei": 1_000_000 * (10 ** 18),
            "reason": "BSR exploit recovery",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/insurance-fund/compose-recovery"
    # Renders ⚠ block + WARNING + payload + instructions
    assert "WARNING" in r.upper() or "destructive" in r.lower()
    assert "basescan" in r.lower()
    assert "0x" + "ab" * 20 in r  # recipient
    assert "BSR exploit" in r


@pytest.mark.asyncio
async def test_mcp_compose_requires_recipient():
    r = await handle_prsm_insurance_fund({
        "action": "compose_recovery",
        "amount_wei": 100,
        "reason": "x",
    })
    assert "recipient" in r


@pytest.mark.asyncio
async def test_mcp_compose_requires_amount():
    r = await handle_prsm_insurance_fund({
        "action": "compose_recovery",
        "recipient": "0x" + "ab" * 20,
        "reason": "x",
    })
    assert "amount" in r.lower()


@pytest.mark.asyncio
async def test_mcp_compose_requires_reason():
    r = await handle_prsm_insurance_fund({
        "action": "compose_recovery",
        "recipient": "0x" + "ab" * 20,
        "amount_wei": 100,
    })
    assert "reason" in r.lower()


@pytest.mark.asyncio
async def test_mcp_503_message():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "Insurance fund tracker not initialized.",
        }),
    ):
        r = await handle_prsm_insurance_fund(
            {"action": "status"},
        )
    assert (
        "not wired" in r.lower()
        or "not initialized" in r.lower()
    )
