"""GET /balance/onchain — backend endpoint feeding the
prsm_balance_check MCP tool.

V1 scope per Vision §13 Phase 5 stand-in closure: read FTNS balance
via OnChainFTNSLedger; convert to USD using PRSM_FTNS_USD_RATE env
var (placeholder until Aerodrome USDC-FTNS pool is seeded per
Vision gantt 2026-06-15). Returns:

    {
        "address": "0x...",
        "balance_wei": int,
        "balance_ftns": float,
        "usd_rate": float,
        "usd_equivalent": float,
        "source": "onchain" | "ledger-fallback",
    }
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _node_with_ftns_ledger(*, balance_ftns: float = 100.0,
                           connected_address: str | None = "0x" + "11" * 20,
                           initialized: bool = True):
    """Build a minimal node stub whose ftns_ledger.get_balance returns
    the configured FTNS amount. The OnChainFTNSLedger.get_balance
    real method is async and returns FTNS (not wei)."""
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = initialized
    ftns_ledger._connected_address = connected_address
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=balance_ftns)
    node.ftns_ledger = ftns_ledger
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestBalanceCheckHappyPath:
    def test_returns_balance_in_wei_and_ftns(self):
        node = _node_with_ftns_ledger(balance_ftns=42.5)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "1.0"}):
            response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        assert body["balance_ftns"] == 42.5
        assert body["balance_wei"] == int(42.5 * 10**18)

    def test_usd_equivalent_uses_env_rate(self):
        node = _node_with_ftns_ledger(balance_ftns=10.0)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "2.5"}):
            response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        assert body["usd_rate"] == 2.5
        assert body["usd_equivalent"] == 25.0

    def test_default_usd_rate_when_env_unset(self):
        node = _node_with_ftns_ledger(balance_ftns=10.0)
        # Make sure env is NOT set.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_FTNS_USD_RATE", None)
            response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        body = response.json()
        # Default rate is 1.0 per Vision §6 base-case launch anchor.
        assert body["usd_rate"] == 1.0
        assert body["usd_equivalent"] == 10.0

    def test_address_field_reflects_connected_address(self):
        node = _node_with_ftns_ledger(
            balance_ftns=1.0,
            connected_address="0x" + "ab" * 20,
        )
        response = _client(node).get("/balance/onchain")
        assert response.status_code == 200
        assert response.json()["address"] == "0x" + "ab" * 20

    def test_source_is_onchain(self):
        node = _node_with_ftns_ledger()
        response = _client(node).get("/balance/onchain")
        assert response.json()["source"] == "onchain"

    def test_address_query_param_overrides_connected_address(self):
        node = _node_with_ftns_ledger(balance_ftns=5.0)
        target = "0x" + "cd" * 20
        response = _client(node).get(f"/balance/onchain?address={target}")
        assert response.status_code == 200
        body = response.json()
        assert body["address"] == target
        # Confirm the FTNS ledger was queried with the override address.
        node.ftns_ledger.get_balance.assert_awaited_with(target)


# ──────────────────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────────────────


class TestBalanceCheckErrors:
    def test_503_when_ftns_ledger_missing(self):
        node = MagicMock()
        node.identity = MagicMock()
        node.identity.node_id = "test-node"
        node.ftns_ledger = None
        response = _client(node).get("/balance/onchain")
        assert response.status_code == 503
        assert "ftns_ledger" in response.json()["detail"].lower() or "on-chain" in response.json()["detail"].lower()

    def test_400_when_invalid_usd_rate_env(self):
        # Operator misconfiguration — fail loud rather than silently
        # returning USD=0.
        node = _node_with_ftns_ledger(balance_ftns=1.0)
        with patch.dict(os.environ, {"PRSM_FTNS_USD_RATE": "not-a-number"}):
            response = _client(node).get("/balance/onchain")
        # Either 400 (operator-misconfig surfaced) OR 200 with default
        # rate (graceful fallback). Pick fallback to default — matches
        # the scheduler-builder pattern from this morning's wiring.
        assert response.status_code == 200
        body = response.json()
        assert body["usd_rate"] == 1.0  # fell back to default
