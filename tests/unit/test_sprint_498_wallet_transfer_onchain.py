"""Sprint 498 — F38: ship missing /wallet/transfer/onchain endpoint.

Sprint 496 runbook anticipated this gap explicitly:

  "If `/wallet/transfer/onchain` doesn't exist yet
   (only `/wallet/transfer/gasless` was indexed), use
   the RoyaltyDistributor's direct path or expose a
   real onchain transfer surface as a sub-deliverable
   of this sprint."

Sprint 498 walked the runbook with a funded active wallet
(0x4acdE458…) on Base mainnet and hit the gap immediately —
TX-1 returned 404 because the endpoint doesn't exist.

This sprint ships the endpoint. It wraps the existing
`OnChainFTNSLedger.transfer()` capability (which already
signs + broadcasts ERC-20 transfers using
FTNS_WALLET_PRIVATE_KEY) behind the canonical
`/wallet/transfer/onchain` HTTP surface.

These pin tests use FastAPI's TestClient against a node
whose `ftns_ledger` is mocked at the boundary — they
verify the HTTP contract, NOT the real broadcast path
(which is exercised by the live runbook walk on Base
mainnet).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_node():
    """Build the FastAPI app wired to a mock node.

    Mirrors the pattern used by other /wallet/* pin tests
    — only attaches the minimum surface the new endpoint
    touches.
    """
    from prsm.node.api import create_api_app

    node = MagicMock()
    # Default: ledger present + initialized
    ledger = MagicMock()
    ledger._connected_address = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    return app, node, ledger


def test_endpoint_returns_503_when_ledger_missing(
    app_with_node,
):
    """If the daemon was started without
    FTNS_WALLET_PRIVATE_KEY (no on-chain ledger),
    /wallet/transfer/onchain must return 503 with an
    actionable hint — not a generic 500."""
    app, node, _ = app_with_node
    node.ftns_ledger = None
    client = TestClient(app)
    r = client.post(
        "/wallet/transfer/onchain",
        json={
            "to_address": (
                "0x4acdE458766C704B2511583572303e77109cFFE8"
            ),
            "amount_ftns": 0.000001,
        },
    )
    assert r.status_code == 503
    detail = r.json().get("detail", "")
    assert "FTNS_WALLET_PRIVATE_KEY" in detail or (
        "ledger" in detail.lower()
    )


def test_endpoint_returns_422_on_missing_to_address(
    app_with_node,
):
    """Empty or missing to_address must surface a clear
    422 (validation error) — not a downstream web3
    failure."""
    app, _, _ = app_with_node
    client = TestClient(app)
    r = client.post(
        "/wallet/transfer/onchain",
        json={"to_address": "", "amount_ftns": 0.000001},
    )
    assert r.status_code == 422


def test_endpoint_returns_422_on_non_positive_amount(
    app_with_node,
):
    """Zero or negative FTNS amount must be rejected
    before signing — never bother building a TX that
    would be a no-op."""
    app, _, _ = app_with_node
    client = TestClient(app)
    r = client.post(
        "/wallet/transfer/onchain",
        json={
            "to_address": (
                "0x4acdE458766C704B2511583572303e77109cFFE8"
            ),
            "amount_ftns": 0,
        },
    )
    assert r.status_code == 422


def test_endpoint_happy_path_returns_tx_hash(
    app_with_node,
):
    """Happy path: ledger.transfer() returns a confirmed
    FTNSTransaction record — endpoint surfaces tx_hash +
    block_number + status."""
    app, _, ledger = app_with_node
    tx_record = MagicMock()
    tx_record.tx_hash = "0x" + "ab" * 32
    tx_record.status = "confirmed"
    tx_record.block_number = 46160100
    tx_record.from_addr = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    tx_record.to_addr = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    tx_record.amount_ftns = 0.000001
    ledger.transfer = AsyncMock(return_value=tx_record)

    client = TestClient(app)
    r = client.post(
        "/wallet/transfer/onchain",
        json={
            "to_address": (
                "0x4acdE458766C704B2511583572303e77109cFFE8"
            ),
            "amount_ftns": 0.000001,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tx_hash"] == tx_record.tx_hash
    assert body["status"] == "confirmed"
    assert body["block_number"] == 46160100
    assert body["from_address"] == tx_record.from_addr
    assert body["to_address"] == tx_record.to_addr


def test_endpoint_surfaces_failed_transfer(
    app_with_node,
):
    """If ledger.transfer returns None (no account
    configured) or status='failed', endpoint must surface
    the failure as 500 with a clear reason — not silently
    return 200 with a missing tx_hash."""
    app, _, ledger = app_with_node
    ledger.transfer = AsyncMock(return_value=None)

    client = TestClient(app)
    r = client.post(
        "/wallet/transfer/onchain",
        json={
            "to_address": (
                "0x4acdE458766C704B2511583572303e77109cFFE8"
            ),
            "amount_ftns": 0.000001,
        },
    )
    assert r.status_code == 500
    assert "transfer" in r.json().get("detail", "").lower()
