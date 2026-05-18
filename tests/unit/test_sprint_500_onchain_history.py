"""Sprint 500 — on-chain TX history surface.

Sprint 498 shipped /wallet/transfer/onchain (F38). Sprint 499
wrapped it in `prsm ftns transfer-onchain`. Sprint 500 closes
the audit-trail loop: operators need to see the on-chain TX
their daemon has broadcast.

Existing `prsm ftns history` only queries the off-chain DAG
ledger (user-to-user FTNS movement). On-chain TX go into
`OnChainFTNSLedger._transactions` (an in-memory list) but
have no HTTP surface — operators can only see them by
grepping daemon logs or hitting BaseScan.

Sprint 500 ships:
  1. GET /wallet/transactions/onchain → list of {tx_hash,
     status, block_number, from, to, amount_ftns,
     created_at}
  2. `prsm ftns history --onchain` flag flips the existing
     CLI to query the new endpoint and render a Rich table

Honest-scope: current-session-only (in-memory list resets on
daemon restart). Persistence is a sprint-501 follow-on if
needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient


# ── Backend endpoint tests ──────────────────────────────


@pytest.fixture
def app_with_ledger():
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    # Sprint 501 added these attributes — explicit defaults
    # for the legacy sprint-500 honest-scope check.
    ledger.is_persistent = False
    ledger.db_path = None
    # _transactions is the in-memory list maintained by
    # OnChainFTNSLedger.transfer()
    tx1 = MagicMock()
    tx1.tx_hash = "0x" + "11" * 32
    tx1.status = "confirmed"
    tx1.block_number = 46160224
    tx1.from_addr = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    tx1.to_addr = tx1.from_addr
    tx1.amount_ftns = 0.000001
    tx1.created_at = 1747560000.0
    tx1.job_id = "manual-aaa"

    tx2 = MagicMock()
    tx2.tx_hash = "0x" + "22" * 32
    tx2.status = "confirmed"
    tx2.block_number = 46160279
    tx2.from_addr = tx1.from_addr
    tx2.to_addr = (
        "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
    )
    tx2.amount_ftns = 1.0
    tx2.created_at = 1747560050.0
    tx2.job_id = "manual-bbb"

    ledger._transactions = [tx1, tx2]
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    return app, node, ledger


def test_endpoint_returns_503_when_ledger_missing(
    app_with_ledger,
):
    """Same 503 contract as /wallet/transfer/onchain
    — actionable hint pointing to FTNS_WALLET_PRIVATE_KEY."""
    app, node, _ = app_with_ledger
    node.ftns_ledger = None
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain")
    assert r.status_code == 503
    detail = r.json().get("detail", "")
    assert "FTNS_WALLET_PRIVATE_KEY" in detail or (
        "ledger" in detail.lower()
    )


def test_endpoint_returns_session_transactions(
    app_with_ledger,
):
    """Endpoint must return the full list of in-memory
    transactions with canonical schema."""
    app, _, _ = app_with_ledger
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain")
    assert r.status_code == 200
    body = r.json()
    assert "transactions" in body
    assert "count" in body
    assert "connected_address" in body
    assert body["count"] == 2
    assert len(body["transactions"]) == 2

    t0 = body["transactions"][0]
    for field in (
        "tx_hash", "status", "block_number",
        "from_address", "to_address", "amount_ftns",
        "created_at", "job_id",
    ):
        assert field in t0, (
            f"transaction field missing: {field}"
        )
    assert t0["tx_hash"] == "0x" + "11" * 32
    assert t0["block_number"] == 46160224
    assert t0["amount_ftns"] == 0.000001


def test_endpoint_returns_empty_list_when_no_transactions(
    app_with_ledger,
):
    """Fresh daemon (no transfers yet) must return clean
    empty-state, NOT 404 or 500."""
    app, _, ledger = app_with_ledger
    ledger._transactions = []
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["transactions"] == []


def test_endpoint_honest_scope_in_memory_documented(
    app_with_ledger,
):
    """Response must include the honest-scope note that
    this is session-scoped (resets on daemon restart).
    Operators planning long-term audits need to know."""
    app, _, _ = app_with_ledger
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain")
    body = r.json()
    assert "scope" in body
    assert "in-memory" in body["scope"].lower() or (
        "session" in body["scope"].lower()
    )


# ── CLI tests ───────────────────────────────────────────


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_onchain_flag_exists(runner):
    """`prsm ftns history --onchain` flag must exist."""
    from prsm.cli import main as cli

    result = runner.invoke(
        cli, ["ftns", "history", "--help"],
    )
    assert result.exit_code == 0
    assert "--onchain" in result.output


def test_cli_onchain_renders_transactions(runner):
    """When --onchain set, CLI hits the new endpoint and
    renders a table with the key columns."""
    from prsm.cli import main as cli

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "count": 2,
        "connected_address": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "scope": "in-memory (resets on daemon restart)",
        "transactions": [
            {
                "tx_hash": "0x" + "ab" * 32,
                "status": "confirmed",
                "block_number": 46160224,
                "from_address": (
                    "0x4acdE458766C704B2511583572303e77109cFFE8"
                ),
                "to_address": (
                    "0x4acdE458766C704B2511583572303e77109cFFE8"
                ),
                "amount_ftns": 0.000001,
                "created_at": 1747560000.0,
                "job_id": "manual-aaa",
            },
        ],
    }

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            cli,
            [
                "ftns", "history", "--onchain",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "46160224" in result.output
    assert "confirmed" in result.output
    # tx_hash truncated in table is ok; just need to see prefix
    assert "0xab" in result.output


def test_cli_onchain_empty_state(runner):
    """Empty list must render a clean empty-state
    message, not crash on an empty Rich table."""
    from prsm.cli import main as cli

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "count": 0,
        "connected_address": (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        ),
        "scope": "in-memory (resets on daemon restart)",
        "transactions": [],
    }

    with patch("httpx.get", return_value=mock_response):
        result = runner.invoke(
            cli,
            [
                "ftns", "history", "--onchain",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0
    assert "0" in result.output  # count
    # Some empty-state acknowledgement
    assert (
        "no" in result.output.lower()
        or "empty" in result.output.lower()
        or "0 " in result.output
    )
