"""Sprint 505 — on-chain TX stats surface.

Sprint 500/501 ship the raw TX list + persistence. Operators
still have to compute aggregates manually (total sent, pending
count, etc.). Sprint 505 ships a dedicated stats endpoint +
CLI flag.

GET /wallet/transactions/onchain/stats returns:
  {
    address,
    total_count,
    confirmed_count, pending_count, rejected_count,
    total_ftns_sent,      (sum amount_ftns where status=confirmed)
    first_tx_at,          (min created_at, null if empty)
    last_tx_at,           (max created_at, null if empty)
    scope                 (mirrors history endpoint)
  }

CLI: `prsm ftns history --onchain --stats` prints a compact
summary instead of the full table.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from fastapi.testclient import TestClient


def _build_app(transactions=None, has_ledger=True):
    from prsm.node.api import create_api_app

    node = MagicMock()
    if has_ledger:
        ledger = MagicMock()
        ledger._connected_address = "0xAAAA"
        ledger._transactions = transactions or []
        ledger.is_persistent = False
        ledger.db_path = None
        node.ftns_ledger = ledger
    else:
        node.ftns_ledger = None
    return create_api_app(node, enable_security=False)


def _tx(status, amount, when):
    m = MagicMock()
    m.status = status
    m.amount_ftns = amount
    m.created_at = when
    return m


def test_stats_503_when_ledger_missing():
    app = _build_app(has_ledger=False)
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain/stats")
    assert r.status_code == 503


def test_stats_empty_state():
    """No TX → zeros + nulls, not 404."""
    app = _build_app(transactions=[])
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total_count"] == 0
    assert body["confirmed_count"] == 0
    assert body["pending_count"] == 0
    assert body["rejected_count"] == 0
    assert body["total_ftns_sent"] == 0.0
    assert body["first_tx_at"] is None
    assert body["last_tx_at"] is None


def test_stats_aggregates_correctly():
    """Mixed-status TX list → correct counts + sums."""
    txs = [
        _tx("confirmed", 1.0, 1700000100.0),
        _tx("confirmed", 0.5, 1700000200.0),
        _tx("pending",   2.0, 1700000300.0),
        _tx("rejected",  9.9, 1700000400.0),
        _tx("confirmed", 0.25, 1700000500.0),
    ]
    app = _build_app(transactions=txs)
    client = TestClient(app)
    body = client.get(
        "/wallet/transactions/onchain/stats"
    ).json()
    assert body["total_count"] == 5
    assert body["confirmed_count"] == 3
    assert body["pending_count"] == 1
    assert body["rejected_count"] == 1
    # confirmed amounts only: 1.0 + 0.5 + 0.25 = 1.75
    assert body["total_ftns_sent"] == 1.75
    assert body["first_tx_at"] == 1700000100.0
    assert body["last_tx_at"] == 1700000500.0


def test_stats_address_passthrough():
    """Address field mirrors history endpoint for consistency."""
    app = _build_app(transactions=[])
    client = TestClient(app)
    body = client.get(
        "/wallet/transactions/onchain/stats"
    ).json()
    assert body["address"] == "0xAAAA"


# ── CLI ────────────────────────────────────────────────


def test_cli_stats_flag_exists():
    """`prsm ftns history --onchain --stats` must be a
    valid flag combination."""
    from prsm.cli import main as cli
    runner = CliRunner()
    result = runner.invoke(
        cli, ["ftns", "history", "--help"],
    )
    assert result.exit_code == 0
    assert "--stats" in result.output


def test_cli_stats_renders_summary():
    """--stats hits the new endpoint and renders compact
    summary (not the full Rich table)."""
    from prsm.cli import main as cli

    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "address": "0x4acdE458766C704B2511583572303e77109cFFE8",
        "total_count": 5,
        "confirmed_count": 3,
        "pending_count": 1,
        "rejected_count": 1,
        "total_ftns_sent": 1.75,
        "first_tx_at": 1700000100.0,
        "last_tx_at": 1700000500.0,
        "scope": (
            "persistent (sqlite: ~/.prsm/onchain_tx.db)"
        ),
    }
    with patch("httpx.get", return_value=mock):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "ftns", "history",
                "--onchain", "--stats",
                "--api-url", "http://127.0.0.1:8000",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "5" in result.output  # total_count
    assert "1.75" in result.output  # total_ftns_sent
    assert "confirmed" in result.output.lower()
