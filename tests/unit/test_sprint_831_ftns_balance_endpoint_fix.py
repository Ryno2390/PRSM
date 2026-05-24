"""Sprint 831 — F29 fix: `prsm ftns balance` hits working endpoint.

Sprint 830's audit surfaced 17 unmounted routers in production.
Sprint 831 closes the FIRST cascade-effect from that finding:
the CLI's `prsm ftns balance` command targeted
/api/v1/ftns/balance (ftns_api router, allow-listed inert per
sprint 830), so every operator running the command saw a 404.

Sprint 831 switches the CLI to the inline /balance endpoint
(node/api.py:2266), which is the operationally-correct production
surface. Response shape differs from FTNSBalanceResponse — the
inline endpoint returns {wallet_id, balance, recent_transactions}
rather than {user_id, balance, locked_balance, available_balance}.
The CLI is updated to match.

Pin tests:
- CLI POSTs to /balance, NOT /api/v1/ftns/balance (F29
  regression guard).
- Successful response renders the balance.
- 503 surfaces actionable "ledger not initialized" message.
- ConnectError surfaces actionable "start server" message.

Live-attested 2026-05-24 against mock-executor daemon: command
returns "Balance: 100.000000 FTNS" (bootstrap balance) + 1
recent transaction. Pre-831 the command 404'd.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner


def _invoke(args=None):
    from prsm.cli import ftns as _ftns_group
    return CliRunner().invoke(
        _ftns_group, ["balance"] + (args or []),
    )


def _balance_ok():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "wallet_id": "node-abc",
        "balance": 100.0,
        "recent_transactions": [
            {"tx_id": "t1", "amount": 1.0},
        ],
    }
    return r


# ---- URL regression guard (F29 lesson) -----------------------


def test_balance_command_hits_inline_balance_endpoint():
    """Sprint 831 regression guard: F29 — pre-831 this command
    hit the legacy /api/v1/ftns/balance route which is NOT
    mounted on the production daemon. The fix MUST target the
    inline /balance endpoint or the bug returns."""
    with patch("httpx.get", return_value=_balance_ok()) as mg:
        result = _invoke(["--api-url", "http://node:8000"])
    assert result.exit_code == 0, result.output
    call_url = mg.call_args.args[0]
    assert call_url == "http://node:8000/balance", (
        f"Sprint 831 regressed — CLI now targets {call_url!r}. "
        f"Must hit /balance, not /api/v1/ftns/balance."
    )


def test_balance_command_does_not_hit_phantom_url():
    """Explicit guard: /api/v1/ftns/balance MUST NOT appear in
    the GET URL. Sprint 830 documents this prefix as inert."""
    with patch("httpx.get", return_value=_balance_ok()) as mg:
        _invoke(["--api-url", "http://node:8000"])
    call_url = mg.call_args.args[0]
    assert "/api/v1/ftns/balance" not in call_url


# ---- Successful response rendering ---------------------------


def test_balance_renders_inline_shape():
    """Inline /balance returns {wallet_id, balance,
    recent_transactions}; CLI MUST render against that shape."""
    with patch("httpx.get", return_value=_balance_ok()):
        result = _invoke(["--api-url", "http://node:8000"])
    assert result.exit_code == 0
    assert "100.000000" in result.output
    assert "node-abc" in result.output


# ---- Actionable error paths ----------------------------------


def test_balance_503_surfaces_actionable_message():
    """503 from /balance (ledger not initialized) must produce
    actionable operator guidance, not a cryptic HTTP code."""
    bad = MagicMock()
    bad.status_code = 503
    bad.text = '{"detail":"Node not initialized"}'
    with patch("httpx.get", return_value=bad):
        result = _invoke(["--api-url", "http://node:8000"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "ledger not initialized" in flat
    assert "node start" in flat


def test_balance_connect_error_surfaces_start_hint():
    """ConnectError must point operators at `prsm node start`,
    not the legacy `prsm serve` (which doesn't exist anymore)."""
    with patch(
        "httpx.get",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = _invoke(["--api-url", "http://node:8000"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "node start" in flat
