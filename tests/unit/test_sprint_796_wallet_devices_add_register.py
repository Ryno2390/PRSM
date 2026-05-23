"""Sprint 796 — `wallet devices add --register` closes the operator round-trip.

Pre-796 `wallet devices add` minted a delegation blob and
printed it to stdout. Operators then had to either:
  1. Copy the JSON to a file + export PRSM_OPERATOR_DELEGATION
     on the target device (single-device path), AND/OR
  2. Manually POST the JSON to /api/v1/auth/wallet/bind to
     register with the daemon's binding store so it shows up
     in `wallet devices list`.

Step 2 is consistently missed. The round-trip is broken: every
operator who runs `wallet devices add` sees their `wallet
devices list` stay empty, then has to debug why.

Sprint 796 closes the round-trip with a single new flag:

  prsm wallet devices add --node-id <hex> --register
                          [--api-url <url>]
                          [--format text|json]

When --register is set, after minting the delegation the CLI
POSTs to /api/v1/auth/wallet/bind with the right schema:
  {wallet_address, node_id_hex, signature, issued_at}

Errors:
  - daemon unreachable → exit 2 + delegation still printed
    (operator can retry register later without losing the sig)
  - 409 conflict (node already bound to different wallet) →
    exit 1 with the actionable conflict message
  - other non-200 → exit 1 with status + detail

Without --register, NOTHING is sent (sprint 789 behavior
preserved).

Pin tests:
- --register flag exists on the add command.
- Without --register: no httpx.post call made.
- With --register: POST to /bind with correct body shape.
- Body fields match WalletBindRequest (issued_at, not
  issued_at_iso — the wire format).
- Happy path returns 200 → exit 0 + "registered" text in output.
- Unreachable daemon → exit 2 + delegation still printed
  (so operator can retry).
- 409 conflict → exit 1 + conflict message surfaced.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _invoke(args, env=None):
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    return runner.invoke(
        _wallet_group, ["devices"] + list(args), env=env or {},
    )


def _make_pk():
    from eth_account import Account
    return Account.create()


# ---- Flag exists, default off ----------------------------------


def test_register_flag_exists():
    """--help on add includes --register."""
    from prsm.cli import wallet as _wallet_group
    runner = CliRunner()
    devices = _wallet_group.commands["devices"]
    add_cmd = devices.commands["add"]
    result = runner.invoke(add_cmd, ["--help"])
    assert result.exit_code == 0
    assert "--register" in result.output


def test_add_without_register_does_not_post():
    """No --register flag → no daemon call."""
    acct = _make_pk()
    with patch("httpx.post") as mock_post:
        result = _invoke(
            ["add", "--node-id", "a" * 32, "--format", "json"],
            env={"PRIVATE_KEY": acct.key.to_0x_hex()},
        )
    assert result.exit_code == 0
    mock_post.assert_not_called()


# ---- Happy path with --register --------------------------------


def test_add_register_posts_correct_body():
    """With --register, CLI POSTs to /api/v1/auth/wallet/bind
    with the correct WalletBindRequest schema."""
    acct = _make_pk()
    node_id = "a" * 32

    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "wallet_address": acct.address,
        "node_id_hex": node_id,
        "bound_at_unix": 12345,
        "signing_message_hash": "0xhash",
    }
    with patch("httpx.post", return_value=fake) as mock_post:
        result = _invoke(
            [
                "add", "--node-id", node_id,
                "--register",
                "--format", "json",
            ],
            env={"PRIVATE_KEY": acct.key.to_0x_hex()},
        )
    assert result.exit_code == 0, result.output
    mock_post.assert_called_once()

    # The call hit /api/v1/auth/wallet/bind
    call_args = mock_post.call_args
    url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
    assert "/api/v1/auth/wallet/bind" in url

    # Body shape — must use issued_at (the wire field), NOT
    # issued_at_iso (which is the delegation-blob field).
    body = call_args.kwargs.get("json") or {}
    assert body.get("wallet_address") == acct.address
    assert body.get("node_id_hex") == node_id
    assert "signature" in body
    assert body["signature"].startswith("0x")
    assert "issued_at" in body
    assert "issued_at_iso" not in body  # wire format check


def test_add_register_text_mode_shows_success():
    """Happy-path text output includes operator-facing
    "registered" or similar success signal."""
    acct = _make_pk()
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "wallet_address": acct.address,
        "node_id_hex": "b" * 32,
        "bound_at_unix": 1,
        "signing_message_hash": "0xh",
    }
    with patch("httpx.post", return_value=fake):
        result = _invoke(
            [
                "add", "--node-id", "b" * 32,
                "--register", "--format", "text",
            ],
            env={"PRIVATE_KEY": acct.key.to_0x_hex()},
        )
    assert result.exit_code == 0
    out = result.output.lower()
    assert "registered" in out or "bound" in out or "recorded" in out


# ---- Error paths ----------------------------------------------


def test_add_register_unreachable_daemon_exits_2():
    """Daemon unreachable → exit 2 + delegation still printed so
    operator can retry register without losing the signed blob."""
    acct = _make_pk()
    node_id = "c" * 32
    with patch(
        "httpx.post",
        side_effect=ConnectionError("connection refused"),
    ):
        result = _invoke(
            [
                "add", "--node-id", node_id,
                "--register", "--format", "text",
            ],
            env={"PRIVATE_KEY": acct.key.to_0x_hex()},
        )
    assert result.exit_code == 2
    # The signed blob MUST still appear so the operator can
    # save it locally and retry register later.
    assert node_id in result.output
    assert "signature" in result.output


def test_add_register_409_conflict_exits_1():
    """Daemon returns 409 (node already bound to a different
    wallet — operator-collision defense). CLI surfaces the
    detail + exits 1."""
    acct = _make_pk()
    fake = MagicMock()
    fake.status_code = 409
    fake.text = (
        '{"detail":{"error":"binding_conflict",'
        '"message":"node ... already bound to wallet ..."}}'
    )
    with patch("httpx.post", return_value=fake):
        result = _invoke(
            [
                "add", "--node-id", "d" * 32,
                "--register", "--format", "text",
            ],
            env={"PRIVATE_KEY": acct.key.to_0x_hex()},
        )
    assert result.exit_code == 1
    assert "409" in result.output or "conflict" in result.output.lower()


def test_add_register_json_mode_returns_full_payload():
    """JSON mode: emit both the minted delegation AND the
    registration response, so callers can chain."""
    acct = _make_pk()
    node_id = "e" * 32
    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "wallet_address": acct.address,
        "node_id_hex": node_id,
        "bound_at_unix": 9999,
        "signing_message_hash": "0xh",
    }
    with patch("httpx.post", return_value=fake):
        result = _invoke(
            [
                "add", "--node-id", node_id,
                "--register", "--format", "json",
            ],
            env={"PRIVATE_KEY": acct.key.to_0x_hex()},
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    # Must include both: the delegation we minted + the binding
    # response from the daemon
    assert "delegation" in data
    assert "registration" in data
    assert data["delegation"]["node_id_hex"] == node_id
    assert data["registration"]["bound_at_unix"] == 9999
