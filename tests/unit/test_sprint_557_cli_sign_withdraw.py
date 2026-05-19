"""Sprint 557 — CLI ``wallet sign-withdraw`` helper.

Closes the operator UX side of the user-sig arc. Operators can
drive the full signed-withdraw happy path through the PRSM CLI
without writing Python or talking to MetaMask:

  prsm wallet sign-withdraw \\
      --amount 0.5 --to 0xABCD... \\
      --private-key $PRSM_USER_SIGNING_KEY

The command:
  1. GETs /wallet/deposit/info to read the current wallet_id +
     linked eth_address + next_withdraw_nonce (sprint 557
     extends the response with the last two).
  2. Builds the canonical EIP-712 payload (sprint 555).
  3. Signs locally with the user's private key.
  4. POSTs /wallet/withdraw with signature + nonce + expiry_unix.
  5. Pretty-prints the response (or the 401 detail).

Pin tests cover the endpoint extension + CLI build path. The
end-to-end live flow against a real daemon is verified separately
in the live-verify step at sprint commit time.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from eth_account import Account
from fastapi.testclient import TestClient


# ── deposit/info extension ────────────────────────────────


class _StubLedger:
    def __init__(
        self,
        wallet_id="w1",
        linked_eth=None,
        requires_sig=False,
        nonce=0,
    ):
        self.wallet_id = wallet_id
        self._linked = linked_eth
        self._requires = requires_sig
        self._nonce = nonce

    async def eth_address_for_wallet(self, wallet_id):
        return self._linked if wallet_id == self.wallet_id else None

    async def get_requires_user_signature(self, wallet_id):
        return self._requires if wallet_id == self.wallet_id else False

    async def get_next_withdraw_nonce(self, wallet_id):
        return self._nonce


def _stub_node(ledger, ftns_ledger):
    n = MagicMock()
    n.identity = MagicMock(node_id=ledger.wallet_id)
    n.ledger = ledger
    n.ftns_ledger = ftns_ledger
    return n


class _StubFTNSLedger:
    def __init__(self, address="0xescrow"):
        self._connected_address = address
        self.contract_address = "0xtoken"
        self.chain_id = 8453


def _make_app(ledger):
    from prsm.node.api import create_api_app
    return create_api_app(
        _stub_node(ledger, _StubFTNSLedger()),
        enable_security=False,
    )


def test_deposit_info_surfaces_requires_user_signature_and_nonce():
    """Sprint 557: /wallet/deposit/info gains
    ``requires_user_signature`` + ``next_withdraw_nonce`` fields so
    CLI clients can read everything they need to build a signed
    withdraw in one HTTP call."""
    ledger = _StubLedger(
        wallet_id="w1",
        linked_eth="0x" + "ab" * 20,
        requires_sig=True,
        nonce=7,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.get("/wallet/deposit/info")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["wallet_id"] == "w1"
    assert body["linked_eth_address"] == "0x" + "ab" * 20
    assert body["requires_user_signature"] is True
    assert body["next_withdraw_nonce"] == 7


def test_deposit_info_default_values_when_flag_off():
    """Wallet without flag set → requires_user_signature:False,
    next_withdraw_nonce:0 (the defaults from sprint 554's schema)."""
    ledger = _StubLedger(
        wallet_id="w1",
        linked_eth=None,
        requires_sig=False,
        nonce=0,
    )
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.get("/wallet/deposit/info")
    body = response.json()
    assert body["requires_user_signature"] is False
    assert body["next_withdraw_nonce"] == 0


# ── CLI sign-withdraw build ───────────────────────────────


def test_cli_sign_withdraw_command_registered():
    """`prsm wallet sign-withdraw` is exposed as a CLI command."""
    from prsm.cli import main as cli  # top-level Click group is `main`
    # Click stores sub-groups in commands; introspect to find
    # `wallet sign-withdraw`.
    wallet_group = cli.commands.get("wallet")
    assert wallet_group is not None, "no `wallet` group registered"
    assert "sign-withdraw" in wallet_group.commands, (
        "wallet group missing sign-withdraw command; "
        f"present: {sorted(wallet_group.commands.keys())}"
    )


def test_sign_withdraw_builds_request_body():
    """The CLI's body-builder helper (refactored for testability)
    produces a JSON body with signature/nonce/expiry_unix populated
    correctly given an in-process signing key."""
    from prsm.cli import _build_signed_withdraw_body

    pk = "0x" + "11" * 32
    acct = Account.from_key(pk)

    body = _build_signed_withdraw_body(
        amount_ftns=0.5,
        wallet_id="w1",
        to_eth_address=acct.address,
        nonce=3,
        expiry_unix=1700000000,
        private_key=pk,
    )
    assert body["amount_ftns"] == 0.5
    assert body["wallet_id"] == "w1"
    assert body["to_eth_address"] == acct.address
    assert body["nonce"] == 3
    assert body["expiry_unix"] == 1700000000
    assert body["signature"].startswith("0x")
    assert len(body["signature"]) == 2 + 130  # 0x + 65*2

    # Round-trip: verify with sprint-555's primitive recovers acct
    # (the helper must be using the canonical EIP-712 payload).
    from prsm.economy.withdraw_signature import (
        verify_withdraw_signature,
    )
    payload = {
        "wallet_id": "w1",
        "amount_ftns_wei": int(0.5 * 1e18),
        "to_eth_address": acct.address,
        "nonce": 3,
        "expiry_unix": 1700000000,
    }
    recovered = verify_withdraw_signature(payload, body["signature"])
    assert recovered.lower() == acct.address.lower()


def test_sign_withdraw_default_expiry_in_future():
    """When the caller doesn't pass expiry_unix, the helper defaults
    to now + 300 (5-minute window per sprint-554 user input)."""
    from prsm.cli import _build_signed_withdraw_body

    pk = "0x" + "22" * 32
    acct = Account.from_key(pk)

    before = int(time.time())
    body = _build_signed_withdraw_body(
        amount_ftns=1.0,
        wallet_id="w1",
        to_eth_address=acct.address,
        nonce=0,
        private_key=pk,
    )
    after = int(time.time())
    # Default expiry should be in the (now, now+600) window so the
    # test isn't flaky on slow machines but still catches "did the
    # default actually fire".
    assert body["expiry_unix"] > before
    assert body["expiry_unix"] <= after + 600
