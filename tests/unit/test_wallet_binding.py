"""Unit tests for prsm.interface.onboarding.wallet_binding.

Covers Task 2 matrix from docs/2026-04-22-phase4-wallet-sdk-design-plan.md §6:

  - new-user sign_in creates node_id + returns is_new_user=True
  - returning-user sign_in fetches existing binding
  - bind() stores IdentityBinding on valid signature
  - bind() is idempotent (second call returns same record)
  - bind() rejects wrong signature
  - bind() rejects when node_id already bound to a different wallet
  - query by wallet_address works
  - query by node_id works
  - SqliteWalletBindingStore round-trip

Plus the in-memory store as a control for the sqlite variant.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_utils import to_checksum_address

from prsm.interface.onboarding.wallet_binding import (
    BindingConflictError,
    BindingSignatureError,
    IdentityBinding,
    InMemoryWalletBindingStore,
    SqliteWalletBindingStore,
    WalletBindingService,
    build_binding_message,
)

ISSUED_AT = "2026-04-22T12:00:00Z"


def _account(tag: str = "a"):
    # Two distinct deterministic accounts for multi-wallet tests.
    if tag == "a":
        return Account.from_key("0x" + "ab" * 31 + "cd")
    if tag == "b":
        return Account.from_key("0x" + "de" * 31 + "ad")
    raise ValueError(tag)


def _sign_binding(acct, wallet_address: str, node_id_hex: str, issued_at: str = ISSUED_AT) -> str:
    message = build_binding_message(wallet_address, node_id_hex, issued_at)
    signed = acct.sign_message(encode_defunct(text=message))
    return signed.signature.hex()


@pytest.fixture
def service():
    return WalletBindingService(store=InMemoryWalletBindingStore())


# --- sign_in -----------------------------------------------------------------


def test_sign_in_new_user_returns_fresh_node_id(service):
    wallet = _account("a").address
    node_id, is_new_user = service.sign_in(wallet)
    assert is_new_user is True
    assert len(node_id) == 32  # 32-char hex per node.identity.generate_node_identity


def test_sign_in_returning_user_returns_existing_binding(service):
    acct = _account("a")
    node_id, _ = service.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)
    service.bind(acct.address, node_id, sig, ISSUED_AT)

    # Second sign_in for same wallet returns the already-bound node_id.
    node_id_again, is_new_user = service.sign_in(acct.address)
    assert is_new_user is False
    assert node_id_again == node_id


# --- bind --------------------------------------------------------------------


def test_bind_stores_record_on_valid_signature(service):
    acct = _account("a")
    node_id, _ = service.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)

    binding = service.bind(acct.address, node_id, sig, ISSUED_AT, now_unix=1713787200)

    assert isinstance(binding, IdentityBinding)
    assert binding.wallet_address == to_checksum_address(acct.address)
    assert binding.node_id_hex == node_id
    assert binding.bound_at_unix == 1713787200
    assert binding.wallet_signature == sig
    assert binding.signing_message_hash.startswith("0x")
    assert len(binding.signing_message_hash) == 66  # 0x + 64 hex


def test_bind_is_idempotent(service):
    acct = _account("a")
    node_id, _ = service.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)

    first = service.bind(acct.address, node_id, sig, ISSUED_AT)
    second = service.bind(acct.address, node_id, sig, ISSUED_AT)

    assert first == second


def test_bind_rejects_wrong_signature(service):
    acct_a = _account("a")
    acct_b = _account("b")
    node_id, _ = service.sign_in(acct_a.address)

    # B signs a binding for A's wallet — must not recover to A.
    bad_sig = _sign_binding(acct_b, acct_a.address, node_id)

    with pytest.raises(BindingSignatureError):
        service.bind(acct_a.address, node_id, bad_sig, ISSUED_AT)


def test_bind_rejects_node_already_bound_to_other_wallet(service):
    acct_a = _account("a")
    acct_b = _account("b")
    node_id, _ = service.sign_in(acct_a.address)

    # A binds the node.
    sig_a = _sign_binding(acct_a, acct_a.address, node_id)
    service.bind(acct_a.address, node_id, sig_a, ISSUED_AT)

    # B attempts to bind the same node to B's wallet — must be rejected.
    sig_b = _sign_binding(acct_b, acct_b.address, node_id)
    with pytest.raises(BindingConflictError):
        service.bind(acct_b.address, node_id, sig_b, ISSUED_AT)


def test_bind_rejects_wallet_rebinding_different_node(service):
    acct = _account("a")
    node_id_1, _ = service.sign_in(acct.address)
    sig_1 = _sign_binding(acct, acct.address, node_id_1)
    service.bind(acct.address, node_id_1, sig_1, ISSUED_AT)

    # Same wallet attempts to bind to a new node — conflict (wallet already bound).
    node_id_2 = "f" * 32
    sig_2 = _sign_binding(acct, acct.address, node_id_2)
    with pytest.raises(BindingConflictError):
        service.bind(acct.address, node_id_2, sig_2, ISSUED_AT)


# --- queries -----------------------------------------------------------------


def test_get_by_wallet(service):
    acct = _account("a")
    node_id, _ = service.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)
    service.bind(acct.address, node_id, sig, ISSUED_AT)

    found = service.get_by_wallet(acct.address)
    assert found is not None
    assert found.node_id_hex == node_id


def test_get_by_node_id(service):
    acct = _account("a")
    node_id, _ = service.sign_in(acct.address)
    sig = _sign_binding(acct, acct.address, node_id)
    service.bind(acct.address, node_id, sig, ISSUED_AT)

    found = service.get_by_node_id(node_id)
    assert found is not None
    assert found.wallet_address.lower() == acct.address.lower()


def test_get_returns_none_for_unknown(service):
    assert service.get_by_wallet("0x" + "00" * 20) is None
    assert service.get_by_node_id("0" * 32) is None


# --- SqliteWalletBindingStore -------------------------------------------------


def test_sqlite_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "bindings.sqlite"
        store = SqliteWalletBindingStore(db_path)

        acct = _account("a")
        service = WalletBindingService(store=store)
        node_id, _ = service.sign_in(acct.address)
        sig = _sign_binding(acct, acct.address, node_id)
        binding = service.bind(acct.address, node_id, sig, ISSUED_AT, now_unix=1713787200)

        # New store instance against same file — must reload the binding.
        store2 = SqliteWalletBindingStore(db_path)
        found = store2.get_by_wallet(acct.address)
        assert found == binding

        found_by_node = store2.get_by_node_id(node_id)
        assert found_by_node == binding
