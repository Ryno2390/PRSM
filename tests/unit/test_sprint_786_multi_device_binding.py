"""Sprint 786 — multi-device operator account: lift WalletBindingService
1:1 invariant to 1:N (one wallet can bind multiple devices/node_ids).

Pre-786 the WalletBindingService enforced strict 1:1 — a wallet
bound to one node_id couldn't bind another. That made sense for
the Phase 4 onboarding flow (each user has one device). For
multi-device operators (laptop + desktop + cloud spot under
one ETH wallet), this becomes the blocker.

Sprint 786 changes:
- Store gets `get_all_by_wallet(wallet) -> List[IdentityBinding]`.
- InMemoryWalletBindingStore: many bindings per wallet supported.
- Service.bind(): no conflict when wallet already has a different
  binding — that's expected for multi-device. A conflict is still
  raised when the SAME node_id is bound to a different wallet
  (node can't be stolen by another operator).
- get_by_wallet() returns the FIRST binding for back-compat with
  callers that expect Optional[IdentityBinding] (deprecated path;
  new callers use get_all_by_wallet).

Pin tests:
- get_all_by_wallet method exists on Protocol + both stores.
- InMemoryStore: bind two different node_ids to same wallet → both retrievable.
- Service.bind: same wallet + different node_id → no conflict.
- Service.bind: same node_id + different wallet → STILL raises
  BindingConflictError (node-side uniqueness is load-bearing —
  prevents operator collision attacks).
- get_by_wallet still returns Optional (first binding) for
  back-compat.
- Bound node_ids list ordered by bound_at_unix (oldest first).

Sprint 787 will ship the on-disk migration for Sqlite stores
(schema change from wallet PRIMARY KEY to composite key). 786
covers the protocol + in-memory + service-layer change.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


def _make_identity():
    from prsm.node.identity import generate_node_identity
    return generate_node_identity("test")


def _sign_binding(identity, wallet_address: str, wallet_privkey_hex: str):
    """Build a real EIP-191 wallet signature over the canonical
    binding message for the given identity."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    from prsm.interface.onboarding.wallet_binding import (
        build_binding_message,
    )
    issued_at_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    msg = build_binding_message(
        wallet_address, identity.node_id, issued_at_iso,
    )
    encoded = encode_defunct(text=msg)
    signed = Account.from_key(wallet_privkey_hex).sign_message(encoded)
    return signed.signature.to_0x_hex(), issued_at_iso


# ---- Protocol + store: get_all_by_wallet ------------------------


def test_protocol_has_get_all_by_wallet():
    """The WalletBindingStore Protocol declares get_all_by_wallet."""
    from prsm.interface.onboarding import wallet_binding as _wb
    assert hasattr(_wb.WalletBindingStore, "get_all_by_wallet")


def test_in_memory_store_returns_empty_list_for_unknown_wallet():
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
    )
    s = InMemoryWalletBindingStore()
    assert s.get_all_by_wallet("0x" + "1" * 40) == []


def test_in_memory_store_supports_multiple_bindings_per_wallet():
    """Insert two bindings with same wallet + different node_ids."""
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
        IdentityBinding,
    )
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "a" * 40)
    s = InMemoryWalletBindingStore()
    b1 = IdentityBinding(
        wallet_address=wallet,
        node_id_hex="a" * 32,
        bound_at_unix=1000,
        wallet_signature="0xsig1",
        signing_message_hash="0xh1",
    )
    b2 = IdentityBinding(
        wallet_address=wallet,
        node_id_hex="b" * 32,
        bound_at_unix=2000,
        wallet_signature="0xsig2",
        signing_message_hash="0xh2",
    )
    s.insert(b1)
    s.insert(b2)
    bindings = s.get_all_by_wallet(wallet)
    assert len(bindings) == 2
    node_ids = {b.node_id_hex for b in bindings}
    assert node_ids == {"a" * 32, "b" * 32}


def test_get_all_by_wallet_ordered_by_bound_at_unix():
    """Sprint 786: oldest-first ordering for stable enumeration."""
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
        IdentityBinding,
    )
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "c" * 40)
    s = InMemoryWalletBindingStore()
    # Insert out-of-order
    s.insert(IdentityBinding(
        wallet_address=wallet, node_id_hex="2" * 32,
        bound_at_unix=2000, wallet_signature="0x", signing_message_hash="0x",
    ))
    s.insert(IdentityBinding(
        wallet_address=wallet, node_id_hex="1" * 32,
        bound_at_unix=1000, wallet_signature="0x", signing_message_hash="0x",
    ))
    s.insert(IdentityBinding(
        wallet_address=wallet, node_id_hex="3" * 32,
        bound_at_unix=3000, wallet_signature="0x", signing_message_hash="0x",
    ))
    bindings = s.get_all_by_wallet(wallet)
    assert [b.bound_at_unix for b in bindings] == [1000, 2000, 3000]


# ---- Service: bind() no-conflict on multi-device --------------


def test_service_bind_two_devices_same_wallet_no_conflict():
    """Bind two different node_ids to the same wallet — both succeed."""
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
        WalletBindingService,
    )
    from eth_account import Account

    acct = Account.create()
    wallet = acct.address
    pk_hex = acct.key.to_0x_hex()

    store = InMemoryWalletBindingStore()
    svc = WalletBindingService(store=store)

    ident1 = _make_identity()
    sig1, iso1 = _sign_binding(ident1, wallet, pk_hex)
    b1 = svc.bind(
        wallet_address=wallet, node_id_hex=ident1.node_id,
        signature=sig1, issued_at_iso=iso1,
    )
    assert b1 is not None

    ident2 = _make_identity()
    sig2, iso2 = _sign_binding(ident2, wallet, pk_hex)
    b2 = svc.bind(
        wallet_address=wallet, node_id_hex=ident2.node_id,
        signature=sig2, issued_at_iso=iso2,
    )
    assert b2 is not None
    assert b1.node_id_hex != b2.node_id_hex

    # Both retrievable
    bindings = store.get_all_by_wallet(wallet)
    assert len(bindings) == 2


# ---- Service: node-side uniqueness STILL enforced -------------


def test_service_bind_same_node_different_wallet_still_conflicts():
    """Sprint 786 lifts wallet-side 1:1 but NODE-side uniqueness
    remains: a node_id can't be bound to two different wallets.
    This is load-bearing — without it, operator A could "steal"
    operator B's node by binding it to A's wallet."""
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
        WalletBindingService,
        BindingConflictError,
    )
    from eth_account import Account

    acct_a = Account.create()
    acct_b = Account.create()

    store = InMemoryWalletBindingStore()
    svc = WalletBindingService(store=store)

    ident = _make_identity()

    # Bind to wallet A first
    sig_a, iso_a = _sign_binding(ident, acct_a.address, acct_a.key.to_0x_hex())
    svc.bind(
        wallet_address=acct_a.address, node_id_hex=ident.node_id,
        signature=sig_a, issued_at_iso=iso_a,
    )

    # Wallet B tries to claim the same node → MUST raise
    sig_b, iso_b = _sign_binding(ident, acct_b.address, acct_b.key.to_0x_hex())
    import pytest
    with pytest.raises(BindingConflictError):
        svc.bind(
            wallet_address=acct_b.address, node_id_hex=ident.node_id,
            signature=sig_b, issued_at_iso=iso_b,
        )


# ---- Back-compat: get_by_wallet ---------------------------------


def test_get_by_wallet_returns_first_binding_for_backcompat():
    """Callers that haven't migrated to get_all_by_wallet still
    work — get_by_wallet returns the first (oldest) binding."""
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
        IdentityBinding,
    )
    from eth_utils import to_checksum_address
    wallet = to_checksum_address("0x" + "d" * 40)
    s = InMemoryWalletBindingStore()
    s.insert(IdentityBinding(
        wallet_address=wallet, node_id_hex="2" * 32,
        bound_at_unix=2000, wallet_signature="0x", signing_message_hash="0x",
    ))
    s.insert(IdentityBinding(
        wallet_address=wallet, node_id_hex="1" * 32,
        bound_at_unix=1000, wallet_signature="0x", signing_message_hash="0x",
    ))
    # First-by-bound_at_unix = node 1
    b = s.get_by_wallet(wallet)
    assert b is not None
    assert b.node_id_hex == "1" * 32


# ---- Idempotency preserved ------------------------------------


def test_service_bind_idempotent_same_wallet_same_node():
    """Re-binding the SAME (wallet, node_id) returns a binding
    without raising (existing idempotency contract preserved)."""
    from prsm.interface.onboarding.wallet_binding import (
        InMemoryWalletBindingStore,
        WalletBindingService,
    )
    from eth_account import Account

    acct = Account.create()
    store = InMemoryWalletBindingStore()
    svc = WalletBindingService(store=store)

    ident = _make_identity()
    sig, iso = _sign_binding(ident, acct.address, acct.key.to_0x_hex())

    b1 = svc.bind(
        wallet_address=acct.address, node_id_hex=ident.node_id,
        signature=sig, issued_at_iso=iso,
    )
    b2 = svc.bind(
        wallet_address=acct.address, node_id_hex=ident.node_id,
        signature=sig, issued_at_iso=iso,
    )
    # Same binding returned (or at least same node_id+wallet)
    assert b1.node_id_hex == b2.node_id_hex
    assert b1.wallet_address == b2.wallet_address

    # Still only ONE binding stored (idempotent insert)
    bindings = store.get_all_by_wallet(acct.address)
    assert len(bindings) == 1
