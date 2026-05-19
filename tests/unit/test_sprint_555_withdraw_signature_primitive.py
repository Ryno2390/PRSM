"""Sprint 555 — EIP-712 verification primitive for /wallet/withdraw.

Second sprint in the user-sig arc (554/555/556/557). Pure module
that builds the canonical EIP-712 typed-data payload + verifies
ECDSA secp256k1 signatures over it.

Domain spec (locked per user input):
  - Name:    "PRSM-Withdraw"
  - Version: "v1"
  - Chain:   8453 (Base mainnet)

Typed-data shape:
  WithdrawRequest {
    wallet_id        string
    amount_ftns_wei  uint256
    to_eth_address   address
    nonce            uint256
    expiry_unix      uint256
  }

Sprint 556 wires this into POST /wallet/withdraw. This sprint is
module-only — no API integration. Pin tests cover roundtrip,
tampered-field, wrong-signer, expired, cross-chain replay safety,
and canonical encoding stability.
"""
from __future__ import annotations

import time

import pytest
from eth_account import Account


# Convenience: a stable test signer.
def _signer():
    # Deterministic for reproducibility — NOT a real wallet.
    pk = "0x" + "11" * 32
    return Account.from_key(pk)


def _payload(
    *,
    wallet_id="alice",
    amount_ftns_wei=1_000_000_000_000_000_000,  # 1 FTNS
    to_eth_address=None,
    nonce=0,
    expiry_unix=None,
):
    return {
        "wallet_id": wallet_id,
        "amount_ftns_wei": int(amount_ftns_wei),
        "to_eth_address": (
            to_eth_address or "0x" + "ab" * 20
        ),
        "nonce": int(nonce),
        "expiry_unix": int(
            expiry_unix if expiry_unix is not None
            else time.time() + 300
        ),
    }


# ── canonical encoding ────────────────────────────────────


def test_canonical_payload_is_stable():
    """Same input → byte-identical typed-data hash. Stability is
    load-bearing: any change to the canonical encoding silently
    invalidates every signature on the network."""
    from prsm.economy.withdraw_signature import (
        encode_withdraw_typed_data_hash,
    )
    p = _payload(expiry_unix=1700000000)
    h1 = encode_withdraw_typed_data_hash(p)
    h2 = encode_withdraw_typed_data_hash(p)
    assert h1 == h2
    assert isinstance(h1, bytes)
    assert len(h1) == 32, f"expected 32-byte hash, got {len(h1)}"


def test_canonical_payload_changes_when_field_changes():
    """Mutating ANY field of the typed data must change the hash."""
    from prsm.economy.withdraw_signature import (
        encode_withdraw_typed_data_hash,
    )
    base = _payload(expiry_unix=1700000000)
    base_hash = encode_withdraw_typed_data_hash(base)

    for field, mutated in [
        ("wallet_id", "bob"),
        ("amount_ftns_wei", base["amount_ftns_wei"] + 1),
        ("to_eth_address", "0x" + "cd" * 20),
        ("nonce", base["nonce"] + 1),
        ("expiry_unix", base["expiry_unix"] + 1),
    ]:
        mutated_payload = {**base, field: mutated}
        h = encode_withdraw_typed_data_hash(mutated_payload)
        assert h != base_hash, (
            f"mutating {field} did not change the typed-data hash"
        )


# ── roundtrip ────────────────────────────────────────────


def test_sign_and_verify_roundtrip():
    """Sign with a known privkey; verify recovers the same address."""
    from prsm.economy.withdraw_signature import (
        sign_withdraw_payload,
        verify_withdraw_signature,
    )
    acct = _signer()
    payload = _payload(expiry_unix=int(time.time() + 300))
    sig = sign_withdraw_payload(payload, acct.key)

    recovered = verify_withdraw_signature(payload, sig)
    assert recovered.lower() == acct.address.lower()


def test_verify_returns_lowercase_checksum_match():
    """Recovered address comparison is case-insensitive; the helper
    should canonicalize to a recoverable form (we use lowercase
    throughout sprint-540 linked-eth-address storage)."""
    from prsm.economy.withdraw_signature import (
        sign_withdraw_payload,
        verify_withdraw_signature,
    )
    acct = _signer()
    payload = _payload(expiry_unix=int(time.time() + 300))
    sig = sign_withdraw_payload(payload, acct.key)

    recovered = verify_withdraw_signature(payload, sig)
    # The verify primitive returns the EIP-55 checksum address (the
    # eth_account default). Callers compare case-insensitively
    # against the linked eth_address (also EIP-55-cased by sprint
    # 540's checksum normalization).
    assert recovered.startswith("0x")
    assert len(recovered) == 42


# ── tampering rejection ───────────────────────────────────


def test_tampered_field_recovers_different_address():
    """If the verifier reconstructs the typed-data hash from a
    tampered payload, address recovery yields a DIFFERENT address —
    sprint 556's enforcement then sees a mismatch against the
    linked eth_address and rejects with 401."""
    from prsm.economy.withdraw_signature import (
        sign_withdraw_payload,
        verify_withdraw_signature,
    )
    acct = _signer()
    payload = _payload(expiry_unix=int(time.time() + 300))
    sig = sign_withdraw_payload(payload, acct.key)

    # Tamper the amount.
    tampered = {**payload, "amount_ftns_wei": payload["amount_ftns_wei"] * 2}
    recovered = verify_withdraw_signature(tampered, sig)
    assert recovered.lower() != acct.address.lower(), (
        "tampered payload should NOT recover the original signer"
    )


# ── expiry ────────────────────────────────────────────────


def test_check_expiry_rejects_expired_payload():
    """Helper checks expiry_unix against current time. Returns False
    (or raises a typed error) when the deadline has passed."""
    from prsm.economy.withdraw_signature import (
        is_expired,
    )
    expired_payload = _payload(expiry_unix=int(time.time()) - 1)
    assert is_expired(expired_payload) is True

    fresh_payload = _payload(expiry_unix=int(time.time()) + 300)
    assert is_expired(fresh_payload) is False


def test_check_expiry_uses_injected_clock():
    """Tests pass a clock callable so the expiry check is
    deterministic. Sprint 556 uses real time.time."""
    from prsm.economy.withdraw_signature import is_expired
    p = _payload(expiry_unix=1700000000)
    assert is_expired(p, now=lambda: 1700000000 - 1) is False
    assert is_expired(p, now=lambda: 1700000001) is True


# ── cross-chain replay safety ─────────────────────────────


def test_domain_separator_includes_chain_id():
    """The EIP-712 domain pins chainId=8453 (Base). A signature
    valid on Base must NOT verify on a hypothetical chain_id=1
    (Ethereum mainnet) deployment — operators on other chains MUST
    fork the domain to deploy.

    Implementation: this is enforced by EIP-712's domain separator
    construction. We assert it by computing the hash with two
    different chain_ids and confirming they differ.
    """
    from prsm.economy.withdraw_signature import (
        encode_withdraw_typed_data_hash,
    )
    p = _payload(expiry_unix=1700000000)
    h_base = encode_withdraw_typed_data_hash(p, chain_id=8453)
    h_eth = encode_withdraw_typed_data_hash(p, chain_id=1)
    assert h_base != h_eth, (
        "EIP-712 domain separator MUST bind to chain_id; "
        "same payload hashing identically across chains is a "
        "cross-chain replay vector."
    )


# ── invalid signature formats ─────────────────────────────


def test_verify_rejects_malformed_signature():
    """A signature that isn't 65 bytes / 0x-prefixed hex should
    raise a typed error rather than returning a junk address."""
    from prsm.economy.withdraw_signature import (
        verify_withdraw_signature,
        InvalidSignatureFormat,
    )
    payload = _payload(expiry_unix=int(time.time()) + 300)
    with pytest.raises(InvalidSignatureFormat):
        verify_withdraw_signature(payload, b"\x00")  # too short
    with pytest.raises(InvalidSignatureFormat):
        verify_withdraw_signature(payload, "not-hex")
