"""Unit tests for prsm.interface.onboarding.siwe.

Covers the Task 1 test matrix from
docs/2026-04-22-phase4-wallet-sdk-design-plan.md §6:

  - valid sig
  - expired message
  - wrong chain id
  - replay (nonce reuse)
  - address / signature mismatch

Plus supporting tests for the in-memory nonce store and malformed messages.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from prsm.interface.onboarding.siwe import (
    InMemoryNonceStore,
    SiweChainIdError,
    SiweDomainError,
    SiweExpiredError,
    SiweMalformedError,
    SiweNonceError,
    SiweNotYetValidError,
    SiweSignatureError,
    VerifiedSiwe,
    verify,
)

DOMAIN = "app.prsm-network.com"
CHAIN_ID = 8453  # Base mainnet
URI = "https://app.prsm-network.com/login"
VERSION = "1"
STATEMENT = "Sign in to PRSM."


def _account():
    # Deterministic key for reproducibility.
    return Account.from_key(
        "0x" + "ab" * 31 + "cd"
    )


def _build_message(
    address: str,
    nonce: str,
    *,
    domain: str = DOMAIN,
    chain_id: int = CHAIN_ID,
    issued_at: str = "2026-04-22T12:00:00Z",
    expiration_time: str | None = None,
    not_before: str | None = None,
    uri: str = URI,
    version: str = VERSION,
    statement: str | None = STATEMENT,
) -> str:
    lines = [f"{domain} wants you to sign in with your Ethereum account:", address, ""]
    if statement:
        lines.append(statement)
        lines.append("")
    lines.append(f"URI: {uri}")
    lines.append(f"Version: {version}")
    lines.append(f"Chain ID: {chain_id}")
    lines.append(f"Nonce: {nonce}")
    lines.append(f"Issued At: {issued_at}")
    if expiration_time:
        lines.append(f"Expiration Time: {expiration_time}")
    if not_before:
        lines.append(f"Not Before: {not_before}")
    return "\n".join(lines)


def _sign(message: str, acct) -> str:
    encoded = encode_defunct(text=message)
    signed = acct.sign_message(encoded)
    return signed.signature.hex()


@pytest.fixture
def store():
    return InMemoryNonceStore()


@pytest.fixture
def acct():
    return _account()


@pytest.fixture
def now():
    return datetime(2026, 4, 22, 12, 5, tzinfo=timezone.utc)


# --- Valid path ---------------------------------------------------------------


def test_valid_signature_verifies_and_consumes_nonce(store, acct, now):
    nonce = store.issue()
    msg = _build_message(acct.address, nonce)
    sig = _sign(msg, acct)

    result = verify(
        msg,
        sig,
        expected_domain=DOMAIN,
        expected_chain_id=CHAIN_ID,
        nonce_store=store,
        now=now,
    )

    assert isinstance(result, VerifiedSiwe)
    assert result.address.lower() == acct.address.lower()
    assert result.chain_id == CHAIN_ID
    assert result.nonce == nonce

    # Nonce consumed — second call with same nonce must fail replay check.
    with pytest.raises(SiweNonceError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


# --- Failure paths ------------------------------------------------------------


def test_wrong_chain_id_raises(store, acct, now):
    nonce = store.issue()
    msg = _build_message(acct.address, nonce, chain_id=1)  # ethereum mainnet, not base
    sig = _sign(msg, acct)

    with pytest.raises(SiweChainIdError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_wrong_domain_raises(store, acct, now):
    nonce = store.issue()
    msg = _build_message(acct.address, nonce, domain="evil.com")
    sig = _sign(msg, acct)

    with pytest.raises(SiweDomainError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_expired_message_raises(store, acct, now):
    nonce = store.issue()
    msg = _build_message(
        acct.address,
        nonce,
        issued_at="2026-04-22T11:00:00Z",
        expiration_time="2026-04-22T12:00:00Z",  # before `now`
    )
    sig = _sign(msg, acct)

    with pytest.raises(SiweExpiredError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_not_yet_valid_message_raises(store, acct, now):
    nonce = store.issue()
    msg = _build_message(
        acct.address,
        nonce,
        issued_at="2026-04-22T12:00:00Z",
        not_before="2026-04-22T13:00:00Z",  # after `now`
    )
    sig = _sign(msg, acct)

    with pytest.raises(SiweNotYetValidError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_address_mismatch_raises(store, acct, now):
    """Message claims address X, but was signed by address Y."""
    other = Account.from_key("0x" + "de" * 31 + "ad")
    nonce = store.issue()
    msg = _build_message(other.address, nonce)  # claim other's address
    sig = _sign(msg, acct)  # sign with our key — recovery won't match

    with pytest.raises(SiweSignatureError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_tampered_message_signature_fails(store, acct, now):
    """Signature is valid over a different message than submitted."""
    nonce = store.issue()
    signed_msg = _build_message(acct.address, nonce, statement="Sign in to PRSM.")
    submitted_msg = _build_message(acct.address, nonce, statement="Transfer 1000 FTNS.")
    sig = _sign(signed_msg, acct)

    with pytest.raises(SiweSignatureError):
        verify(
            submitted_msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_malformed_message_raises(store, now):
    with pytest.raises(SiweMalformedError):
        verify(
            "this is not a SIWE message at all",
            "0x" + "00" * 65,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


def test_unregistered_nonce_raises(store, acct, now):
    """Nonce never issued by our store — must be rejected even with valid sig."""
    msg = _build_message(acct.address, "nonceNeverIssued1234")
    sig = _sign(msg, acct)

    with pytest.raises(SiweNonceError):
        verify(
            msg,
            sig,
            expected_domain=DOMAIN,
            expected_chain_id=CHAIN_ID,
            nonce_store=store,
            now=now,
        )


# --- InMemoryNonceStore behaviour --------------------------------------------


def test_nonce_store_issue_is_single_use():
    store = InMemoryNonceStore()
    nonce = store.issue()
    assert store.consume(nonce) is True
    assert store.consume(nonce) is False  # replay


def test_nonce_store_unknown_nonce():
    store = InMemoryNonceStore()
    assert store.consume("never-issued") is False


def test_nonce_store_ttl_expiry():
    t = [1000.0]
    store = InMemoryNonceStore(_now=lambda: t[0])
    nonce = store.issue(ttl_seconds=60)

    t[0] = 1050.0  # still within TTL
    # peek without consuming — we use a separate nonce to avoid consuming the target
    other = store.issue(ttl_seconds=60)
    assert store.consume(other) is True

    t[0] = 1100.0  # past TTL of original
    assert store.consume(nonce) is False
