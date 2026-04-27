"""
Unit tests — Phase 3.x.3 Task 3 — PublisherKeyAnchorClient.

Acceptance per design plan §4 Task 3: register-self happy path,
lookup hits cache, lookup falls through to chain on cache miss,
cache TTL expiry, register revert maps to PublisherAlreadyRegisteredError,
RPC failure surfaces as AnchorRPCError.

Uses unittest.mock to patch the two private contract-wrapper methods
(``_call_lookup``, ``_call_register``) — no live RPC, no chain. The
methods themselves are exercised by the E2E integration test (Task 7)
against a local Hardhat node.
"""

from __future__ import annotations

import base64
import hashlib
from unittest.mock import MagicMock

import pytest

from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.security.publisher_key_anchor import (
    AnchorRPCError,
    PublisherAlreadyRegisteredError,
    PublisherKeyAnchorClient,
    PublisherKeyAnchorError,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


# A fake-but-valid 0x-prefixed Ethereum address. The client uses
# Web3.to_checksum_address only when given an rpc_url path; with the
# explicit `contract=` injection path (used by every test below) the
# address is stored as-given.
ANCHOR_ADDR = "0x0000000000000000000000000000000000000001"


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.3-task3-publisher")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.3-task3-other")


@pytest.fixture
def fake_contract():
    """A bare MagicMock standing in for the Web3 contract object.

    Tests that need to control specific contract behavior patch the
    client's private wrapper methods directly (cleaner than mocking
    the full functions().call() chain).
    """
    return MagicMock(name="FakeContract")


@pytest.fixture
def read_only_client(fake_contract):
    """Read-only client (no signing wallet) suitable for verifier
    code paths."""
    return PublisherKeyAnchorClient(
        contract_address=ANCHOR_ADDR,
        contract=fake_contract,
        cache_ttl_seconds=3600,
    )


@pytest.fixture
def writable_client(fake_contract):
    """Client with a deterministic dummy private key. The actual
    transaction-build path is patched in tests; the key is here to
    satisfy the constructor's RW-mode check."""
    # Hardhat account 0 private key — deterministic, no real ETH ever
    # touches it; safe to pin in test fixtures.
    dummy_key = (
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    )
    return PublisherKeyAnchorClient(
        contract_address=ANCHOR_ADDR,
        contract=fake_contract,
        private_key=dummy_key,
        cache_ttl_seconds=3600,
    )


# ──────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_read_only_construction(self, fake_contract):
        c = PublisherKeyAnchorClient(
            contract_address=ANCHOR_ADDR, contract=fake_contract
        )
        assert c.address == ANCHOR_ADDR
        assert c.signer_address is None

    def test_writable_construction(self, writable_client):
        # Hardhat account 0 has a known address.
        assert writable_client.signer_address == (
            "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        )

    def test_missing_rpc_and_web3_raises(self, fake_contract):
        # If neither rpc_url nor web3 is provided AND we're going
        # through the "build my own contract" path (no contract=),
        # construction must fail. With contract= injection, rpc_url
        # is optional.
        with pytest.raises(ValueError, match="rpc_url or web3"):
            PublisherKeyAnchorClient(contract_address=ANCHOR_ADDR)


# ──────────────────────────────────────────────────────────────────────────
# lookup — happy path + cache
# ──────────────────────────────────────────────────────────────────────────


class TestLookup:
    def test_lookup_unregistered_returns_none(self, read_only_client):
        read_only_client._call_lookup = MagicMock(return_value=b"")
        assert read_only_client.lookup("ab" * 16) is None
        # Cached as negative — second call doesn't hit the chain.
        assert read_only_client.lookup("ab" * 16) is None
        assert read_only_client._call_lookup.call_count == 1

    def test_lookup_registered_returns_b64(
        self, read_only_client, identity
    ):
        # Build a real identity so the node_id and pubkey are coherent.
        pubkey_bytes = identity.public_key_bytes
        read_only_client._call_lookup = MagicMock(return_value=pubkey_bytes)

        result = read_only_client.lookup(identity.node_id)
        assert result == base64.b64encode(pubkey_bytes).decode("ascii")
        # Round-trips through identity.public_key_b64
        assert result == identity.public_key_b64

    def test_lookup_caches_positive_result(
        self, read_only_client, identity
    ):
        read_only_client._call_lookup = MagicMock(
            return_value=identity.public_key_bytes
        )
        # Two lookups; only one chain hit
        read_only_client.lookup(identity.node_id)
        read_only_client.lookup(identity.node_id)
        assert read_only_client._call_lookup.call_count == 1

    def test_lookup_node_id_normalized_lowercase(
        self, read_only_client, identity
    ):
        read_only_client._call_lookup = MagicMock(
            return_value=identity.public_key_bytes
        )
        # Mixed case in node_id should hit the same cache entry
        upper = identity.node_id.upper()
        read_only_client.lookup(identity.node_id)
        read_only_client.lookup(upper)
        assert read_only_client._call_lookup.call_count == 1

    def test_lookup_accepts_0x_prefix(self, read_only_client, identity):
        read_only_client._call_lookup = MagicMock(
            return_value=identity.public_key_bytes
        )
        prefixed = "0x" + identity.node_id
        result = read_only_client.lookup(prefixed)
        assert result == identity.public_key_b64

    def test_lookup_rejects_wrong_length_node_id(self, read_only_client):
        with pytest.raises(ValueError, match="32 hex chars"):
            read_only_client.lookup("abc")

    def test_lookup_rejects_non_hex_node_id(self, read_only_client):
        with pytest.raises(ValueError, match="not valid hex"):
            read_only_client.lookup("z" * 32)

    def test_lookup_rejects_non_string(self, read_only_client):
        with pytest.raises(ValueError, match="hex string"):
            read_only_client.lookup(b"not a string")  # type: ignore[arg-type]

    def test_lookup_rpc_failure_raises_anchor_rpc_error(
        self, read_only_client
    ):
        read_only_client._call_lookup = MagicMock(
            side_effect=ConnectionError("rpc unreachable")
        )
        with pytest.raises(AnchorRPCError, match="lookup.*RPC failed"):
            read_only_client.lookup("ab" * 16)

    def test_lookup_rejects_unexpected_pubkey_length(self, read_only_client):
        # Defensive: contract guarantees 32-byte pubkeys, but a
        # downgraded / forked contract could deviate. Don't trust it.
        read_only_client._call_lookup = MagicMock(return_value=b"\x00" * 64)
        with pytest.raises(AnchorRPCError, match="unexpected length"):
            read_only_client.lookup("ab" * 16)


# ──────────────────────────────────────────────────────────────────────────
# Cache TTL semantics
# ──────────────────────────────────────────────────────────────────────────


class TestCacheTTL:
    def test_cache_expiry_re_fetches(
        self, fake_contract, identity, monkeypatch
    ):
        client = PublisherKeyAnchorClient(
            contract_address=ANCHOR_ADDR,
            contract=fake_contract,
            cache_ttl_seconds=60,
        )
        client._call_lookup = MagicMock(return_value=identity.public_key_bytes)

        # First call at t=0
        fake_now = [1000.0]
        monkeypatch.setattr(
            "prsm.security.publisher_key_anchor.client.time.monotonic",
            lambda: fake_now[0],
        )
        client.lookup(identity.node_id)

        # 30 seconds later — still cached
        fake_now[0] = 1030.0
        client.lookup(identity.node_id)
        assert client._call_lookup.call_count == 1

        # 90 seconds later — TTL expired, re-fetch
        fake_now[0] = 1090.0
        client.lookup(identity.node_id)
        assert client._call_lookup.call_count == 2

    def test_invalidate_evicts_one_entry(
        self, read_only_client, identity, other_identity
    ):
        read_only_client._call_lookup = MagicMock(
            return_value=identity.public_key_bytes
        )
        read_only_client.lookup(identity.node_id)
        read_only_client.lookup(other_identity.node_id)
        assert read_only_client._call_lookup.call_count == 2

        read_only_client.invalidate(identity.node_id)
        # First publisher re-fetches; other is still cached
        read_only_client._call_lookup.return_value = identity.public_key_bytes
        read_only_client.lookup(identity.node_id)
        read_only_client.lookup(other_identity.node_id)
        assert read_only_client._call_lookup.call_count == 3

    def test_invalidate_all_clears_everything(
        self, read_only_client, identity, other_identity
    ):
        read_only_client._call_lookup = MagicMock(
            return_value=identity.public_key_bytes
        )
        read_only_client.lookup(identity.node_id)
        read_only_client.lookup(other_identity.node_id)
        read_only_client.invalidate_all()
        # Both re-fetch after invalidate_all
        read_only_client.lookup(identity.node_id)
        read_only_client.lookup(other_identity.node_id)
        assert read_only_client._call_lookup.call_count == 4

    def test_negative_cache_evicted_by_invalidate(
        self, read_only_client, identity
    ):
        read_only_client._call_lookup = MagicMock(return_value=b"")
        assert read_only_client.lookup(identity.node_id) is None
        read_only_client.invalidate(identity.node_id)
        # After invalidate, a registered lookup is now visible
        read_only_client._call_lookup.return_value = identity.public_key_bytes
        assert read_only_client.lookup(identity.node_id) == identity.public_key_b64


# ──────────────────────────────────────────────────────────────────────────
# register_self
# ──────────────────────────────────────────────────────────────────────────


class TestRegisterSelf:
    def test_register_returns_tx_hash(self, writable_client, identity):
        writable_client._call_register = MagicMock(return_value="0xabcd")
        tx_hash = writable_client.register_self(identity)
        assert tx_hash == "0xabcd"

    def test_register_populates_cache(self, writable_client, identity):
        # After a successful register, lookup of the same node_id must
        # return the registered pubkey from cache without hitting chain.
        writable_client._call_register = MagicMock(return_value="0xabcd")
        writable_client._call_lookup = MagicMock()
        writable_client.register_self(identity)
        result = writable_client.lookup(identity.node_id)
        assert result == identity.public_key_b64
        # Lookup never went to chain
        writable_client._call_lookup.assert_not_called()

    def test_read_only_register_raises(self, read_only_client, identity):
        with pytest.raises(RuntimeError, match="signing wallet"):
            read_only_client.register_self(identity)

    def test_register_already_registered_revert(
        self, writable_client, identity
    ):
        # A real Web3 revert exposes error_name on the exception.
        revert_exc = Exception("execution reverted: AlreadyRegistered")
        revert_exc.error_name = "AlreadyRegistered"  # type: ignore[attr-defined]
        writable_client._call_register = MagicMock(side_effect=revert_exc)

        with pytest.raises(
            PublisherAlreadyRegisteredError, match=identity.node_id
        ):
            writable_client.register_self(identity)

    def test_register_other_revert_surfaces_as_anchor_rpc_error(
        self, writable_client, identity
    ):
        revert_exc = Exception("execution reverted: SomeOtherError")
        writable_client._call_register = MagicMock(side_effect=revert_exc)
        with pytest.raises(AnchorRPCError, match="register_self"):
            writable_client.register_self(identity)

    def test_register_passes_pubkey_bytes_to_contract(
        self, writable_client, identity
    ):
        writable_client._call_register = MagicMock(return_value="0xdead")
        writable_client.register_self(identity)
        writable_client._call_register.assert_called_once_with(
            identity.public_key_bytes
        )


# ──────────────────────────────────────────────────────────────────────────
# node_id ↔ contract bytes16 conversion
# ──────────────────────────────────────────────────────────────────────────


class TestNodeIdConversion:
    def test_lookup_passes_correct_bytes16_to_contract(
        self, read_only_client, identity
    ):
        # The contract uses bytes16; the client must convert from PRSM's
        # 32-char hex node_id to the raw 16-byte form.
        read_only_client._call_lookup = MagicMock(
            return_value=identity.public_key_bytes
        )
        read_only_client.lookup(identity.node_id)
        passed = read_only_client._call_lookup.call_args.args[0]
        assert isinstance(passed, bytes)
        assert len(passed) == 16
        # Matches first 16 bytes of sha256(pubkey) — same derivation
        # rule as the Solidity contract enforces on-chain.
        expected = hashlib.sha256(identity.public_key_bytes).digest()[:16]
        assert passed == expected


# ──────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        from prsm.security.publisher_key_anchor import (
            AnchorRPCError,
            PublisherAlreadyRegisteredError,
            PublisherKeyAnchorError,
            PublisherNotRegisteredError,
        )

        assert issubclass(PublisherAlreadyRegisteredError, PublisherKeyAnchorError)
        assert issubclass(PublisherNotRegisteredError, PublisherKeyAnchorError)
        assert issubclass(AnchorRPCError, PublisherKeyAnchorError)
