"""
Unit tests — Phase 3.x.4 Task 3 — PrivacyBudgetStore ABC + InMemoryPrivacyBudgetStore.

Acceptance per design plan §4 Task 3: append-only invariant + chain
invariant enforced by both append and verify; test coverage for both
directions (write-time rejection AND read-time tamper detection).

Real Ed25519 signatures and real chain construction — no mocks.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from prsm.security.privacy_budget_persistence import (
    GENESIS_PREV_HASH,
    InMemoryPrivacyBudgetStore,
    JournalCorruptionError,
    OutOfOrderAppendError,
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
    PrivacyBudgetStoreError,
    hash_entry_payload,
    sign_entry,
)
from prsm.node.identity import NodeIdentity, generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Helpers — build a real signed entry chain
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.4-task3-signer")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.4-task3-impostor")


def _build_entry(
    sequence_number: int,
    prev_entry_hash: bytes,
    *,
    epsilon: float = 8.0,
    operation: str = "inference",
    model_id: str = "llama-3-8b",
    timestamp: float = 1714000000.0,
    entry_type: PrivacyBudgetEntryType = PrivacyBudgetEntryType.SPEND,
    node_id: str = "placeholder",
) -> PrivacyBudgetEntry:
    """Build an unsigned entry with the given sequence + prev_hash."""
    if entry_type == PrivacyBudgetEntryType.RESET:
        epsilon = 0.0
        operation = ""
        model_id = ""
    return PrivacyBudgetEntry(
        sequence_number=sequence_number,
        entry_type=entry_type,
        node_id=node_id,
        epsilon=epsilon,
        operation=operation,
        model_id=model_id,
        timestamp=timestamp + sequence_number,  # distinct per entry
        prev_entry_hash=prev_entry_hash,
    )


def _build_signed_chain(
    identity: NodeIdentity, length: int
) -> list[PrivacyBudgetEntry]:
    """Build a signed chain of ``length`` entries, each chained to its predecessor."""
    chain = []
    prev_hash = GENESIS_PREV_HASH
    for i in range(length):
        entry = _build_entry(i, prev_hash)
        signed = sign_entry(entry, identity)
        chain.append(signed)
        prev_hash = hash_entry_payload(signed)
    return chain


@pytest.fixture
def store() -> InMemoryPrivacyBudgetStore:
    return InMemoryPrivacyBudgetStore()


# ──────────────────────────────────────────────────────────────────────────
# hash_entry_payload — public helper, contract test
# ──────────────────────────────────────────────────────────────────────────


class TestHashEntryPayload:
    def test_matches_sha256_of_signing_payload(self, identity):
        e = sign_entry(_build_entry(0, GENESIS_PREV_HASH), identity)
        assert hash_entry_payload(e) == hashlib.sha256(e.signing_payload()).digest()

    def test_returns_32_bytes(self, identity):
        e = sign_entry(_build_entry(0, GENESIS_PREV_HASH), identity)
        # Same width as a sha256 digest — chain logic depends on this.
        assert len(hash_entry_payload(e)) == 32


# ──────────────────────────────────────────────────────────────────────────
# Empty store
# ──────────────────────────────────────────────────────────────────────────


class TestEmptyStore:
    def test_len_zero(self, store):
        assert len(store) == 0

    def test_replay_yields_nothing(self, store):
        assert list(store.replay()) == []

    def test_latest_hash_is_genesis(self, store):
        assert store.latest_hash() == GENESIS_PREV_HASH

    def test_verify_chain_true_for_empty(self, store, identity):
        # Empty journal trivially verifies.
        assert store.verify_chain(identity.public_key_b64) is True


# ──────────────────────────────────────────────────────────────────────────
# Append — happy path + invariants
# ──────────────────────────────────────────────────────────────────────────


class TestAppend:
    def test_basic_append(self, store, identity):
        chain = _build_signed_chain(identity, 1)
        store.append(chain[0])
        assert len(store) == 1
        assert store.latest_hash() == hash_entry_payload(chain[0])

    def test_chain_of_three(self, store, identity):
        chain = _build_signed_chain(identity, 3)
        for entry in chain:
            store.append(entry)
        assert len(store) == 3
        assert store.latest_hash() == hash_entry_payload(chain[-1])

    def test_genesis_entry_must_use_genesis_prev_hash(self, store, identity):
        # Sequence 0 with a non-genesis prev_hash is a chain violation.
        bad_genesis = sign_entry(
            _build_entry(0, prev_entry_hash=hashlib.sha256(b"x").digest()),
            identity,
        )
        with pytest.raises(OutOfOrderAppendError, match="prev_entry_hash"):
            store.append(bad_genesis)

    def test_sequence_gap_rejected(self, store, identity):
        # Append entry 0, then try to append entry 2 (skipping 1).
        chain = _build_signed_chain(identity, 3)
        store.append(chain[0])
        with pytest.raises(OutOfOrderAppendError, match="sequence number gap"):
            store.append(chain[2])

    def test_duplicate_sequence_rejected(self, store, identity):
        # Append entry 0, then try to append another sequence-0 entry.
        chain = _build_signed_chain(identity, 1)
        store.append(chain[0])
        # A second, distinct entry with sequence_number=0
        another_zero = sign_entry(
            _build_entry(0, GENESIS_PREV_HASH, operation="other"),
            identity,
        )
        with pytest.raises(OutOfOrderAppendError, match="sequence number gap"):
            store.append(another_zero)

    def test_wrong_prev_hash_rejected(self, store, identity):
        # Build a normal genesis, then build entry 1 with a wrong
        # prev_entry_hash (not the actual hash of entry 0).
        chain = _build_signed_chain(identity, 1)
        store.append(chain[0])
        bad_next = sign_entry(
            _build_entry(1, prev_entry_hash=hashlib.sha256(b"wrong").digest()),
            identity,
        )
        with pytest.raises(OutOfOrderAppendError, match="prev_entry_hash mismatch"):
            store.append(bad_next)

    def test_failed_append_does_not_change_state(self, store, identity):
        # If append raises, the store must not be modified.
        chain = _build_signed_chain(identity, 1)
        store.append(chain[0])
        bad_next = sign_entry(
            _build_entry(2, prev_entry_hash=hash_entry_payload(chain[0])),
            identity,
        )
        with pytest.raises(OutOfOrderAppendError):
            store.append(bad_next)
        # State unchanged
        assert len(store) == 1
        assert store.latest_hash() == hash_entry_payload(chain[0])


# ──────────────────────────────────────────────────────────────────────────
# Replay — order + completeness
# ──────────────────────────────────────────────────────────────────────────


class TestReplay:
    def test_replay_yields_in_sequence_order(self, store, identity):
        chain = _build_signed_chain(identity, 5)
        for entry in chain:
            store.append(entry)
        replayed = list(store.replay())
        assert [e.sequence_number for e in replayed] == [0, 1, 2, 3, 4]

    def test_replay_returns_iterator(self, store, identity):
        # Iterator protocol is part of the ABC contract — callers can
        # stream from disk-backed stores without loading everything.
        chain = _build_signed_chain(identity, 2)
        for e in chain:
            store.append(e)
        result = store.replay()
        assert hasattr(result, "__next__")  # iterator protocol

    def test_replay_full_byte_equality(self, store, identity):
        # Replayed entries must be byte-identical to what was appended.
        chain = _build_signed_chain(identity, 3)
        for e in chain:
            store.append(e)
        replayed = list(store.replay())
        for original, returned in zip(chain, replayed):
            assert original == returned


# ──────────────────────────────────────────────────────────────────────────
# verify_chain — read-time integrity check
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyChain:
    def test_verifies_clean_chain(self, store, identity):
        for e in _build_signed_chain(identity, 5):
            store.append(e)
        assert store.verify_chain(identity.public_key_b64) is True

    def test_verifies_single_entry(self, store, identity):
        store.append(_build_signed_chain(identity, 1)[0])
        assert store.verify_chain(identity.public_key_b64) is True

    def test_rejects_wrong_pubkey(self, store, identity, other_identity):
        for e in _build_signed_chain(identity, 3):
            store.append(e)
        # Same chain, different pubkey → every signature fails.
        assert store.verify_chain(other_identity.public_key_b64) is False

    def test_rejects_post_write_signature_tamper(self, store, identity):
        # Append a clean chain, then poke a flipped signature byte into
        # the in-memory list (simulating disk corruption or attacker
        # write).
        chain = _build_signed_chain(identity, 3)
        for e in chain:
            store.append(e)
        # Reach into private state to corrupt entry 1's signature
        bad_sig = bytes([store._entries[1].signature[0] ^ 0xFF]) + store._entries[1].signature[1:]
        store._entries[1] = dataclasses.replace(store._entries[1], signature=bad_sig)
        assert store.verify_chain(identity.public_key_b64) is False

    def test_rejects_post_write_field_tamper(self, store, identity):
        # Tamper with epsilon on a stored entry. Signature now invalid.
        chain = _build_signed_chain(identity, 3)
        for e in chain:
            store.append(e)
        store._entries[1] = dataclasses.replace(
            store._entries[1], epsilon=store._entries[1].epsilon / 2
        )
        assert store.verify_chain(identity.public_key_b64) is False

    def test_rejects_chain_link_break(self, store, identity):
        # Mutate prev_entry_hash on entry 1 — chain hash check will fail
        # before we even hit the signature check.
        chain = _build_signed_chain(identity, 3)
        for e in chain:
            store.append(e)
        store._entries[1] = dataclasses.replace(
            store._entries[1],
            prev_entry_hash=hashlib.sha256(b"forged").digest(),
        )
        assert store.verify_chain(identity.public_key_b64) is False

    def test_rejects_swapped_entries(self, store, identity):
        # Swap entries 1 and 2 in the internal list — sequence numbers
        # are now out of order, verify must fail.
        chain = _build_signed_chain(identity, 3)
        for e in chain:
            store.append(e)
        store._entries[1], store._entries[2] = store._entries[2], store._entries[1]
        assert store.verify_chain(identity.public_key_b64) is False

    def test_genesis_with_wrong_prev_hash_rejected(self, store, identity):
        # Bypass append() to insert a genesis-position entry with a
        # non-genesis prev_hash. verify_chain must catch it on read.
        bad_genesis = sign_entry(
            _build_entry(0, prev_entry_hash=hashlib.sha256(b"x").digest()),
            identity,
        )
        store._entries.append(bad_genesis)
        store._latest_hash = hash_entry_payload(bad_genesis)
        assert store.verify_chain(identity.public_key_b64) is False


# ──────────────────────────────────────────────────────────────────────────
# Exception hierarchy
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(OutOfOrderAppendError, PrivacyBudgetStoreError)
        assert issubclass(JournalCorruptionError, PrivacyBudgetStoreError)

    def test_distinct_types(self):
        # Callers want to catch chain-corruption distinct from append-time
        # ordering bugs; same base, different concrete types.
        assert OutOfOrderAppendError is not JournalCorruptionError


# ──────────────────────────────────────────────────────────────────────────
# Restart-semantics property: append-then-replay round-trip
# ──────────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    """Even for the in-memory store, the pattern that
    PersistentPrivacyBudgetTracker (Task 5) will use must be byte-stable:
    append a chain, replay it, every entry comes back unchanged."""

    def test_round_trip_preserves_entries(self, store, identity):
        original = _build_signed_chain(identity, 4)
        for e in original:
            store.append(e)
        replayed = list(store.replay())
        for o, r in zip(original, replayed):
            assert o == r
            assert o.signature == r.signature
            assert o.prev_entry_hash == r.prev_entry_hash

    def test_round_trip_chain_remains_verifiable(self, store, identity):
        for e in _build_signed_chain(identity, 4):
            store.append(e)
        # The exact contract Task 5 needs: append → replay → verify_chain
        # all the way through.
        assert store.verify_chain(identity.public_key_b64) is True
