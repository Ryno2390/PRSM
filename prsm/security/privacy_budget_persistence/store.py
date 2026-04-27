"""
Privacy budget journal store — abstract interface + in-memory implementation.

Phase 3.x.4 Task 3.

The ABC pins the contract every backend (in-memory, filesystem,
future DHT, future on-chain anchor) must satisfy:

- ``append(entry)`` enforces structural invariants: gap-free monotonic
  sequence numbers AND the chain link
  (``entry.prev_entry_hash == sha256(prev_entry.signing_payload())``).
  Raises ``OutOfOrderAppendError`` on either violation.
- ``replay()`` yields entries in sequence order so callers can
  reconstitute cumulative state.
- ``latest_hash()`` returns the sha256 of the latest entry's
  signing_payload — the value the next ``prev_entry_hash`` must match.
- ``verify_chain(public_key_b64)`` walks the whole chain checking
  signatures + chain hashes; returns True iff the journal is intact.

Cryptographic verification on read (not write): ``append`` accepts
any entry shape (signed or not — the tracker layer signs before
appending). ``verify_chain`` is the integrity check; production
callers MUST run it after replay to confirm the journal hasn't been
tampered with at rest.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Iterator, List

from prsm.security.privacy_budget_persistence.models import (
    GENESIS_PREV_HASH,
    PrivacyBudgetEntry,
)
from prsm.security.privacy_budget_persistence.signing import verify_entry


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class PrivacyBudgetStoreError(Exception):
    """Base error for any store-layer failure."""


class OutOfOrderAppendError(PrivacyBudgetStoreError):
    """Append rejected: sequence number gap or prev_entry_hash mismatch.

    Raised at write time. Catches the most common journal-integrity
    bugs (caller passing a stale prev_entry_hash, accidental
    out-of-order append from a misordered queue) before any state
    change.
    """


class JournalCorruptionError(PrivacyBudgetStoreError):
    """Journal's on-disk or in-memory state has lost integrity.

    Raised at read time when ``verify_chain`` finds a broken signature
    or chain link. Distinct from ``OutOfOrderAppendError`` because the
    failure is post-write — something tampered with the journal after
    it was correctly appended.
    """


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def hash_entry_payload(entry: PrivacyBudgetEntry) -> bytes:
    """sha256 of entry.signing_payload — the chain link primitive.

    Public helper so callers building entries (the tracker layer in
    Task 5) can compute the next ``prev_entry_hash`` without
    reaching into implementation details.
    """
    return hashlib.sha256(entry.signing_payload()).digest()


# --------------------------------------------------------------------------
# ABC
# --------------------------------------------------------------------------


class PrivacyBudgetStore(abc.ABC):
    """Abstract append-only journal of signed privacy-budget events.

    All implementations MUST honor:

    1. ``append(entry)`` rejects if sequence gap or chain mismatch.
       Sequence 0 must have ``prev_entry_hash == GENESIS_PREV_HASH``;
       sequence N>0 must have ``prev_entry_hash == hash_entry_payload(entry[N-1])``.
       Anything else raises ``OutOfOrderAppendError``.
    2. ``replay()`` yields entries in monotonic sequence order
       (0, 1, 2, …). Implementations may stream from disk or yield
       from memory; the order contract is the same.
    3. ``latest_hash()`` returns sha256 of the highest-sequence entry's
       signing_payload, or ``GENESIS_PREV_HASH`` when empty. Equal to
       what the next ``append`` will require as ``prev_entry_hash``.
    4. ``verify_chain(public_key_b64)`` walks every entry, returning
       False iff any signature fails or any chain link is broken.
       Default impl provided — subclasses MAY override for a
       cheaper verification path (e.g., a content-addressed store
       that can skip already-verified prefixes).
    """

    @abc.abstractmethod
    def append(self, entry: PrivacyBudgetEntry) -> None:
        """Append; raises OutOfOrderAppendError on sequence/chain mismatch."""

    @abc.abstractmethod
    def replay(self) -> Iterator[PrivacyBudgetEntry]:
        """Yield entries in monotonic sequence order."""

    @abc.abstractmethod
    def latest_hash(self) -> bytes:
        """sha256 of the latest entry's payload, or GENESIS_PREV_HASH if empty."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of entries in the journal."""

    def verify_chain(self, public_key_b64: str) -> bool:
        """Walk the chain end-to-end. True iff signatures + chain links hold.

        Catches:
        - Any entry whose signature fails to verify under ``public_key_b64``
        - Any chain link mismatch (entry[N].prev_entry_hash !=
          hash_entry_payload(entry[N-1]))
        - Genesis entry whose prev_entry_hash != GENESIS_PREV_HASH
        - Sequence-number gap or duplicate

        Returns False rather than raising — fail closed, mirrors the
        verify_entry / verify_manifest contract.
        """
        prev_hash = GENESIS_PREV_HASH
        expected_seq = 0
        for entry in self.replay():
            # Sequence must be gap-free and monotonic
            if entry.sequence_number != expected_seq:
                return False
            # Chain link must match
            if entry.prev_entry_hash != prev_hash:
                return False
            # Signature must verify under the supplied pubkey
            if not verify_entry(entry, public_key_b64=public_key_b64):
                return False
            prev_hash = hash_entry_payload(entry)
            expected_seq += 1
        return True


# --------------------------------------------------------------------------
# In-memory implementation
# --------------------------------------------------------------------------


class InMemoryPrivacyBudgetStore(PrivacyBudgetStore):
    """Process-local journal — for tests and the ``store=None``
    fallback path on ``PersistentPrivacyBudgetTracker``.

    NOT suitable for production deployment alone — restart drops the
    journal. Use ``FilesystemPrivacyBudgetStore`` (Task 4) for
    restart-survival.

    Single-writer assumption per node; no internal locking.
    """

    def __init__(self) -> None:
        self._entries: List[PrivacyBudgetEntry] = []
        # Cached latest hash so latest_hash() is O(1) and append's
        # chain check doesn't need to rebuild the predecessor's payload.
        self._latest_hash: bytes = GENESIS_PREV_HASH

    def append(self, entry: PrivacyBudgetEntry) -> None:
        expected_seq = len(self._entries)
        if entry.sequence_number != expected_seq:
            raise OutOfOrderAppendError(
                f"sequence number gap: expected {expected_seq}, "
                f"got {entry.sequence_number}"
            )
        if entry.prev_entry_hash != self._latest_hash:
            raise OutOfOrderAppendError(
                f"prev_entry_hash mismatch at sequence {entry.sequence_number}: "
                f"expected {self._latest_hash.hex()[:16]}…, "
                f"got {entry.prev_entry_hash.hex()[:16]}…"
            )
        self._entries.append(entry)
        self._latest_hash = hash_entry_payload(entry)

    def replay(self) -> Iterator[PrivacyBudgetEntry]:
        # Internal list is already in sequence order (append enforces
        # monotonicity); no sort needed.
        return iter(self._entries)

    def latest_hash(self) -> bytes:
        return self._latest_hash

    def __len__(self) -> int:
        return len(self._entries)
