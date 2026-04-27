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
import json
import os
from pathlib import Path
from typing import Iterator, List, Union

from prsm.security.privacy_budget_persistence.models import (
    GENESIS_PREV_HASH,
    PrivacyBudgetEntry,
)
from prsm.security.privacy_budget_persistence.signing import verify_entry


# Filenames in the filesystem journal layout. Fixed strings — no user
# input maps to a filename — so path-traversal protection here reduces
# to "validate root exists" + "use only sequence-derived filenames".
_LATEST_FILENAME = "latest.json"
_PUBKEY_FILENAME = "node.pubkey"
_ENTRIES_DIRNAME = "entries"
_ENTRY_DIGITS = 6  # zero-padded entry filenames: 000000.json … 999999.json


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


# --------------------------------------------------------------------------
# Filesystem implementation
# --------------------------------------------------------------------------


class FilesystemPrivacyBudgetStore(PrivacyBudgetStore):
    """Persistent journal — entries/NNNNNN.json + latest.json + node.pubkey on disk.

    Layout per design plan §3.2::

        <root>/
        ├── entries/
        │   ├── 000000.json   — genesis (prev_entry_hash = b"\\x00" * 32)
        │   ├── 000001.json
        │   └── ...
        ├── latest.json       — {"sequence_number": N, "entry_hash": "<hex>"}
        └── node.pubkey       — b64 of the signing identity's public key

    Survives node restarts. Two store instances pointing at the same
    ``root`` see each other's writes once they hit the filesystem.

    LATEST-WRITTEN-LAST INVARIANT (do not change without updating tests):
        ``append()`` writes ``entries/NNNNNN.json`` first → ``latest.json``
        second. ``latest.json``'s presence with sequence N means
        "everything up through N is on disk and intact." A crashed
        ``append()`` leaves either:
        - ``latest.json`` still pointing at N-1 (orphaned ``entries/N.json``
          on disk, ignored by readers — wasted space, but reads stay
          consistent), OR
        - both written successfully → next reader picks up at N+1.
        The orphan-on-crash case is handled at __init__ by trusting
        ``latest.json`` over the entries/ directory contents.

    SECURITY — TRUST BOUNDARY:
        Same as Phase 3.x.2 model registry: the journal root is a local
        trust boundary. An attacker with write access to ``<root>/`` can
        replace ``node.pubkey`` AND re-sign every entry under their own
        key — the store will happily verify the substitute. This is
        acceptable for a node-local journal protected by filesystem
        permissions; cross-node verifiability requires the on-chain
        anchor planned for Phase 3.x.3.

        An attacker who deletes the journal entirely also wins — the
        next ``__init__`` will see an empty store and accept fresh
        entries from sequence 0. Documented in production-deployment
        notes: regulators auditing a node should pin the root to
        backed-up storage and check ``latest.json.sequence_number``
        monotonically increases across snapshots.

    Single-writer assumption per node; no cross-process locking.
    Concurrent writers may collide on ``entries/NNNNNN.json`` — the
    second writer's ``append()`` will see its own ``sequence_number``
    already taken (file exists or ``latest.json`` ahead of expected)
    and raise ``OutOfOrderAppendError``.
    """

    def __init__(
        self,
        root: Union[str, Path],
        public_key_b64: str,
    ) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(
                f"FilesystemPrivacyBudgetStore root {self._root} does not exist; "
                f"create the directory before constructing the store"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"FilesystemPrivacyBudgetStore root {self._root} is not a directory"
            )

        # Pubkey sidecar handling: write-on-first-construction;
        # validate-on-subsequent. This pins one journal to one signer
        # for its lifetime — caller passing a different pubkey on a
        # second open is treated as a configuration error.
        sidecar_path = self._root / _PUBKEY_FILENAME
        if sidecar_path.exists():
            existing = sidecar_path.read_text().strip()
            if existing != public_key_b64:
                raise JournalCorruptionError(
                    f"node.pubkey sidecar at {sidecar_path} doesn't match "
                    f"the supplied public key; this journal is bound to a "
                    f"different signer. Pass the matching public key, or "
                    f"start a fresh journal in a different root."
                )
        else:
            # First-time construction. If entries/ already has files,
            # something's wrong: a journal with entries but no sidecar
            # means the sidecar was deleted and we cannot verify the
            # chain. Refuse rather than silently write a (possibly wrong)
            # pubkey.
            entries_dir = self._root / _ENTRIES_DIRNAME
            if entries_dir.exists() and any(entries_dir.iterdir()):
                raise JournalCorruptionError(
                    f"entries directory at {entries_dir} is non-empty but "
                    f"node.pubkey is missing — cannot establish signer "
                    f"binding for the existing journal"
                )
            self._atomic_write_text(sidecar_path, public_key_b64)
            (self._root / _ENTRIES_DIRNAME).mkdir(exist_ok=True)

        self._public_key_b64 = public_key_b64

        # Read latest.json to know where the journal ends. If absent,
        # we're at genesis — an empty journal.
        latest_path = self._root / _LATEST_FILENAME
        if latest_path.exists():
            try:
                latest_data = json.loads(latest_path.read_text())
                self._count = int(latest_data["sequence_number"]) + 1
                self._latest_hash = bytes.fromhex(latest_data["entry_hash"])
            except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
                raise JournalCorruptionError(
                    f"latest.json at {latest_path} unreadable or malformed: {exc}"
                ) from exc
            if len(self._latest_hash) != 32:
                raise JournalCorruptionError(
                    f"latest.json entry_hash must be 32 bytes (sha256 width), "
                    f"got {len(self._latest_hash)}"
                )
        else:
            self._count = 0
            self._latest_hash = GENESIS_PREV_HASH

    # -- write path --

    def append(self, entry: PrivacyBudgetEntry) -> None:
        # Same structural checks as InMemoryPrivacyBudgetStore.append:
        # validate BEFORE writing so a rejected append leaves nothing
        # on disk.
        if entry.sequence_number != self._count:
            raise OutOfOrderAppendError(
                f"sequence number gap: expected {self._count}, "
                f"got {entry.sequence_number}"
            )
        if entry.prev_entry_hash != self._latest_hash:
            raise OutOfOrderAppendError(
                f"prev_entry_hash mismatch at sequence {entry.sequence_number}: "
                f"expected {self._latest_hash.hex()[:16]}…, "
                f"got {entry.prev_entry_hash.hex()[:16]}…"
            )

        entry_path = self._entry_path(entry.sequence_number)
        # Defense in depth: a future bug in _entry_path could theoretically
        # produce a path outside root. is_relative_to catches it.
        candidate = entry_path.resolve()
        root_resolved = self._root.resolve()
        if not candidate.is_relative_to(root_resolved):
            raise ValueError(
                f"entry path {candidate} would escape journal root {root_resolved}"
            )

        # Step 1: write the entry file atomically. If we crash here,
        # latest.json is unchanged and the orphan entry file is invisible
        # to readers.
        entry_json = json.dumps(entry.to_dict(), sort_keys=True, indent=2)
        self._atomic_write_text(entry_path, entry_json)

        # Step 2: update latest.json atomically. Until this completes,
        # readers see the journal at the previous state.
        new_hash = hash_entry_payload(entry)
        latest_payload = json.dumps(
            {
                "sequence_number": entry.sequence_number,
                "entry_hash": new_hash.hex(),
            },
            sort_keys=True,
            indent=2,
        )
        self._atomic_write_text(self._root / _LATEST_FILENAME, latest_payload)

        # Update in-memory cache
        self._count += 1
        self._latest_hash = new_hash

    # -- read path --

    def replay(self) -> Iterator[PrivacyBudgetEntry]:
        # Stream from disk; trust latest.json's count over directory
        # contents (orphan entries beyond the count are ignored).
        for seq in range(self._count):
            yield self._read_entry(seq)

    def latest_hash(self) -> bytes:
        return self._latest_hash

    def __len__(self) -> int:
        return self._count

    # -- internals --

    def _entry_path(self, sequence_number: int) -> Path:
        if sequence_number < 0:
            raise ValueError(
                f"sequence_number must be >= 0, got {sequence_number}"
            )
        # Zero-padded 6-digit filename. Sequence numbers > 999999
        # overflow the format; documented limitation. At ~1 spend/sec
        # this is 11+ days of journal — by which point a rotation
        # would have happened (out of v1 scope).
        if sequence_number >= 10 ** _ENTRY_DIGITS:
            raise ValueError(
                f"sequence_number {sequence_number} exceeds "
                f"FilesystemPrivacyBudgetStore's {10 ** _ENTRY_DIGITS}-entry "
                f"per-journal limit; rotate the journal root"
            )
        filename = f"{sequence_number:0{_ENTRY_DIGITS}d}.json"
        return self._root / _ENTRIES_DIRNAME / filename

    def _read_entry(self, sequence_number: int) -> PrivacyBudgetEntry:
        path = self._entry_path(sequence_number)
        if not path.exists():
            raise JournalCorruptionError(
                f"entries/{path.name} missing — journal references "
                f"sequence {sequence_number} but file is absent"
            )
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise JournalCorruptionError(
                f"entries/{path.name} unreadable or malformed: {exc}"
            ) from exc
        try:
            return PrivacyBudgetEntry.from_dict(data)
        except (TypeError, ValueError, KeyError) as exc:
            raise JournalCorruptionError(
                f"entries/{path.name} schema error: {exc}"
            ) from exc

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
