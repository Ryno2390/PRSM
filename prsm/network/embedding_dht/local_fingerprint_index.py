"""
Embedding DHT — local binary-fingerprint provider index.

PRSM-PROV-1 Item 4 T4.9.

Per-node index of binary fingerprints THIS node can serve over the
DHT, keyed by ``(content_hash, fingerprint_kind)``. Parallel of
:class:`LocalEmbeddingIndex` for the four binary lanes (image-pHash,
audio-Chromaprint, video-multihash, structural).

Why a separate index instead of extending LocalEmbeddingIndex:

The two share storage shape but their record fields differ — embeddings
carry ``dimension`` / ``dtype`` / ``vector_b64`` (typed float32),
fingerprints carry ``payload_b64`` (opaque bytes) under a
``fingerprint_kind`` enum. Mixing the two into one record class would
require optional fields and conditional validation; keeping them
parallel keeps each record type fully validated at construction.

Storage shape: a JSON file at
``<root>/fingerprint_index.json`` containing a list of
:meth:`LocalFingerprintRecord.to_dict` entries. Distinct filename from
``embedding_index.json`` so an operator running both lanes from the
same root directory has two non-conflicting files.

Concurrency: same single-writer-per-node assumption as the embedding
lane. On-disk file is written atomically (.tmp + os.replace).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from prsm.network.embedding_dht.protocol import (
    ALLOWED_FINGERPRINT_KINDS,
    MAX_FINGERPRINT_PAYLOAD_BYTES,
    MalformedMessageError,
)


logger = logging.getLogger(__name__)


# Pinned filename for the on-disk index. Distinct from
# ``embedding_index.json`` (LocalEmbeddingIndex) and ``dht_index.json``
# (LocalManifestIndex) so an operator running all three from the same
# root directory has three non-conflicting files.
_INDEX_FILENAME = "fingerprint_index.json"

# Identifier-safety regex. content_hash and fingerprint_kind are stored
# as JSON keys; restricting their character set keeps the index file
# diff-friendly. fingerprint_kind values are pinned to
# ALLOWED_FINGERPRINT_KINDS but content_hash is creator-supplied.
_SAFE_KEY_PART = re.compile(r"^[A-Za-z0-9._/+:\-]+$")


@dataclass(frozen=True)
class LocalFingerprintRecord:
    """A single binary fingerprint the local node can serve.

    Mirrors the wire-format :class:`FingerprintResponse` minus
    per-request fields (``request_id``, ``protocol_version``). The
    split keeps storage decoupled from the wire format — a future
    protocol bump can add fields to ``FingerprintResponse`` without
    rewriting every record on disk.

    Attributes
    ----------
    content_hash:
        0x-prefixed hex; matches ProvenanceRegistry contentHash.
    fingerprint_kind:
        One of :data:`ALLOWED_FINGERPRINT_KINDS` — image-phash /
        audio-chromaprint / video-multihash / structural.
    payload_b64:
        Base64-encoded backend-specific fingerprint payload. Each
        backend's payload format is self-describing per kind.
    creator_id:
        node_id or DID of the original fingerprinter; used to scope
        the on-chain pubkey lookup at verification time.
    created_at:
        Unix epoch seconds. Bound into the signature.
    signature_b64:
        Base64-encoded Ed25519 signature, 64 bytes. Verifier checks
        against the canonical creator pubkey for ``content_hash``
        anchored on-chain via PublisherKeyAnchor.
    """

    content_hash: str
    fingerprint_kind: str
    payload_b64: str
    creator_id: str
    created_at: float
    signature_b64: str

    def __post_init__(self) -> None:
        _validate_key_part("content_hash", self.content_hash)
        if self.fingerprint_kind not in ALLOWED_FINGERPRINT_KINDS:
            raise ValueError(
                f"fingerprint_kind {self.fingerprint_kind!r} not in "
                f"{sorted(ALLOWED_FINGERPRINT_KINDS)}"
            )
        if not isinstance(self.payload_b64, str) or not self.payload_b64:
            raise ValueError("payload_b64 must be non-empty string")
        if not isinstance(self.signature_b64, str) or not self.signature_b64:
            raise ValueError("signature_b64 must be non-empty string")
        if not isinstance(self.creator_id, str) or not self.creator_id:
            raise ValueError("creator_id must be non-empty string")
        if not isinstance(self.created_at, (int, float)):
            raise ValueError(
                f"created_at must be numeric, got "
                f"{type(self.created_at).__name__}"
            )
        # Cheap sanity: payload_b64 must base64-decode to ≤ raw cap.
        # Catches index corruption + protects against an oversized
        # payload smuggled through a future from_dict path.
        try:
            decoded = base64.b64decode(self.payload_b64, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"payload_b64 not valid base64: {exc}") from exc
        if len(decoded) == 0:
            raise ValueError("payload_b64 decodes to zero bytes")
        if len(decoded) > MAX_FINGERPRINT_PAYLOAD_BYTES:
            raise ValueError(
                f"payload_b64 decodes to {len(decoded)} bytes; "
                f"exceeds MAX_FINGERPRINT_PAYLOAD_BYTES "
                f"({MAX_FINGERPRINT_PAYLOAD_BYTES})"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "content_hash": self.content_hash,
            "fingerprint_kind": self.fingerprint_kind,
            "payload_b64": self.payload_b64,
            "creator_id": self.creator_id,
            "created_at": self.created_at,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LocalFingerprintRecord":
        return cls(
            content_hash=str(data["content_hash"]),
            fingerprint_kind=str(data["fingerprint_kind"]),
            payload_b64=str(data["payload_b64"]),
            creator_id=str(data["creator_id"]),
            created_at=float(data["created_at"]),  # type: ignore[arg-type]
            signature_b64=str(data["signature_b64"]),
        )


class LocalFingerprintIndex:
    """In-memory + on-disk index of locally-servable binary fingerprints.

    Storage shape: a single JSON file at
    ``<root>/fingerprint_index.json`` containing a list of
    :meth:`LocalFingerprintRecord.to_dict` entries. The list form
    keeps the file roundtrip-stable when entries share the same
    ``content_hash`` under different ``fingerprint_kind`` values
    (a creator who re-fingerprints the same content under multiple
    backends).
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(
                f"LocalFingerprintIndex root {self._root} does not exist; "
                f"create the directory before constructing the index"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"LocalFingerprintIndex root {self._root} is not a directory"
            )

        # Internal map: (content_hash, fingerprint_kind) → record.
        self._entries: Dict[
            Tuple[str, str], LocalFingerprintRecord
        ] = {}

        index_path = self._root / _INDEX_FILENAME
        if index_path.exists():
            self._load(index_path)

    # -- public read API ----------------------------------------------------

    def lookup(
        self, content_hash: str, fingerprint_kind: str,
    ) -> Optional[LocalFingerprintRecord]:
        """Return the local record for ``(content_hash, fingerprint_kind)``,
        or None if not in the index.

        Defensively re-validates inputs — a caller passing an
        unsafe content_hash or unknown fingerprint_kind gets None
        rather than risking a bad-key lookup.
        """
        if not _is_safe_key_part(content_hash):
            return None
        if fingerprint_kind not in ALLOWED_FINGERPRINT_KINDS:
            return None
        return self._entries.get((content_hash, fingerprint_kind))

    def has(self, content_hash: str, fingerprint_kind: str) -> bool:
        return self.lookup(content_hash, fingerprint_kind) is not None

    def list_content_hashes(self) -> List[str]:
        """Sorted list of all known content_hashes (deduplicated
        across fingerprint_kinds)."""
        return sorted({ch for (ch, _k) in self._entries.keys()})

    def list_kinds_for(self, content_hash: str) -> List[str]:
        """Sorted list of fingerprint_kinds under which we have a
        fingerprint for ``content_hash``."""
        if not _is_safe_key_part(content_hash):
            return []
        return sorted(
            k for (ch, k) in self._entries.keys() if ch == content_hash
        )

    def list_keys(self) -> List[Tuple[str, str]]:
        """Sorted list of all (content_hash, fingerprint_kind) keys."""
        return sorted(self._entries.keys())

    def lookup_creator_by_content_hash(
        self, content_hash: str,
    ) -> Optional[str]:
        """Return the ``creator_id`` of any record matching
        ``content_hash``, regardless of fingerprint_kind. Used by the
        FingerprintDHT verifier (analogue of LocalEmbeddingIndex's
        same-name method) to resolve content_hash → creator-node-id →
        on-chain pubkey lookup, without the verifier needing to know
        which fingerprint kind the lookup originated from.

        If multiple kinds carry the same content_hash, the chosen
        record is deterministic across calls (sorted by
        fingerprint_kind). Records signed by *different* creator_ids
        under the same content_hash should never legitimately exist —
        registerContent binds a single creator on-chain — but if they
        do, the deterministic tiebreaker keeps the resolver behavior
        stable rather than oscillating.
        """
        if not _is_safe_key_part(content_hash):
            return None
        candidates = sorted(
            (k, r) for (ch, k), r in self._entries.items()
            if ch == content_hash
        )
        if not candidates:
            return None
        return candidates[0][1].creator_id

    def __contains__(self, key: Tuple[str, str]) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        ch, k = key
        return (
            isinstance(ch, str)
            and isinstance(k, str)
            and (ch, k) in self._entries
        )

    def __len__(self) -> int:
        return len(self._entries)

    # -- public write API ---------------------------------------------------

    def register(self, record: LocalFingerprintRecord) -> None:
        """Add or replace a fingerprint record. Persists after every
        mutation. Overwrites silently — the upstream uploader decides
        whether to overwrite based on creator + created_at."""
        if not isinstance(record, LocalFingerprintRecord):
            raise TypeError(
                f"register() expects LocalFingerprintRecord, got "
                f"{type(record).__name__}"
            )
        self._entries[
            (record.content_hash, record.fingerprint_kind)
        ] = record
        self._persist()

    def unregister(self, content_hash: str, fingerprint_kind: str) -> bool:
        """Remove an entry. Returns True if it was present, False
        otherwise. Persists on a True result."""
        if not _is_safe_key_part(content_hash):
            return False
        if fingerprint_kind not in ALLOWED_FINGERPRINT_KINDS:
            return False
        key = (content_hash, fingerprint_kind)
        if key in self._entries:
            del self._entries[key]
            self._persist()
            return True
        return False

    # -- internals ---------------------------------------------------------

    def _load(self, index_path: Path) -> None:
        """Load the on-disk index; reset to empty on parse failure.
        Individual entries that fail validation are dropped (one
        corrupt record can't DoS the whole index)."""
        try:
            data = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "fingerprint_index.json at %s unreadable (%s); "
                "starting empty",
                index_path, exc,
            )
            return

        if not isinstance(data, list):
            logger.warning(
                "fingerprint_index.json at %s has unexpected shape (%s); "
                "starting empty",
                index_path, type(data).__name__,
            )
            return

        for raw in data:
            if not isinstance(raw, dict):
                logger.warning(
                    "fingerprint_index.json: dropping non-dict entry %r",
                    raw,
                )
                continue
            try:
                record = LocalFingerprintRecord.from_dict(raw)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "fingerprint_index.json: dropping invalid entry "
                    "(%s): %r", exc, raw.get("content_hash"),
                )
                continue
            self._entries[
                (record.content_hash, record.fingerprint_kind)
            ] = record

    def _persist(self) -> None:
        """Atomic write: .tmp + os.replace. Same idiom as
        LocalEmbeddingIndex / LocalManifestIndex."""
        index_path = self._root / _INDEX_FILENAME
        tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        # Sort the serialized list by (content_hash, fingerprint_kind)
        # so the on-disk file diffs cleanly across writes.
        sorted_records = sorted(
            self._entries.values(),
            key=lambda r: (r.content_hash, r.fingerprint_kind),
        )
        text = json.dumps(
            [r.to_dict() for r in sorted_records],
            sort_keys=True,
            indent=2,
        )
        with open(tmp, "wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, index_path)


# -- key validation ---------------------------------------------------------


def _is_safe_key_part(value: object) -> bool:
    """Soft check used for lookup/unregister APIs — returns False
    rather than raising, so callers passing bad keys get a clean
    miss instead of a stack trace."""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and bool(_SAFE_KEY_PART.match(value))
    )


def _validate_key_part(name: str, value: object) -> None:
    """Hard check used inside dataclass __post_init__ — raises on
    invalid input so a corrupt record never makes it into the index."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty string")
    if value in {".", ".."}:
        raise ValueError(f"{name} must not be {value!r}")
    if not _SAFE_KEY_PART.match(value):
        raise ValueError(
            f"{name}={value!r} contains characters outside the safe set "
            f"[A-Za-z0-9._/+:-]"
        )


__all__ = [
    "LocalFingerprintIndex",
    "LocalFingerprintRecord",
]
