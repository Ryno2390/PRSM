"""
Embedding DHT — local provider index.

PRSM-PROV-1 Item 3 Task 2.

Per-node index of embedding vectors THIS node can serve over the DHT,
keyed by ``(content_hash, model_id)``. Unlike LocalManifestIndex
(Phase 3.x.5), embeddings have no per-model on-disk directory tree —
the JSON index file is itself the storage. Vectors are persisted as
base64-encoded float32 bytes alongside their metadata + creator
signature.

Why both content_hash AND model_id in the key:

Embedding spaces are model-specific. A vector from
``openai/text-embedding-ada-002`` (1536-dim) shares no meaningful
metric with one from ``sentence-transformers/all-MiniLM-L6-v2``
(384-dim). The local index therefore tracks each (hash, model)
pair independently, just as the wire protocol does (see
``protocol.py``).

Concurrency: same single-writer-per-node assumption as Phase 3.x.2 /
3.x.4 / 3.x.5. No cross-process locking. The on-disk file is
written atomically (.tmp + os.replace).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from prsm.network.embedding_dht.protocol import (
    ALLOWED_DTYPES,
    MAX_VECTOR_DIMENSION,
    MalformedMessageError,
)


logger = logging.getLogger(__name__)


# Pinned filename for the on-disk index. Distinct from
# ``dht_index.json`` (LocalManifestIndex) so an operator running both
# DHTs from the same root directory has two non-conflicting files.
_INDEX_FILENAME = "embedding_index.json"

# Identifier-safety regex — same as Phase 3.x.5 for operational
# consistency and defense in depth. content_hash and model_id are
# stored as JSON keys; restricting their character set keeps the
# index file diff-friendly and avoids edge cases in path-derived
# operations downstream.
_SAFE_KEY_PART = re.compile(r"^[A-Za-z0-9._/+:\-]+$")
_RESERVED_NAMES = frozenset({".", ".."})


@dataclass(frozen=True)
class LocalEmbeddingRecord:
    """A single embedding the local node can serve.

    Mirrors the wire-format ``EmbeddingResponse`` minus per-request
    fields (``request_id``, ``protocol_version``). The split keeps
    storage decoupled from the wire format — a future protocol
    bump can add fields to ``EmbeddingResponse`` without rewriting
    every record on disk.
    """

    content_hash: str
    model_id: str
    dimension: int
    dtype: str
    vector_b64: str
    creator_id: str
    created_at: float
    signature_b64: str

    def __post_init__(self) -> None:
        _validate_key_part("content_hash", self.content_hash)
        _validate_key_part("model_id", self.model_id)
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError(
                f"dimension must be positive int, got {self.dimension!r}"
            )
        if self.dimension > MAX_VECTOR_DIMENSION:
            raise ValueError(
                f"dimension {self.dimension} exceeds "
                f"MAX_VECTOR_DIMENSION={MAX_VECTOR_DIMENSION}"
            )
        if self.dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"dtype {self.dtype!r} not in {sorted(ALLOWED_DTYPES)}"
            )
        if not isinstance(self.vector_b64, str) or not self.vector_b64:
            raise ValueError("vector_b64 must be non-empty string")
        if not isinstance(self.signature_b64, str) or not self.signature_b64:
            raise ValueError("signature_b64 must be non-empty string")
        if not isinstance(self.creator_id, str) or not self.creator_id:
            raise ValueError("creator_id must be non-empty string")
        if not isinstance(self.created_at, (int, float)):
            raise ValueError(
                f"created_at must be numeric, got "
                f"{type(self.created_at).__name__}"
            )
        # Cheap sanity: vector_b64 must base64-decode to dimension*4
        # bytes for float32. Catches index corruption early.
        try:
            decoded = base64.b64decode(self.vector_b64, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"vector_b64 not valid base64: {exc}") from exc
        if len(decoded) != self.dimension * 4:
            raise ValueError(
                f"vector_b64 decodes to {len(decoded)} bytes; "
                f"expected dimension*4={self.dimension * 4} for float32"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "content_hash": self.content_hash,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "vector_b64": self.vector_b64,
            "creator_id": self.creator_id,
            "created_at": self.created_at,
            "signature_b64": self.signature_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LocalEmbeddingRecord":
        return cls(
            content_hash=str(data["content_hash"]),
            model_id=str(data["model_id"]),
            dimension=int(data["dimension"]),  # type: ignore[arg-type]
            dtype=str(data["dtype"]),
            vector_b64=str(data["vector_b64"]),
            creator_id=str(data["creator_id"]),
            created_at=float(data["created_at"]),  # type: ignore[arg-type]
            signature_b64=str(data["signature_b64"]),
        )


class LocalEmbeddingIndex:
    """In-memory + on-disk index of locally-servable embeddings.

    Storage shape: a single JSON file at
    ``<root>/embedding_index.json`` containing a list of
    ``LocalEmbeddingRecord.to_dict()`` entries. The list form keeps
    the file roundtrip-stable when entries share the same
    ``content_hash`` under different ``model_id``s — a dict keyed by
    content_hash would collide.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(
                f"LocalEmbeddingIndex root {self._root} does not exist; "
                f"create the directory before constructing the index"
            )
        if not self._root.is_dir():
            raise NotADirectoryError(
                f"LocalEmbeddingIndex root {self._root} is not a directory"
            )

        # Internal map: (content_hash, model_id) → LocalEmbeddingRecord.
        self._entries: Dict[
            Tuple[str, str], LocalEmbeddingRecord
        ] = {}

        index_path = self._root / _INDEX_FILENAME
        if index_path.exists():
            self._load(index_path)
        # No filesystem-walk fallback like LocalManifestIndex — embeddings
        # have no on-disk artifact outside the index file. A missing or
        # corrupt index just yields an empty index, which is the
        # correct fresh-start behavior. (`.tmp` files on crash recovery
        # are reaped at next persist.)

    # -- public read API ----------------------------------------------------

    def lookup(
        self, content_hash: str, model_id: str
    ) -> Optional[LocalEmbeddingRecord]:
        """Return the local record for ``(content_hash, model_id)``,
        or None if not in the index.

        Defensively re-validates inputs against the safe-key regex —
        a caller passing an unsafe id gets None rather than risking a
        bad-key lookup.
        """
        if not _is_safe_key_part(content_hash):
            return None
        if not _is_safe_key_part(model_id):
            return None
        return self._entries.get((content_hash, model_id))

    def has(self, content_hash: str, model_id: str) -> bool:
        return self.lookup(content_hash, model_id) is not None

    def list_content_hashes(self) -> List[str]:
        """Sorted list of all known content_hashes (deduplicated
        across model_ids). Useful for ContentIndex coordination."""
        return sorted({ch for (ch, _m) in self._entries.keys()})

    def list_models_for(self, content_hash: str) -> List[str]:
        """Sorted list of model_ids under which we have an embedding
        for ``content_hash``."""
        if not _is_safe_key_part(content_hash):
            return []
        return sorted(
            m for (ch, m) in self._entries.keys() if ch == content_hash
        )

    def list_keys(self) -> List[Tuple[str, str]]:
        """Sorted list of all (content_hash, model_id) keys."""
        return sorted(self._entries.keys())

    def __contains__(self, key: Tuple[str, str]) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        ch, m = key
        return (
            isinstance(ch, str)
            and isinstance(m, str)
            and (ch, m) in self._entries
        )

    def __len__(self) -> int:
        return len(self._entries)

    # -- public write API ---------------------------------------------------

    def register(self, record: LocalEmbeddingRecord) -> None:
        """Add or replace an embedding record.

        Overwrites silently — the upstream uploader is responsible for
        deciding whether to overwrite an existing entry (typically:
        only on a creator-signed update with a higher created_at).
        Persists after every mutation.
        """
        if not isinstance(record, LocalEmbeddingRecord):
            raise TypeError(
                f"register() expects LocalEmbeddingRecord, got "
                f"{type(record).__name__}"
            )
        self._entries[(record.content_hash, record.model_id)] = record
        self._persist()

    def unregister(self, content_hash: str, model_id: str) -> bool:
        """Remove an entry. Returns True if it was present, False
        otherwise. Persists on a True result."""
        if not _is_safe_key_part(content_hash) or not _is_safe_key_part(
            model_id
        ):
            return False
        key = (content_hash, model_id)
        if key in self._entries:
            del self._entries[key]
            self._persist()
            return True
        return False

    # -- internals ---------------------------------------------------------

    def _load(self, index_path: Path) -> None:
        """Try to load the on-disk index; reset to empty on parse
        failure. Drops individual entries that fail validation so
        one corrupt record cannot deny-of-service the whole index."""
        try:
            data = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "embedding_index.json at %s unreadable (%s); "
                "starting empty",
                index_path, exc,
            )
            return

        if not isinstance(data, list):
            logger.warning(
                "embedding_index.json at %s has unexpected shape (%s); "
                "starting empty",
                index_path, type(data).__name__,
            )
            return

        for raw in data:
            if not isinstance(raw, dict):
                logger.warning(
                    "embedding_index.json: dropping non-dict entry %r",
                    raw,
                )
                continue
            try:
                record = LocalEmbeddingRecord.from_dict(raw)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "embedding_index.json: dropping invalid entry "
                    "(%s): %r", exc, raw.get("content_hash"),
                )
                continue
            # _validate_key_part is enforced inside __post_init__; if
            # it survived the dataclass it's safe to insert.
            self._entries[(record.content_hash, record.model_id)] = record

    def _persist(self) -> None:
        """Atomic write: .tmp + os.replace. Same idiom as Phase 3.x.2 /
        3.x.4 / 3.x.5 stores."""
        index_path = self._root / _INDEX_FILENAME
        tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        # Sort the serialized list by (content_hash, model_id) so the
        # on-disk file diffs cleanly across writes and remains stable
        # under git.
        sorted_records = sorted(
            self._entries.values(),
            key=lambda r: (r.content_hash, r.model_id),
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
        and bool(_SAFE_KEY_PART.fullmatch(value))
        and value not in _RESERVED_NAMES
    )


def _validate_key_part(name: str, value: object) -> None:
    """Hard check used inside ``LocalEmbeddingRecord.__post_init__``."""
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if not _SAFE_KEY_PART.fullmatch(value):
        raise ValueError(
            f"{name}={value!r} unsafe: must match {_SAFE_KEY_PART.pattern}"
        )
    if value in _RESERVED_NAMES:
        raise ValueError(
            f"{name}={value!r} is a reserved name"
        )
