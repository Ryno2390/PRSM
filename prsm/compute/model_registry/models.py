"""
Model registry data models — Phase 3.x.2 Task 1.

Defines the wire format for signed model manifests:

- ``ManifestShardEntry`` — one row per shard (sha256-committed tensor data)
- ``ModelManifest`` — collection of entries plus publisher identity, timestamp,
  and Ed25519 signature

These dataclasses are persistence-format-agnostic: the same manifest bytes
serialize identically through filesystem JSON (``FilesystemModelRegistry``
in Task 4) and any future DHT / on-chain transport (Phase 3.x.3+).

Signing pattern mirrors ``prsm.compute.inference.models.InferenceReceipt`` so
the project has exactly one canonical-bytes idiom for Ed25519-signed
artifacts. Verifiers can audit signing payloads with the same mental model
across both surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------
# Per-shard entry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestShardEntry:
    """One shard's metadata in a model manifest.

    The ``sha256`` digest is the cryptographic commit to the shard's
    tensor bytes. Every registry implementation MUST verify this digest
    against the actual shard data on read; a mismatch is the failure
    signal that an operator (or filesystem corruption) has tampered
    with the model between registration and use.

    ``size_bytes`` is included in the signing payload alongside sha256
    so a length-extension attack on the digest can't fly.
    """

    shard_id: str
    shard_index: int
    tensor_shape: Tuple[int, ...]
    sha256: str          # hex digest of tensor_data bytes
    size_bytes: int

    def __post_init__(self) -> None:
        # Frozen dataclass — coerce types after init.
        if not isinstance(self.tensor_shape, tuple):
            object.__setattr__(self, "tensor_shape", tuple(self.tensor_shape))
        if not isinstance(self.shard_index, int):
            object.__setattr__(self, "shard_index", int(self.shard_index))
        if not isinstance(self.size_bytes, int):
            object.__setattr__(self, "size_bytes", int(self.size_bytes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "tensor_shape": list(self.tensor_shape),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestShardEntry":
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in accepted})


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


# Schema version namespace for the canonical signing payload. Bump when the
# payload format changes — verifiers checking signatures across protocol
# versions will refuse to validate manifests whose declared schema version
# they don't understand.
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_SIGNING_DOMAIN = b"prsm-model-manifest:v1"


@dataclass(frozen=True)
class ModelManifest:
    """Signed manifest committing to a model's identity + shard contents.

    Per the Phase 3.x.2 trust model: the publisher signs this manifest
    under their NodeIdentity Ed25519 key. Any verifier with the
    publisher's public key can independently confirm:

      1. The manifest hasn't been tampered with (signature verifies)
      2. Every shard's actual bytes match the sha256 entry (registry
         enforces this on read)

    The manifest is the wire format. Filesystem layout (Task 4),
    DHT transport (deferred), and any future on-chain anchor all
    treat ``json.dumps(manifest.to_dict(), sort_keys=True)`` as the
    canonical serialization unit.

    Frozen because manifests should be immutable once signed; signing
    creates a new instance via ``dataclasses.replace`` (see Task 2).
    """

    model_id: str
    model_name: str
    publisher_node_id: str
    total_shards: int
    shards: Tuple[ManifestShardEntry, ...]
    published_at: float
    schema_version: int = MANIFEST_SCHEMA_VERSION
    publisher_signature: bytes = b""

    def __post_init__(self) -> None:
        # Coerce shards to a sorted-by-shard_index tuple so the canonical
        # signing payload is independent of construction order. Without
        # this, two manifests with the same shards in different orders
        # would produce different signing payloads → different
        # signatures → false negatives at verify time.
        if not isinstance(self.shards, tuple):
            shards_tuple = tuple(
                s if isinstance(s, ManifestShardEntry) else ManifestShardEntry.from_dict(s)
                for s in self.shards
            )
        else:
            shards_tuple = tuple(
                s if isinstance(s, ManifestShardEntry) else ManifestShardEntry.from_dict(s)
                for s in self.shards
            )
        # Sort by shard_index for canonical order
        shards_sorted = tuple(sorted(shards_tuple, key=lambda s: s.shard_index))
        object.__setattr__(self, "shards", shards_sorted)

        if not isinstance(self.total_shards, int):
            object.__setattr__(self, "total_shards", int(self.total_shards))
        if not isinstance(self.published_at, float):
            object.__setattr__(self, "published_at", float(self.published_at))
        if not isinstance(self.schema_version, int):
            object.__setattr__(self, "schema_version", int(self.schema_version))

    def signing_payload(self) -> bytes:
        """Canonical bytes used for ``publisher_signature`` generation/verification.

        Excludes ``publisher_signature`` itself (would be circular).
        Field order is fixed and pinned by ``MANIFEST_SCHEMA_VERSION``;
        do not reorder without bumping the schema version.

        Includes the schema version in the payload so a downgrade
        attack (re-signing a v2 manifest as v1) doesn't silently
        succeed against a v1-only verifier.
        """
        parts: List[str] = [
            MANIFEST_SIGNING_DOMAIN.decode("ascii"),
            str(self.schema_version),
            self.model_id,
            self.model_name,
            self.publisher_node_id,
            str(self.total_shards),
            f"{self.published_at:.6f}",
        ]
        # One line per shard, in canonical (shard_index ascending) order.
        # Includes shard_id + sha256 + size_bytes — these together commit
        # to which shard occupies which slot AND its exact bytes.
        for entry in self.shards:
            shape_str = ",".join(str(d) for d in entry.tensor_shape)
            parts.append(
                f"{entry.shard_index}:{entry.shard_id}:"
                f"{entry.sha256}:{entry.size_bytes}:{shape_str}"
            )
        return "\n".join(parts).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "publisher_node_id": self.publisher_node_id,
            "total_shards": self.total_shards,
            "shards": [s.to_dict() for s in self.shards],
            "published_at": self.published_at,
            "schema_version": self.schema_version,
            "publisher_signature": self.publisher_signature.hex(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        d = dict(data)
        # Coerce shards: list of dicts → list of ManifestShardEntry
        if "shards" in d:
            d["shards"] = [
                s if isinstance(s, ManifestShardEntry) else ManifestShardEntry.from_dict(s)
                for s in d["shards"]
            ]
        # Coerce signature: hex string → bytes
        if "publisher_signature" in d and isinstance(d["publisher_signature"], str):
            d["publisher_signature"] = bytes.fromhex(d["publisher_signature"])
        accepted = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in accepted})
