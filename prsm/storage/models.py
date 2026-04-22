"""
Storage module data models.

Provides content-addressed identifiers and manifest types used throughout
the native storage subsystem.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

class AlgorithmID(enum.IntEnum):
    """Algorithm prefix byte for ContentHash serialisation."""
    SHA256 = 0x01
    SHA3_256 = 0x02   # reserved
    BLAKE3 = 0x03     # reserved


# ---------------------------------------------------------------------------
# ContentHash
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContentHash:
    """
    Content-addressed identifier with algorithm-agility prefix byte.

    Wire format (hex string): <1-byte algorithm_id><digest>
    Example (SHA-256): "01" + 64 hex chars = 66 characters total.
    """
    algorithm_id: AlgorithmID
    digest: bytes

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        data: bytes,
        algorithm: AlgorithmID = AlgorithmID.SHA256,
    ) -> "ContentHash":
        """Hash *data* with the requested algorithm and return a ContentHash."""
        if algorithm == AlgorithmID.SHA256:
            digest = hashlib.sha256(data).digest()
        elif algorithm == AlgorithmID.SHA3_256:
            digest = hashlib.sha3_256(data).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm!r}")
        return cls(algorithm_id=algorithm, digest=digest)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def hex(self) -> str:
        """Serialise to hex string: 2-char algorithm prefix + hex digest."""
        prefix = f"{int(self.algorithm_id):02x}"
        return prefix + self.digest.hex()

    @classmethod
    def from_hex(cls, hex_str: str) -> "ContentHash":
        """Deserialise from a hex string produced by :meth:`hex`."""
        if len(hex_str) < 4:
            raise ValueError(
                f"ContentHash hex string too short: {hex_str!r}"
            )
        algorithm_byte = int(hex_str[:2], 16)
        try:
            algorithm_id = AlgorithmID(algorithm_byte)
        except ValueError:
            raise ValueError(
                f"Unknown algorithm id byte 0x{algorithm_byte:02x}"
            )
        digest = bytes.fromhex(hex_str[2:])
        return cls(algorithm_id=algorithm_id, digest=digest)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # noqa: D105
        return self.hex()

    def __hash__(self) -> int:  # frozen=True generates this, but be explicit
        return hash((self.algorithm_id, self.digest))

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, ContentHash):
            return NotImplemented
        return self.algorithm_id == other.algorithm_id and self.digest == other.digest


# ---------------------------------------------------------------------------
# ShardManifest
# ---------------------------------------------------------------------------


class ShardingMode(enum.IntEnum):
    """How shards relate to the original content.

    REPLICATION: each shard is an independent chunk of the content;
        reassembly concatenates every shard in order. Tier A default.
    ERASURE: shards are Reed-Solomon-coded; any `erasure_params.k` of the
        `erasure_params.n` shards reconstruct the content. Required for
        Phase 7-storage Tier B / Tier C durability.
    """

    REPLICATION = 0
    ERASURE = 1


@dataclass
class ErasureParams:
    """Reed-Solomon parameters for the ERASURE sharding mode.

    `k` is the reconstruction threshold; `n` is the total number of
    shards produced. `payload_bytes` captures the original plaintext
    length so reassembly can strip the last-block padding; `shard_bytes`
    is the length of each erasure-coded shard's raw bytes; `payload_sha256`
    is the hex digest of the original plaintext, used as the cross-shard
    integrity check at reassembly.
    """

    k: int
    n: int
    payload_bytes: int
    shard_bytes: int
    payload_sha256: str


@dataclass
class ShardManifest:
    """Maps a content hash to its ordered list of shard hashes."""
    content_hash: ContentHash
    shard_hashes: List[ContentHash]
    total_size: int
    shard_size: int
    algorithm_id: AlgorithmID
    created_at: float
    replication_factor: int
    owner_node_id: str
    visibility: str = "public"
    # Phase 7-storage: optional erasure-mode fields. None for the
    # replication path, preserving JSON-roundtrip equivalence for
    # existing Tier A manifests.
    sharding_mode: ShardingMode = ShardingMode.REPLICATION
    erasure_params: Optional[ErasureParams] = None

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the manifest to a JSON string."""
        body = {
            "content_hash": self.content_hash.hex(),
            "shard_hashes": [sh.hex() for sh in self.shard_hashes],
            "total_size": self.total_size,
            "shard_size": self.shard_size,
            "algorithm_id": int(self.algorithm_id),
            "created_at": self.created_at,
            "replication_factor": self.replication_factor,
            "owner_node_id": self.owner_node_id,
            "visibility": self.visibility,
        }
        if self.sharding_mode is not ShardingMode.REPLICATION:
            body["sharding_mode"] = int(self.sharding_mode)
        if self.erasure_params is not None:
            body["erasure_params"] = {
                "k": self.erasure_params.k,
                "n": self.erasure_params.n,
                "payload_bytes": self.erasure_params.payload_bytes,
                "shard_bytes": self.erasure_params.shard_bytes,
                "payload_sha256": self.erasure_params.payload_sha256,
            }
        return json.dumps(body)

    @classmethod
    def from_json(cls, json_str: str) -> "ShardManifest":
        """Deserialise a manifest from a JSON string."""
        from prsm.storage.exceptions import ManifestError

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid manifest JSON: {exc}") from exc

        try:
            erasure_raw = data.get("erasure_params")
            erasure_params = (
                ErasureParams(**erasure_raw) if erasure_raw is not None else None
            )
            mode = ShardingMode(data.get("sharding_mode", int(ShardingMode.REPLICATION)))
            return cls(
                content_hash=ContentHash.from_hex(data["content_hash"]),
                shard_hashes=[ContentHash.from_hex(h) for h in data["shard_hashes"]],
                total_size=data["total_size"],
                shard_size=data["shard_size"],
                algorithm_id=AlgorithmID(data["algorithm_id"]),
                created_at=data["created_at"],
                replication_factor=data["replication_factor"],
                owner_node_id=data["owner_node_id"],
                visibility=data.get("visibility", "public"),
                sharding_mode=mode,
                erasure_params=erasure_params,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Malformed manifest data: {exc}") from exc


# ---------------------------------------------------------------------------
# KeyShare
# ---------------------------------------------------------------------------

@dataclass
class KeyShare:
    """A single share of a threshold-split encryption key."""
    content_hash: ContentHash
    share_index: int
    share_data: bytes
    threshold: int
    total_shares: int
    algorithm_id: int  # 0x01 = AES-256-GCM


# ---------------------------------------------------------------------------
# ReplicationPolicy
# ---------------------------------------------------------------------------

@dataclass
class ReplicationPolicy:
    """Policy constraints that govern how content is replicated."""
    replication_factor: int
    min_asn_diversity: int = 2
    owner_excluded: bool = True
    key_shard_separation: bool = True
    degraded_constraints: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ContentDescriptor
# ---------------------------------------------------------------------------

@dataclass
class ContentDescriptor:
    """
    Full descriptor for a piece of content stored in the network.

    Combines placement metadata, key-share holders, replication policy,
    and cryptographic provenance.
    """
    content_hash: ContentHash
    manifest_holders: List[str]
    key_share_holders: List[str]
    contract_key_share_holders: List[str]
    shard_map: Dict[str, List[str]]
    replication_policy: ReplicationPolicy
    visibility: str
    epoch: int
    version: int
    owner_node_id: str
    contract_pubkey: bytes
    signature: bytes
    signer_type: str  # "owner" or "contract"
    created_at: float
    updated_at: float


# ---------------------------------------------------------------------------
# RetrievalTicket
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTicket:
    """
    Short-lived authorisation token permitting a node to retrieve content.

    Issued by a manifest holder after verifying the requester's access rights.
    """
    content_hash: ContentHash
    requester_node_id: str
    epoch: int
    issued_at: float
    expires_at: float
    nonce: str
    issuer_signature: bytes
