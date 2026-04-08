"""
PRSM Native Storage Module

Provides content-addressed blob storage, erasure-coded sharding, threshold
key management, and network-aware shard distribution — replacing IPFS as the
underlying storage substrate.

Public surface
--------------
ContentHash          — content-addressed identifier with algorithm-agility
ContentStore         — high-level facade (imported once implemented)

Exceptions
----------
StorageError, ContentNotFoundError, ShardIntegrityError,
ManifestError, KeyReconstructionError, PlacementError
"""

from __future__ import annotations

from prsm.storage.models import (
    AlgorithmID,
    ContentDescriptor,
    ContentHash,
    KeyShare,
    ReplicationPolicy,
    RetrievalTicket,
    ShardManifest,
)
from prsm.storage.exceptions import (
    ContentNotFoundError,
    KeyReconstructionError,
    ManifestError,
    PlacementError,
    ShardIntegrityError,
    StorageError,
)

# ContentStore will be added in a later task; expose a sentinel so importers
# can reference the name without an ImportError.
ContentStore = None  # type: ignore[assignment]

__all__ = [
    # Models
    "AlgorithmID",
    "ContentDescriptor",
    "ContentHash",
    "KeyShare",
    "ReplicationPolicy",
    "RetrievalTicket",
    "ShardManifest",
    # Facade (stub until Task 6)
    "ContentStore",
    # Exceptions
    "ContentNotFoundError",
    "KeyReconstructionError",
    "ManifestError",
    "PlacementError",
    "ShardIntegrityError",
    "StorageError",
]
