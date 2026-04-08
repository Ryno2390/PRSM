"""
PRSM Native Storage Module

Provides content-addressed blob storage, erasure-coded sharding, threshold
key management, and network-aware shard distribution — replacing IPFS as the
underlying storage substrate.

Public surface
--------------
ContentHash          — content-addressed identifier with algorithm-agility
ContentStore         — high-level facade (imported once implemented)

Singleton helpers
-----------------
get_content_store()  — return the global ContentStore instance (or None)
init_content_store() — initialize (or return existing) global ContentStore
close_content_store() — clear the global singleton

Exceptions
----------
StorageError, ContentNotFoundError, ShardIntegrityError,
ManifestError, KeyReconstructionError, PlacementError
"""

from __future__ import annotations

import os
from typing import Optional

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

from prsm.storage.content_store import ContentStore

_content_store: Optional[ContentStore] = None


def get_content_store() -> Optional[ContentStore]:
    """Get the global ContentStore instance. Returns None if not initialized."""
    return _content_store


def init_content_store(data_dir: str = "~/.prsm/storage", node_id: str = "") -> ContentStore:
    """Initialize the global ContentStore singleton."""
    global _content_store
    if _content_store is None:
        _content_store = ContentStore(data_dir=os.path.expanduser(data_dir), node_id=node_id)
    return _content_store


def close_content_store() -> None:
    """Clear the global ContentStore singleton."""
    global _content_store
    _content_store = None


__all__ = [
    # Models
    "AlgorithmID",
    "ContentDescriptor",
    "ContentHash",
    "KeyShare",
    "ReplicationPolicy",
    "RetrievalTicket",
    "ShardManifest",
    # Facade
    "ContentStore",
    # Singleton helpers
    "get_content_store",
    "init_content_store",
    "close_content_store",
    # Exceptions
    "ContentNotFoundError",
    "KeyReconstructionError",
    "ManifestError",
    "PlacementError",
    "ShardIntegrityError",
    "StorageError",
]
