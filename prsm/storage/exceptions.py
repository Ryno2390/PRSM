"""
Storage module exceptions.

Hierarchy:
    StorageError
    ├── ContentNotFoundError
    ├── ShardIntegrityError
    ├── ManifestError
    ├── KeyReconstructionError
    └── PlacementError
"""

from __future__ import annotations


class StorageError(Exception):
    """Base exception for all storage-related errors."""


class ContentNotFoundError(StorageError):
    """Raised when requested content cannot be located in the network."""

    def __init__(self, content_hash: str) -> None:
        self.content_hash = content_hash
        super().__init__(f"Content not found: {content_hash}")


class ShardIntegrityError(StorageError):
    """Raised when a retrieved shard fails its integrity check."""

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Shard integrity check failed: expected={expected}, actual={actual}"
        )


class ManifestError(StorageError):
    """Raised when a shard manifest is missing, corrupt, or invalid."""


class KeyReconstructionError(StorageError):
    """Raised when threshold key reconstruction fails."""


class PlacementError(StorageError):
    """Raised when content cannot be placed on enough distinct nodes."""

    def __init__(self, reason: str, min_nodes_needed: int) -> None:
        self.reason = reason
        self.min_nodes_needed = min_nodes_needed
        super().__init__(
            f"Placement failed (min_nodes_needed={min_nodes_needed}): {reason}"
        )
