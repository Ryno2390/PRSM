"""
Pure-Python Merkle Tree
=======================

Drop-in replacement for the ``merkletools`` PyPI package, which depends
on ``pysha3`` (a C extension that fails to build on Python >= 3.13).

This module re-exports the :class:`MerkleTree` that already existed in
``prsm.compute.collaboration.security.integrity_validator`` and adds a
thin :class:`MerkleTools` compatibility wrapper so that all existing
call-sites (``add_leaf / make_tree / get_merkle_root``) keep working
without any external dependency.
"""

import hashlib
from typing import List, Union


class MerkleTree:
    """
    Pure-Python Merkle tree for efficient integrity verification.

    Supports SHA-256 and SHA3-256 hash algorithms.
    """

    def __init__(self, hash_algorithm: str = "sha256"):
        self.hash_algorithm = hash_algorithm
        self.leaf_nodes: List[str] = []
        self.tree_levels: List[List[str]] = []
        self.is_built = False

    def _hash_data(self, data: Union[str, bytes]) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.hash_algorithm == "sha3_256":
            return hashlib.sha3_256(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

    def add_leaf(self, data: Union[str, bytes]) -> int:
        """Add a leaf node and return its index."""
        leaf_hash = self._hash_data(data)
        self.leaf_nodes.append(leaf_hash)
        self.is_built = False
        return len(self.leaf_nodes) - 1

    def build_tree(self) -> None:
        """Build the complete Merkle tree bottom-up."""
        if not self.leaf_nodes:
            raise ValueError("No leaf nodes to build tree")

        self.tree_levels = []
        current_level = self.leaf_nodes.copy()

        while len(current_level) > 1:
            self.tree_levels.append(current_level)
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent_hash = self._hash_data(left + right)
                next_level.append(parent_hash)
            current_level = next_level

        self.tree_levels.append(current_level)
        self.is_built = True

    def get_root(self) -> str:
        """Return the Merkle root hash."""
        if not self.is_built:
            self.build_tree()
        if not self.tree_levels:
            raise ValueError("Tree is empty")
        return self.tree_levels[-1][0]

    def get_proof(self, leaf_index: int) -> List[str]:
        """Return the Merkle proof (sibling hashes) for *leaf_index*."""
        if not self.is_built:
            self.build_tree()
        if leaf_index >= len(self.leaf_nodes):
            raise ValueError(f"Leaf index {leaf_index} out of range")

        proof: List[str] = []
        current_index = leaf_index
        for level in self.tree_levels[:-1]:
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1
            if sibling_index < len(level):
                proof.append(level[sibling_index])
            else:
                proof.append(level[current_index])
            current_index //= 2
        return proof

    def verify_proof(self, leaf_hash: str, leaf_index: int,
                     proof: List[str], root_hash: str) -> bool:
        """Verify a Merkle proof against *root_hash*."""
        current_hash = leaf_hash
        current_index = leaf_index
        for sibling_hash in proof:
            if current_index % 2 == 0:
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = self._hash_data(combined)
            current_index //= 2
        return current_hash == root_hash

    def detect_tampering(self, expected_leaves: List[str]) -> List[int]:
        """Return indices of leaves that differ from *expected_leaves*."""
        if len(expected_leaves) != len(self.leaf_nodes):
            raise ValueError("Leaf count mismatch")
        return [i for i, (e, a) in enumerate(zip(expected_leaves, self.leaf_nodes)) if e != a]


class MerkleTools:
    """Drop-in replacement for the ``merkletools`` PyPI package.

    Exposes the same API surface used in the PRSM codebase:
    ``add_leaf``, ``make_tree``, ``get_merkle_root``.
    """

    def __init__(self) -> None:
        self._tree = MerkleTree()

    def add_leaf(self, data: str) -> None:
        self._tree.add_leaf(data)

    def make_tree(self) -> None:
        self._tree.build_tree()

    def get_merkle_root(self) -> str:
        return self._tree.get_root()
