"""Phase 3.1 Task 5: canonical receipt encoding + Merkle tree.

Locks the Python↔Solidity canonical-form parity that Tasks 6-8 depend
on for commit/finalize/challenge flows to work end-to-end.

Two guarantees:

1. **Canonical leaf encoding.** A BatchedReceipt converts to a
   ReceiptLeaf whose `keccak256(abi.encode(leaf))` matches exactly
   what the Solidity BatchSettlementRegistry's `_hashLeaf` computes
   on chain. Field order + types + hash discipline are locked.

2. **Merkle tree + proofs.** OpenZeppelin's sorted-pair convention:
   at each internal node, the two children are sorted ascending
   before hashing. Proofs are position-free lists of sibling hashes
   that verify via the same sorted-pair fold. Odd nodes at a layer
   are promoted as-is (no duplication).

Cross-reference: the on-chain `ReceiptLeaf` struct + `_hashLeaf` function
live in contracts/contracts/BatchSettlementRegistry.sol. Field order
in that struct MUST match `_LEAF_SIGNATURE` below; mismatch produces
silently-different roots that break settlement.
"""
from __future__ import annotations

from base64 import b64decode
from dataclasses import dataclass
from typing import List

from eth_abi import encode as abi_encode
from eth_utils import keccak

from prsm.settlement.accumulator import BatchedReceipt

# The ABI type signature for ReceiptLeaf. MUST stay in lockstep with
# contracts/contracts/BatchSettlementRegistry.sol. Any field-order or
# type change here requires the same change there (and the Solidity
# test suite will catch the parity break).
_LEAF_SIGNATURE = "(bytes32,uint32,bytes32,bytes32,bytes32,uint64,uint128,bytes32)"


# Bounds for integer fields — match the Solidity uint widths.
_MAX_UINT32 = 2**32 - 1
_MAX_UINT64 = 2**64 - 1
_MAX_UINT128 = 2**128 - 1


@dataclass(frozen=True)
class ReceiptLeaf:
    """Canonical on-chain leaf form for a Phase 2 ShardExecutionReceipt.

    Field order locked to the Solidity struct — the ABI-encoder relies
    on it. All bytes32 fields are 32-byte `bytes`; int fields are
    native Python ints bounded by their declared Solidity uint width.
    """
    job_id_hash: bytes            # bytes32 — keccak256(utf8(job_id))
    shard_index: int              # uint32
    provider_id_hash: bytes       # bytes32 — keccak256(utf8(provider_id))
    provider_pubkey_hash: bytes   # bytes32 — keccak256(b64decode(pubkey))
    output_hash: bytes            # bytes32 — hex-decoded sha256
    executed_at_unix: int         # uint64
    value_ftns: int               # uint128
    signature_hash: bytes         # bytes32 — keccak256(b64decode(signature))

    def __post_init__(self):
        _check_bytes32("job_id_hash", self.job_id_hash)
        _check_uint("shard_index", self.shard_index, _MAX_UINT32)
        _check_bytes32("provider_id_hash", self.provider_id_hash)
        _check_bytes32("provider_pubkey_hash", self.provider_pubkey_hash)
        _check_bytes32("output_hash", self.output_hash)
        _check_uint("executed_at_unix", self.executed_at_unix, _MAX_UINT64)
        _check_uint("value_ftns", self.value_ftns, _MAX_UINT128)
        _check_bytes32("signature_hash", self.signature_hash)


def _check_bytes32(name: str, value: bytes) -> None:
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{name}: expected bytes, got {type(value).__name__}")
    if len(value) != 32:
        raise ValueError(f"{name}: expected 32 bytes, got {len(value)}")


def _check_uint(name: str, value: int, max_value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name}: expected int, got {type(value).__name__}")
    if value < 0 or value > max_value:
        raise ValueError(
            f"{name}: value {value} out of range [0, {max_value}]"
        )


def batched_receipt_to_leaf(br: BatchedReceipt) -> ReceiptLeaf:
    """Convert a BatchedReceipt into the canonical on-chain ReceiptLeaf.

    The Phase 2 ShardExecutionReceipt's string + base64 fields are
    hashed into their canonical bytes32 forms here. Conversions:

        job_id (str utf8) → keccak256(utf8_bytes)
        shard_index (int) → uint32 (bounds-checked)
        provider_id (str utf8) → keccak256(utf8_bytes)
        provider_pubkey_b64 (b64 str) → keccak256(decoded bytes)
        output_hash (hex str, 64 chars) → raw 32 bytes
        executed_at_unix (int) → uint64 (bounds-checked)
        signature (b64 str) → keccak256(decoded bytes)

    Plus from the BatchedReceipt wrapper:
        value_ftns (int wei) → uint128 (bounds-checked)

    Raises ValueError on any out-of-range integer or wrong-length
    output hash; raises binascii.Error on malformed base64.
    """
    r = br.receipt

    # output_hash is a hex sha256 digest per Phase 2.
    try:
        output_hash_bytes = bytes.fromhex(r.output_hash)
    except ValueError as exc:
        raise ValueError(
            f"output_hash is not valid hex: {r.output_hash!r}"
        ) from exc
    if len(output_hash_bytes) != 32:
        raise ValueError(
            f"output_hash must be 32 bytes (64 hex chars), got "
            f"{len(output_hash_bytes)} bytes from {r.output_hash!r}"
        )

    return ReceiptLeaf(
        job_id_hash=keccak(r.job_id.encode("utf-8")),
        shard_index=r.shard_index,
        provider_id_hash=keccak(r.provider_id.encode("utf-8")),
        provider_pubkey_hash=keccak(b64decode(r.provider_pubkey_b64)),
        output_hash=output_hash_bytes,
        executed_at_unix=r.executed_at_unix,
        value_ftns=br.value_ftns,
        signature_hash=keccak(b64decode(r.signature)),
    )


def encode_leaf(leaf: ReceiptLeaf) -> bytes:
    """ABI-encode the ReceiptLeaf tuple. Output matches Solidity's
    `abi.encode(ReceiptLeaf leaf)` byte-for-byte for static-only
    field types (which this struct is)."""
    return abi_encode(
        [_LEAF_SIGNATURE],
        [(
            leaf.job_id_hash,
            leaf.shard_index,
            leaf.provider_id_hash,
            leaf.provider_pubkey_hash,
            leaf.output_hash,
            leaf.executed_at_unix,
            leaf.value_ftns,
            leaf.signature_hash,
        )],
    )


def hash_leaf(leaf: ReceiptLeaf) -> bytes:
    """keccak256 of the ABI-encoded leaf. Matches the Solidity
    `_hashLeaf` helper's output exactly."""
    return keccak(encode_leaf(leaf))


# ── Merkle tree ───────────────────────────────────────────────────


def _hash_pair(a: bytes, b: bytes) -> bytes:
    """OpenZeppelin MerkleProof._hashPair: sort before hashing.

    Sorting lets verifiers avoid tracking left/right sibling positions
    in proofs — the fold order is recovered from comparing the running
    hash to the incoming sibling."""
    if a < b:
        return keccak(a + b)
    return keccak(b + a)


def build_merkle_root(leaf_hashes: List[bytes]) -> bytes:
    """Build the Merkle root over a list of already-hashed leaves.

    Single-leaf case: root = that leaf's hash.
    Odd-count layers: the last node is promoted unchanged to the next
    layer (no duplication).

    Raises ValueError on empty input.
    """
    if not leaf_hashes:
        raise ValueError("build_merkle_root: empty leaf list")
    layer = [bytes(h) for h in leaf_hashes]  # defensive copy
    while len(layer) > 1:
        next_layer: List[bytes] = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                next_layer.append(_hash_pair(layer[i], layer[i + 1]))
            else:
                next_layer.append(layer[i])  # odd promotion
        layer = next_layer
    return layer[0]


def build_merkle_proof(
    leaf_hashes: List[bytes], leaf_index: int,
) -> List[bytes]:
    """Build a Merkle proof for the leaf at `leaf_index`.

    Returns the list of sibling hashes to fold in during verify.
    Raises IndexError for out-of-range index or empty input.
    """
    if not leaf_hashes:
        raise IndexError("build_merkle_proof: empty leaf list")
    if leaf_index < 0 or leaf_index >= len(leaf_hashes):
        raise IndexError(
            f"build_merkle_proof: leaf_index {leaf_index} out of range "
            f"[0, {len(leaf_hashes)})"
        )

    proof: List[bytes] = []
    layer = [bytes(h) for h in leaf_hashes]
    idx = leaf_index
    while len(layer) > 1:
        # Record this layer's sibling for idx.
        if idx % 2 == 0:
            # Left child; sibling is at idx+1 if it exists.
            if idx + 1 < len(layer):
                proof.append(layer[idx + 1])
            # else: odd promotion — no sibling at this layer
        else:
            # Right child; sibling is always at idx-1.
            proof.append(layer[idx - 1])

        # Build next layer.
        next_layer: List[bytes] = []
        for i in range(0, len(layer), 2):
            if i + 1 < len(layer):
                next_layer.append(_hash_pair(layer[i], layer[i + 1]))
            else:
                next_layer.append(layer[i])
        layer = next_layer
        idx //= 2

    return proof


def verify_merkle_proof(
    proof: List[bytes],
    root: bytes,
    leaf_hash: bytes,
) -> bool:
    """Verify a Merkle proof using OpenZeppelin's sorted-pair convention.

    Matches Solidity MerkleProof.verify byte-for-byte: fold each proof
    element into the running hash via sorted concatenation, then check
    against the root. Empty proof succeeds iff root == leaf_hash (the
    single-leaf tree case)."""
    computed = bytes(leaf_hash)
    for sibling in proof:
        computed = _hash_pair(computed, sibling)
    return computed == root


def build_tree_and_proofs(
    leaf_hashes: List[bytes],
) -> "MerkleTree":
    """Convenience: build the root + proofs for every leaf in one pass.

    Returns a MerkleTree object with .root, .leaf_hashes, and .proof(i)
    method. Useful for batch-commit paths where the client builds the
    root once and references proofs later (e.g., for inclusion in
    challenge auxData).
    """
    return MerkleTree(
        root=build_merkle_root(leaf_hashes),
        leaf_hashes=list(leaf_hashes),
        _proofs={
            i: build_merkle_proof(leaf_hashes, i)
            for i in range(len(leaf_hashes))
        },
    )


@dataclass(frozen=True)
class MerkleTree:
    """Opaque tree container. Use build_tree_and_proofs to construct."""
    root: bytes
    leaf_hashes: List[bytes]
    _proofs: dict  # { index: List[bytes] }

    def proof(self, leaf_index: int) -> List[bytes]:
        """Pre-computed proof for the leaf at `leaf_index`."""
        if leaf_index not in self._proofs:
            raise IndexError(
                f"MerkleTree.proof: leaf_index {leaf_index} not in tree "
                f"(size={len(self.leaf_hashes)})"
            )
        return self._proofs[leaf_index]

    def verify(self, leaf_index: int, leaf_hash: bytes) -> bool:
        """Self-check a proof against the tree's root."""
        return verify_merkle_proof(
            self.proof(leaf_index), self.root, leaf_hash
        )
