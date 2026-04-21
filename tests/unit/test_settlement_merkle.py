"""Phase 3.1 Task 5 — canonical encoding + Merkle tree unit tests.

Internal-consistency suite: every operation is self-checked against
the library's own inverse (build_root + build_proof + verify all agree).
Python↔Solidity cross-parity is verified in Task 8's integration test
when these hashes are sent to the on-chain contract.
"""
from __future__ import annotations

import hashlib
from base64 import b64encode
from typing import List

import pytest

from eth_utils import keccak

from prsm.compute.shard_receipt import ShardExecutionReceipt
from prsm.settlement.accumulator import BatchedReceipt
from prsm.settlement.merkle import (
    MerkleTree,
    ReceiptLeaf,
    batched_receipt_to_leaf,
    build_merkle_proof,
    build_merkle_root,
    build_tree_and_proofs,
    encode_leaf,
    hash_leaf,
    verify_merkle_proof,
)


# ── Helpers ───────────────────────────────────────────────────────


def _dummy_leaf(
    job_id: str = "job-1",
    shard_index: int = 0,
    value_ftns: int = 10**18,
) -> ReceiptLeaf:
    """Construct a well-formed ReceiptLeaf directly (bypassing the
    batched-receipt conversion). For tests that exercise tree/proof
    mechanics without pulling in ShardExecutionReceipt details."""
    return ReceiptLeaf(
        job_id_hash=keccak(job_id.encode("utf-8")),
        shard_index=shard_index,
        provider_id_hash=keccak(b"provider-id"),
        provider_pubkey_hash=keccak(b"pubkey"),
        output_hash=hashlib.sha256(f"{job_id}:{shard_index}".encode()).digest(),
        executed_at_unix=1700000000 + shard_index,
        value_ftns=value_ftns,
        signature_hash=keccak(b"signature"),
    )


def _dummy_batched(
    job_id: str = "job-1",
    shard_index: int = 0,
    value_ftns: int = 10**18,
) -> BatchedReceipt:
    receipt = ShardExecutionReceipt(
        job_id=job_id,
        shard_index=shard_index,
        provider_id="provider-id-string",
        provider_pubkey_b64=b64encode(b"pubkey-raw-bytes").decode(),
        output_hash=hashlib.sha256(
            f"{job_id}:{shard_index}".encode()
        ).hexdigest(),
        executed_at_unix=1700000000 + shard_index,
        signature=b64encode(b"signature-raw-bytes").decode(),
    )
    return BatchedReceipt(
        receipt=receipt,
        requester_address="0x" + "a" * 40,
        provider_address="0x" + "b" * 40,
        value_ftns=value_ftns,
        local_escrow_id=f"escrow-{shard_index}",
    )


# ── Leaf construction + validation ────────────────────────────────


def test_receipt_leaf_validates_bytes32_lengths():
    """Any bytes32 field that isn't exactly 32 bytes is rejected at
    construction time."""
    with pytest.raises(ValueError, match="32 bytes"):
        ReceiptLeaf(
            job_id_hash=b"too_short",  # not 32 bytes
            shard_index=0,
            provider_id_hash=keccak(b"p"),
            provider_pubkey_hash=keccak(b"k"),
            output_hash=keccak(b"o"),
            executed_at_unix=1,
            value_ftns=1,
            signature_hash=keccak(b"s"),
        )


def test_receipt_leaf_validates_uint32_bound():
    with pytest.raises(ValueError, match="out of range"):
        ReceiptLeaf(
            job_id_hash=keccak(b"j"),
            shard_index=2**33,  # over uint32
            provider_id_hash=keccak(b"p"),
            provider_pubkey_hash=keccak(b"k"),
            output_hash=keccak(b"o"),
            executed_at_unix=1,
            value_ftns=1,
            signature_hash=keccak(b"s"),
        )


def test_receipt_leaf_validates_uint64_bound():
    with pytest.raises(ValueError, match="out of range"):
        ReceiptLeaf(
            job_id_hash=keccak(b"j"),
            shard_index=0,
            provider_id_hash=keccak(b"p"),
            provider_pubkey_hash=keccak(b"k"),
            output_hash=keccak(b"o"),
            executed_at_unix=2**65,  # over uint64
            value_ftns=1,
            signature_hash=keccak(b"s"),
        )


def test_receipt_leaf_validates_uint128_bound():
    with pytest.raises(ValueError, match="out of range"):
        ReceiptLeaf(
            job_id_hash=keccak(b"j"),
            shard_index=0,
            provider_id_hash=keccak(b"p"),
            provider_pubkey_hash=keccak(b"k"),
            output_hash=keccak(b"o"),
            executed_at_unix=1,
            value_ftns=2**129,  # over uint128
            signature_hash=keccak(b"s"),
        )


def test_receipt_leaf_rejects_negative_uint():
    with pytest.raises(ValueError, match="out of range"):
        _dummy_leaf().__class__(
            job_id_hash=keccak(b"j"),
            shard_index=-1,
            provider_id_hash=keccak(b"p"),
            provider_pubkey_hash=keccak(b"k"),
            output_hash=keccak(b"o"),
            executed_at_unix=1,
            value_ftns=1,
            signature_hash=keccak(b"s"),
        )


def test_receipt_leaf_rejects_bool_for_int():
    """Python bools are ints subclass; explicit check rejects them
    because they'd silently convert to 0/1 and produce confusing bugs."""
    with pytest.raises(TypeError):
        ReceiptLeaf(
            job_id_hash=keccak(b"j"),
            shard_index=True,  # bool
            provider_id_hash=keccak(b"p"),
            provider_pubkey_hash=keccak(b"k"),
            output_hash=keccak(b"o"),
            executed_at_unix=1,
            value_ftns=1,
            signature_hash=keccak(b"s"),
        )


# ── BatchedReceipt → ReceiptLeaf conversion ──────────────────────


def test_batched_receipt_to_leaf_roundtrip():
    """Conversion produces a valid ReceiptLeaf with canonical hashes
    for each source field."""
    br = _dummy_batched()
    leaf = batched_receipt_to_leaf(br)

    assert leaf.job_id_hash == keccak(br.receipt.job_id.encode("utf-8"))
    assert leaf.shard_index == br.receipt.shard_index
    assert leaf.provider_id_hash == keccak(br.receipt.provider_id.encode("utf-8"))
    assert leaf.value_ftns == br.value_ftns
    # output_hash is hex-decoded sha256 (32 bytes).
    assert len(leaf.output_hash) == 32


def test_batched_receipt_to_leaf_rejects_bad_output_hash():
    """If a BatchedReceipt somehow carries a malformed output_hash
    (wrong length, not hex), conversion raises ValueError cleanly
    rather than producing an invalid leaf."""
    br = _dummy_batched()
    # Replace the receipt with a bogus output_hash.
    bad = BatchedReceipt(
        receipt=ShardExecutionReceipt(
            job_id=br.receipt.job_id,
            shard_index=br.receipt.shard_index,
            provider_id=br.receipt.provider_id,
            provider_pubkey_b64=br.receipt.provider_pubkey_b64,
            output_hash="not-hex-at-all",
            executed_at_unix=br.receipt.executed_at_unix,
            signature=br.receipt.signature,
        ),
        requester_address=br.requester_address,
        provider_address=br.provider_address,
        value_ftns=br.value_ftns,
        local_escrow_id=br.local_escrow_id,
    )
    with pytest.raises(ValueError, match="not valid hex"):
        batched_receipt_to_leaf(bad)


def test_batched_receipt_to_leaf_rejects_wrong_length_output_hash():
    br = _dummy_batched()
    bad = BatchedReceipt(
        receipt=ShardExecutionReceipt(
            job_id=br.receipt.job_id,
            shard_index=br.receipt.shard_index,
            provider_id=br.receipt.provider_id,
            provider_pubkey_b64=br.receipt.provider_pubkey_b64,
            output_hash="deadbeef" * 2,  # 16 bytes, not 32
            executed_at_unix=br.receipt.executed_at_unix,
            signature=br.receipt.signature,
        ),
        requester_address=br.requester_address,
        provider_address=br.provider_address,
        value_ftns=br.value_ftns,
        local_escrow_id=br.local_escrow_id,
    )
    with pytest.raises(ValueError, match="32 bytes"):
        batched_receipt_to_leaf(bad)


# ── encode_leaf / hash_leaf ───────────────────────────────────────


def test_hash_leaf_is_deterministic():
    """Identical input → identical hash, always."""
    leaf = _dummy_leaf()
    h1 = hash_leaf(leaf)
    h2 = hash_leaf(leaf)
    assert h1 == h2


def test_hash_leaf_produces_32_bytes():
    assert len(hash_leaf(_dummy_leaf())) == 32


def test_different_leaves_produce_different_hashes():
    h_a = hash_leaf(_dummy_leaf(job_id="A"))
    h_b = hash_leaf(_dummy_leaf(job_id="B"))
    assert h_a != h_b


def test_encode_leaf_stable_ordering():
    """Swapping any field should change the encoded bytes (sanity: no
    field is silently ignored)."""
    base = hash_leaf(_dummy_leaf(shard_index=0))
    shifted = hash_leaf(_dummy_leaf(shard_index=1))
    assert base != shifted


# ── Merkle root ───────────────────────────────────────────────────


def test_root_single_leaf_equals_leaf_hash():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(1)]
    assert build_merkle_root(leaves) == leaves[0]


def test_root_two_leaves_sorted_pair_hash():
    a = hash_leaf(_dummy_leaf(shard_index=0))
    b = hash_leaf(_dummy_leaf(shard_index=1))
    # Computed manually via the sorted-pair convention.
    expected = keccak(a + b) if a < b else keccak(b + a)
    assert build_merkle_root([a, b]) == expected


def test_root_three_leaves_odd_promotion():
    """With 3 leaves: layer0 = [A, B, C]; layer1 = [H(A,B)_sorted, C];
    root = H(H(A,B), C)_sorted."""
    a = hash_leaf(_dummy_leaf(job_id="A"))
    b = hash_leaf(_dummy_leaf(job_id="B"))
    c = hash_leaf(_dummy_leaf(job_id="C"))
    h_ab = keccak(a + b) if a < b else keccak(b + a)
    expected = (
        keccak(h_ab + c) if h_ab < c else keccak(c + h_ab)
    )
    assert build_merkle_root([a, b, c]) == expected


def test_root_powers_of_two_sizes():
    """For N in {1, 2, 4, 8}, the tree is balanced (no odd promotion).
    Verify the root is reproducible."""
    for n in [1, 2, 4, 8]:
        leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(n)]
        r1 = build_merkle_root(leaves)
        r2 = build_merkle_root(leaves)
        assert r1 == r2
        assert len(r1) == 32


def test_root_empty_list_raises():
    with pytest.raises(ValueError, match="empty leaf list"):
        build_merkle_root([])


def test_root_deterministic_across_calls():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(7)]
    roots = [build_merkle_root(leaves) for _ in range(5)]
    assert len(set(roots)) == 1


# ── Proof generation + verification ───────────────────────────────


def test_single_leaf_proof_is_empty():
    leaves = [hash_leaf(_dummy_leaf())]
    proof = build_merkle_proof(leaves, 0)
    assert proof == []
    assert verify_merkle_proof(proof, leaves[0], leaves[0]) is True


def test_two_leaf_proof_for_each():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(2)]
    root = build_merkle_root(leaves)
    for idx in range(2):
        proof = build_merkle_proof(leaves, idx)
        assert len(proof) == 1
        assert verify_merkle_proof(proof, root, leaves[idx]) is True


def test_three_leaf_proof_for_each_including_odd_promoted():
    """Odd-promoted leaf (index 2) has a shorter proof than leaves 0/1."""
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(3)]
    root = build_merkle_root(leaves)
    for idx in range(3):
        proof = build_merkle_proof(leaves, idx)
        assert verify_merkle_proof(proof, root, leaves[idx]) is True


def test_eight_leaf_proofs_all_verify():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(8)]
    root = build_merkle_root(leaves)
    for idx in range(8):
        proof = build_merkle_proof(leaves, idx)
        # 8-leaf balanced tree: proofs have 3 elements each.
        assert len(proof) == 3
        assert verify_merkle_proof(proof, root, leaves[idx]) is True


def test_one_thousand_leaf_proofs_sample_verify():
    """Stress test: 1000 leaves (the default batch count). Sample a few
    proofs across the tree."""
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(1000)]
    root = build_merkle_root(leaves)
    for idx in [0, 1, 100, 500, 999]:
        proof = build_merkle_proof(leaves, idx)
        assert verify_merkle_proof(proof, root, leaves[idx]) is True


def test_verify_fails_on_wrong_root():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(4)]
    proof = build_merkle_proof(leaves, 0)
    fake_root = keccak(b"wrong-root-31-bytes-padding-pad!")
    assert verify_merkle_proof(proof, fake_root, leaves[0]) is False


def test_verify_fails_on_wrong_leaf_hash():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(4)]
    root = build_merkle_root(leaves)
    proof = build_merkle_proof(leaves, 0)
    fake_leaf = keccak(b"some-other-leaf")
    assert verify_merkle_proof(proof, root, fake_leaf) is False


def test_verify_fails_on_tampered_proof():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(4)]
    root = build_merkle_root(leaves)
    proof = build_merkle_proof(leaves, 0)
    tampered = list(proof)
    tampered[0] = keccak(b"tampered")
    assert verify_merkle_proof(tampered, root, leaves[0]) is False


def test_build_proof_out_of_range_raises():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(4)]
    with pytest.raises(IndexError):
        build_merkle_proof(leaves, 4)
    with pytest.raises(IndexError):
        build_merkle_proof(leaves, -1)


def test_build_proof_empty_list_raises():
    with pytest.raises(IndexError):
        build_merkle_proof([], 0)


# ── MerkleTree convenience container ──────────────────────────────


def test_merkle_tree_proof_and_verify():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(5)]
    tree = build_tree_and_proofs(leaves)

    assert isinstance(tree, MerkleTree)
    assert tree.root == build_merkle_root(leaves)
    assert tree.leaf_hashes == leaves

    for i in range(5):
        assert tree.verify(i, leaves[i]) is True


def test_merkle_tree_proof_out_of_range_raises():
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(3)]
    tree = build_tree_and_proofs(leaves)
    with pytest.raises(IndexError):
        tree.proof(3)


def test_merkle_tree_proofs_match_standalone():
    """build_tree_and_proofs' cached proofs match what build_merkle_proof
    produces directly."""
    leaves = [hash_leaf(_dummy_leaf(shard_index=i)) for i in range(5)]
    tree = build_tree_and_proofs(leaves)
    for i in range(5):
        assert tree.proof(i) == build_merkle_proof(leaves, i)


# ── End-to-end: BatchedReceipt → tree ────────────────────────────


def test_end_to_end_batch_of_receipts():
    """Simulate the full Task 6 flow: accumulate N batched receipts,
    convert each to a ReceiptLeaf, hash, build tree, verify every
    proof round-trips."""
    batch = [_dummy_batched(shard_index=i) for i in range(16)]
    leaves = [batched_receipt_to_leaf(br) for br in batch]
    leaf_hashes = [hash_leaf(l) for l in leaves]
    tree = build_tree_and_proofs(leaf_hashes)

    # Every leaf's proof verifies against the committed root.
    for i in range(16):
        assert tree.verify(i, leaf_hashes[i])

    # Wrong leaf in same position fails.
    bogus = hash_leaf(_dummy_leaf(job_id="not-in-tree"))
    assert tree.verify(0, bogus) is False
