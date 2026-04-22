"""Unit tests for prsm.storage.proof.

Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 4.
"""

from __future__ import annotations

import hashlib

import pytest

from prsm.storage.proof import (
    ChallengeIssuer,
    MerkleTree,
    ProofChallenge,
    ProofResponse,
    ProofResult,
    ProofVerdict,
    ProofVerifier,
    verify_merkle_proof,
)


# -----------------------------------------------------------------------------
# MerkleTree + verify_merkle_proof
# -----------------------------------------------------------------------------


def test_merkle_tree_single_chunk_root_is_chunk_hash():
    """Single-chunk tree has no padding (target=1 is already a power of 2)
    and no internal nodes — root IS the leaf digest."""
    tree = MerkleTree([b"hello"])
    assert tree.root() == hashlib.sha256(b"hello").digest()
    assert tree.leaf_count == 1
    assert tree.proof(0) == []


def test_merkle_tree_two_chunks_root_combines_leaves():
    """Two-chunk tree: root = hash(leaf0 || leaf1) with no padding."""
    tree = MerkleTree([b"a", b"b"])
    leaf_a = hashlib.sha256(b"a").digest()
    leaf_b = hashlib.sha256(b"b").digest()
    assert tree.root() == hashlib.sha256(leaf_a + leaf_b).digest()


def test_merkle_tree_leaf_count_rounds_up_to_power_of_2():
    tree = MerkleTree([b"a", b"b", b"c"])  # 3 chunks → padded to 4
    assert tree.leaf_count == 4


def test_merkle_tree_proof_verifies_roundtrip():
    chunks = [f"chunk-{i}".encode() for i in range(8)]
    tree = MerkleTree(chunks)
    for i in range(8):
        proof = tree.proof(i)
        assert verify_merkle_proof(chunks[i], proof, i, tree.root())


def test_merkle_tree_proof_for_padded_leaf_index_rejects():
    chunks = [b"a", b"b", b"c"]  # 3 real, 1 padding
    tree = MerkleTree(chunks)
    # Caller asks for a proof of real leaf 0.
    proof = tree.proof(0)
    assert verify_merkle_proof(chunks[0], proof, 0, tree.root())
    # Claiming chunk[0]'s bytes live at index 1 (where chunk[1] is) must fail.
    assert not verify_merkle_proof(chunks[0], proof, 1, tree.root())


def test_verify_merkle_proof_rejects_wrong_leaf():
    chunks = [b"a", b"b", b"c", b"d"]
    tree = MerkleTree(chunks)
    proof = tree.proof(2)
    assert not verify_merkle_proof(b"not-c", proof, 2, tree.root())


def test_verify_merkle_proof_rejects_wrong_root():
    chunks = [b"a", b"b", b"c", b"d"]
    tree = MerkleTree(chunks)
    proof = tree.proof(2)
    wrong_root = b"\xff" * 32
    assert not verify_merkle_proof(chunks[2], proof, 2, wrong_root)


def test_merkle_tree_rejects_empty_input():
    with pytest.raises(ValueError):
        MerkleTree([])


def test_merkle_tree_proof_index_out_of_range():
    tree = MerkleTree([b"a", b"b"])
    with pytest.raises(IndexError):
        tree.proof(99)


# -----------------------------------------------------------------------------
# ChallengeIssuer
# -----------------------------------------------------------------------------


def test_issuer_produces_challenge_for_in_range_chunk():
    issuer = ChallengeIssuer(
        clock=lambda: 1_700_000_000.0,
        rng=lambda n: b"\x00" * n,  # deterministic rng
    )
    c = issuer.issue("provider-1", "shard-hash-abc", num_chunks=10)
    assert 0 <= c.chunk_index < 10
    assert c.shard_id == "shard-hash-abc"
    assert c.provider_id == "provider-1"
    assert c.issued_at_unix == 1_700_000_000
    assert c.deadline_unix > c.issued_at_unix
    assert len(c.nonce) == 16


def test_issuer_rejects_zero_num_chunks():
    issuer = ChallengeIssuer()
    with pytest.raises(ValueError):
        issuer.issue("p", "s", num_chunks=0)


def test_issuer_generates_unique_challenge_ids():
    issuer = ChallengeIssuer()
    ids = {issuer.issue("p", "s", num_chunks=10).challenge_id for _ in range(20)}
    assert len(ids) == 20


def test_issuer_nonces_are_unique_across_calls():
    issuer = ChallengeIssuer()
    nonces = {issuer.issue("p", "s", num_chunks=4).nonce for _ in range(20)}
    assert len(nonces) == 20


# -----------------------------------------------------------------------------
# ProofVerifier — happy path
# -----------------------------------------------------------------------------


def _build_scenario(num_chunks: int = 8, chunk_size: int = 32):
    chunks = [(f"chunk-{i}-").ljust(chunk_size, "x").encode() for i in range(num_chunks)]
    tree = MerkleTree(chunks)
    return chunks, tree


def test_verifier_accepts_valid_proof():
    chunks, tree = _build_scenario()
    clock = [1_700_000_000.0]

    issuer = ChallengeIssuer(clock=lambda: clock[0])
    challenge = issuer.issue("provider-1", "shard-1", num_chunks=len(chunks))

    response = ProofResponse(
        challenge_id=challenge.challenge_id,
        chunk_data=chunks[challenge.chunk_index],
        merkle_proof=tree.proof(challenge.chunk_index),
    )

    verifier = ProofVerifier(
        challenger_id="challenger-1",
        clock=lambda: clock[0] + 5,
    )
    result = verifier.verify(challenge, response, tree.root())
    assert result.verified
    assert result.verdict is ProofVerdict.OK


# -----------------------------------------------------------------------------
# ProofVerifier — failure paths
# -----------------------------------------------------------------------------


class _StubSlashHook:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_proof_failure(
        self, provider_id, shard_id, evidence_hash, challenger
    ):
        self.calls.append(
            dict(
                provider_id=provider_id,
                shard_id=shard_id,
                evidence_hash=evidence_hash,
                challenger=challenger,
            )
        )


def test_verifier_rejects_tampered_chunk_and_escalates():
    chunks, tree = _build_scenario()
    clock = [1_700_000_000.0]

    issuer = ChallengeIssuer(clock=lambda: clock[0])
    challenge = issuer.issue("provider-X", "shard-Y", num_chunks=len(chunks))

    # Tampered chunk bytes.
    tampered = bytes(len(chunks[challenge.chunk_index]))  # all zeros
    response = ProofResponse(
        challenge_id=challenge.challenge_id,
        chunk_data=tampered,
        merkle_proof=tree.proof(challenge.chunk_index),
    )

    hook = _StubSlashHook()
    verifier = ProofVerifier(
        challenger_id="c1", clock=lambda: clock[0] + 5, slash_hook=hook
    )
    result = verifier.verify(challenge, response, tree.root())

    assert result.verdict is ProofVerdict.MERKLE_MISMATCH
    assert not result.verified
    # On-chain escalation fired exactly once with correct args.
    assert len(hook.calls) == 1
    assert hook.calls[0]["provider_id"] == "provider-X"
    assert hook.calls[0]["shard_id"] == "shard-Y"
    assert hook.calls[0]["challenger"] == "c1"
    assert hook.calls[0]["evidence_hash"].startswith("0x")


def test_verifier_rejects_response_with_mismatched_challenge_id():
    chunks, tree = _build_scenario()
    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("p", "s", num_chunks=len(chunks))

    response = ProofResponse(
        challenge_id="wrong-id",
        chunk_data=chunks[challenge.chunk_index],
        merkle_proof=tree.proof(challenge.chunk_index),
    )

    verifier = ProofVerifier(
        challenger_id="c", clock=lambda: 1_700_000_000.0 + 5
    )
    result = verifier.verify(challenge, response, tree.root())
    assert result.verdict is ProofVerdict.CHALLENGE_ID_MISMATCH


def test_verifier_rejects_missing_response_past_deadline():
    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("p", "s", num_chunks=4, deadline_seconds=30)

    # Clock now past deadline.
    verifier = ProofVerifier(
        challenger_id="c", clock=lambda: 1_700_000_000.0 + 60
    )
    result = verifier.verify(challenge, None, b"\x00" * 32)
    assert result.verdict is ProofVerdict.DEADLINE_EXCEEDED


def test_verifier_rejects_missing_response_before_deadline_as_pending():
    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("p", "s", num_chunks=4, deadline_seconds=30)

    verifier = ProofVerifier(
        challenger_id="c", clock=lambda: 1_700_000_000.0 + 5
    )
    result = verifier.verify(challenge, None, b"\x00" * 32)
    assert result.verdict is ProofVerdict.MISSING_RESPONSE


def test_verifier_rejects_late_response_as_deadline_exceeded():
    chunks, tree = _build_scenario()
    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("p", "s", num_chunks=len(chunks), deadline_seconds=30)
    response = ProofResponse(
        challenge_id=challenge.challenge_id,
        chunk_data=chunks[challenge.chunk_index],
        merkle_proof=tree.proof(challenge.chunk_index),
    )
    verifier = ProofVerifier(
        challenger_id="c", clock=lambda: 1_700_000_000.0 + 100
    )
    result = verifier.verify(challenge, response, tree.root())
    assert result.verdict is ProofVerdict.DEADLINE_EXCEEDED


def test_verifier_does_not_escalate_on_success():
    chunks, tree = _build_scenario()
    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("p", "s", num_chunks=len(chunks))
    response = ProofResponse(
        challenge_id=challenge.challenge_id,
        chunk_data=chunks[challenge.chunk_index],
        merkle_proof=tree.proof(challenge.chunk_index),
    )
    hook = _StubSlashHook()
    verifier = ProofVerifier(
        challenger_id="c",
        clock=lambda: 1_700_000_000.0 + 5,
        slash_hook=hook,
    )
    result = verifier.verify(challenge, response, tree.root())
    assert result.verified
    assert hook.calls == []


def test_verifier_no_hook_does_not_raise_on_failure():
    """No slash hook = silent failure mode (for local / dev flows)."""
    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("p", "s", num_chunks=4, deadline_seconds=30)

    verifier = ProofVerifier(
        challenger_id="c",
        clock=lambda: 1_700_000_000.0 + 100,
        slash_hook=None,
    )
    # Past deadline, missing response.
    result = verifier.verify(challenge, None, b"\x00" * 32)
    assert not result.verified
    # No exception, no hook — just returns a failed ProofResult.
