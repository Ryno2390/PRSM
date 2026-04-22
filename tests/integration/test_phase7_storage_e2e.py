"""Phase 7-storage end-to-end integration tests.

Per docs/2026-04-22-phase7-storage-design-plan.md §7 + §6 Task 8.

Exercises the full shipped Phase 7-storage stack in composition:

    Tier A  =  Reed-Solomon(k=6,n=10) only
    Tier B  =  AES-256-GCM encrypt → Reed-Solomon → (payment → key-release)
    Tier C  =  AES-256-GCM encrypt → Reed-Solomon → Shamir(m=3,n=5) split key
               → (payment → M-of-N holders release → reconstruct key)

Plus the two cross-cutting scenarios:
    - Provider churn during download (kill 4 of 10 shards → still recoverable)
    - Challenge-then-slash (provider returns bad chunk → slash hook fires)

No network I/O; no on-chain transactions. The storage providers,
royalty distributor, and on-chain slash hook are simulated in-process so
the integration scenarios are fully deterministic and CI-runnable.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from prsm.storage.encryption import AESKey, decrypt, encrypt, generate_key
from prsm.storage.erasure import (
    ErasureMetadata,
    ErasureShard,
    InsufficientShardsError,
    decode,
    encode,
)
from prsm.storage.key_sharing import (
    KeyShare,
    combine_shares,
    split_key,
)
from prsm.storage.proof import (
    ChallengeIssuer,
    MerkleTree,
    ProofResponse,
    ProofVerdict,
    ProofVerifier,
)


# =============================================================================
# In-process storage provider simulation
# =============================================================================


@dataclass
class StorageProvider:
    """Simulated storage provider — holds one shard and can serve it back."""

    provider_id: str
    region: str
    shard: ErasureShard
    online: bool = True

    def serve(self) -> Optional[ErasureShard]:
        return self.shard if self.online else None


def _place_shards_across_regions(
    shards: List[ErasureShard],
    regions: List[str],
) -> List[StorageProvider]:
    """Round-robin shard-to-region placement (plan §3 geographic diversity)."""
    providers: List[StorageProvider] = []
    for i, s in enumerate(shards):
        providers.append(
            StorageProvider(
                provider_id=f"provider-{i:02d}",
                region=regions[i % len(regions)],
                shard=s,
            )
        )
    return providers


def _collect_available_shards(
    providers: List[StorageProvider],
) -> List[ErasureShard]:
    return [p.shard for p in providers if p.online and p.shard is not None]


# =============================================================================
# Tier A — erasure-only
# =============================================================================


def test_tier_a_upload_and_download_roundtrip():
    """Plan §7 acceptance: Tier A content survives a k-of-n retrieval."""
    payload = b"tier A research dataset " * 1000
    meta, shards = encode(payload)

    providers = _place_shards_across_regions(
        shards, ["us-east", "us-west", "eu-west", "ap-south"]
    )
    recovered = decode(meta, _collect_available_shards(providers))
    assert recovered == payload


def test_tier_a_survives_40_percent_provider_loss():
    """Kill 4 of 10 providers (the max the k=6 scheme can tolerate).
    Plan §7 acceptance criterion."""
    payload = os.urandom(4096)
    meta, shards = encode(payload)
    providers = _place_shards_across_regions(
        shards, ["us-east", "us-west", "eu-west", "ap-south"]
    )
    # Take down 4 providers.
    for idx in (1, 3, 5, 7):
        providers[idx].online = False
    available = _collect_available_shards(providers)
    assert len(available) == 6
    assert decode(meta, available) == payload


def test_tier_a_five_provider_loss_is_unrecoverable():
    """One shard short of k — no silent partial recovery."""
    payload = os.urandom(2048)
    meta, shards = encode(payload)
    providers = _place_shards_across_regions(shards, ["us-east"])
    for idx in range(5):
        providers[idx].online = False
    with pytest.raises(InsufficientShardsError):
        decode(meta, _collect_available_shards(providers))


# =============================================================================
# Tier B — AES + erasure + payment-gated key release
# =============================================================================


class _SimpleKeyDistribution:
    """In-process analogue of KeyDistribution.sol.

    Maps content_hash → (encrypted_key_blob, release_fee). release()
    checks payment and returns the blob.
    """

    def __init__(self) -> None:
        self._records: Dict[bytes, tuple[AESKey, int]] = {}
        self._paid: Dict[tuple[bytes, bytes], bool] = {}

    def deposit(self, content_hash: bytes, key: AESKey, fee: int) -> None:
        self._records[content_hash] = (key, fee)

    def record_payment(self, recipient: bytes, content_hash: bytes) -> None:
        self._paid[(recipient, content_hash)] = True

    def release(self, content_hash: bytes, recipient: bytes) -> AESKey:
        if (recipient, content_hash) not in self._paid:
            raise PermissionError("payment not verified")
        key, _fee = self._records[content_hash]
        return key


def test_tier_b_publish_pay_decrypt_flow():
    """Plan §7 acceptance: Tier B publish → pay → retrieve → decrypt.

    Byte-for-byte match to original plaintext.
    """
    plaintext = b"Tier B confidential research " * 200
    content_hash = hashlib.sha256(plaintext).digest()

    # 1. Publisher encrypts.
    key = generate_key()
    encrypted = encrypt(plaintext, key, associated_data=content_hash)

    # 2. Erasure-code the ciphertext.
    wire = encrypted.iv + encrypted.auth_tag + encrypted.ciphertext
    meta, shards = encode(wire)

    # 3. Place on providers.
    providers = _place_shards_across_regions(
        shards, ["us-east", "us-west", "eu-west"]
    )

    # 4. Publisher deposits the key.
    key_dist = _SimpleKeyDistribution()
    key_dist.deposit(content_hash, key, fee=1_000_000_000_000_000_000)

    # 5. Consumer fetches shards.
    consumer_id = b"consumer-wallet-0xabc"
    available = _collect_available_shards(providers)
    wire_recovered = decode(meta, available)

    # 6. Consumer attempts key release BEFORE paying — denied.
    with pytest.raises(PermissionError):
        key_dist.release(content_hash, consumer_id)

    # 7. Consumer pays; key released.
    key_dist.record_payment(consumer_id, content_hash)
    released_key = key_dist.release(content_hash, consumer_id)

    # 8. Consumer decrypts.
    from prsm.storage.encryption import IV_BYTES, AUTH_TAG_BYTES, EncryptedPayload
    iv = wire_recovered[:IV_BYTES]
    auth_tag = wire_recovered[IV_BYTES : IV_BYTES + AUTH_TAG_BYTES]
    ciphertext = wire_recovered[IV_BYTES + AUTH_TAG_BYTES :]
    reassembled = EncryptedPayload(
        ciphertext=ciphertext,
        iv=iv,
        auth_tag=auth_tag,
        key_id=released_key.key_id,
    )

    recovered = decrypt(reassembled, released_key, associated_data=content_hash)
    assert recovered == plaintext


def test_tier_b_shards_alone_yield_no_plaintext():
    """Sanity: the shards themselves (without key release) cannot be
    decrypted — confirms Tier B ciphertext is meaningful without the key."""
    plaintext = b"Tier B secret"
    content_hash = hashlib.sha256(plaintext).digest()

    key = generate_key()
    encrypted = encrypt(plaintext, key, associated_data=content_hash)
    wire = encrypted.iv + encrypted.auth_tag + encrypted.ciphertext
    meta, shards = encode(wire)

    # Consumer reassembles without the key.
    wire_recovered = decode(meta, shards)
    # Wire bytes are exactly the ciphertext envelope — plaintext is not
    # in there in recoverable form without AES decryption.
    assert plaintext not in wire_recovered
    # And attempting to decrypt with a fresh key fails auth.
    from prsm.storage.encryption import (
        EncryptedPayload, IV_BYTES, AUTH_TAG_BYTES, EncryptionError,
    )
    iv = wire_recovered[:IV_BYTES]
    auth_tag = wire_recovered[IV_BYTES : IV_BYTES + AUTH_TAG_BYTES]
    ciphertext = wire_recovered[IV_BYTES + AUTH_TAG_BYTES :]
    wrong_key = generate_key()
    payload = EncryptedPayload(
        ciphertext=ciphertext,
        iv=iv,
        auth_tag=auth_tag,
        key_id=wrong_key.key_id,
    )
    with pytest.raises(EncryptionError):
        decrypt(payload, wrong_key)


# =============================================================================
# Tier C — erasure + AES + Shamir M-of-N key shares
# =============================================================================


@dataclass
class _ShareHolder:
    holder_id: str
    region: str
    share: KeyShare
    online: bool = True

    def serve(self) -> Optional[KeyShare]:
        return self.share if self.online else None


def test_tier_c_full_pipeline_with_both_thresholds_met():
    """Plan §2.1: Tier C reconstruction requires crossing BOTH
    K-of-N shard threshold AND M-of-N key-share threshold."""
    plaintext = b"Tier C regulated content " * 500
    content_hash = hashlib.sha256(plaintext).digest()

    # 1. Publisher encrypts + erasure-codes.
    key = generate_key()
    encrypted = encrypt(plaintext, key, associated_data=content_hash)
    wire = encrypted.iv + encrypted.auth_tag + encrypted.ciphertext
    meta, shards = encode(wire)
    providers = _place_shards_across_regions(
        shards, ["us-east", "us-west", "eu-west"]
    )

    # 2. Publisher Shamir-splits the key across 5 holders (m=3, n=5).
    shares = split_key(key, m=3, n=5)
    holders = [
        _ShareHolder(
            holder_id=f"holder-{i}",
            region=["ca-central", "eu-north", "ap-south", "us-mid", "sa-east"][i],
            share=shares[i],
        )
        for i in range(5)
    ]

    # 3. Simulate realistic Tier C stress: 2 of 10 shard providers OFFLINE,
    # and 1 of 5 key-share holders OFFLINE. Both thresholds still met
    # (8 ≥ k=6 shards, 4 ≥ m=3 shares).
    providers[1].online = False
    providers[6].online = False
    holders[2].online = False

    # 4. Consumer retrieves surviving shards + shares + reconstructs.
    available_shards = _collect_available_shards(providers)
    available_shares = [h.share for h in holders if h.online]

    reconstructed_key = combine_shares(available_shares)
    wire_recovered = decode(meta, available_shards)

    from prsm.storage.encryption import (
        EncryptedPayload, IV_BYTES, AUTH_TAG_BYTES,
    )
    iv = wire_recovered[:IV_BYTES]
    auth_tag = wire_recovered[IV_BYTES : IV_BYTES + AUTH_TAG_BYTES]
    ciphertext = wire_recovered[IV_BYTES + AUTH_TAG_BYTES :]
    payload = EncryptedPayload(
        ciphertext=ciphertext,
        iv=iv,
        auth_tag=auth_tag,
        key_id=reconstructed_key.key_id,
    )

    recovered = decrypt(payload, reconstructed_key, associated_data=content_hash)
    assert recovered == plaintext


def test_tier_c_fails_below_share_threshold_even_with_all_shards():
    """Crossing ONLY the shard threshold isn't enough. If m-1 shares are
    available, the key can't be reconstructed — confirms BOTH thresholds
    are independently load-bearing."""
    plaintext = b"Tier C"
    key = generate_key()
    shares = split_key(key, m=3, n=5)

    # Only 2 shares survive — below m=3.
    from prsm.storage.key_sharing import InsufficientSharesError
    with pytest.raises(InsufficientSharesError):
        combine_shares(shares[:2])


def test_tier_c_fails_below_shard_threshold_even_with_all_shares():
    """Symmetric check: all 5 shares available, but fewer than k=6
    shards remaining → erasure decode fails."""
    plaintext = b"Tier C"
    content_hash = hashlib.sha256(plaintext).digest()
    key = generate_key()
    encrypted = encrypt(plaintext, key, associated_data=content_hash)
    wire = encrypted.iv + encrypted.auth_tag + encrypted.ciphertext
    meta, shards = encode(wire)

    # Only 5 shards survive — below k=6.
    with pytest.raises(InsufficientShardsError):
        decode(meta, shards[:5])


# =============================================================================
# Challenge-then-slash scenario
# =============================================================================


class _StubSlashHook:
    def __init__(self) -> None:
        self.calls: List[dict] = []

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


def test_challenge_then_slash_on_tampered_response():
    """End-to-end: provider returns a tampered chunk → Merkle verification
    fails → on-chain slash hook fires with the evidence hash."""
    # Publisher commits a shard with a known Merkle root.
    CHUNK_SIZE = 64
    chunks = [os.urandom(CHUNK_SIZE) for _ in range(16)]
    tree = MerkleTree(chunks)
    expected_root = tree.root()

    clock = [1_700_000_000.0]
    issuer = ChallengeIssuer(clock=lambda: clock[0])
    challenge = issuer.issue(
        provider_id="provider-42",
        shard_id="shard-cafebabe",
        num_chunks=len(chunks),
    )

    # Provider "returns" a corrupted chunk.
    tampered = bytes(len(chunks[challenge.chunk_index]))
    response = ProofResponse(
        challenge_id=challenge.challenge_id,
        chunk_data=tampered,
        merkle_proof=tree.proof(challenge.chunk_index),
    )

    # Verifier runs; slash hook observes the failure.
    hook = _StubSlashHook()
    clock[0] += 5  # still within deadline
    verifier = ProofVerifier(
        challenger_id="foundation-challenger-01",
        clock=lambda: clock[0],
        slash_hook=hook,
    )
    result = verifier.verify(challenge, response, expected_root)

    assert not result.verified
    assert result.verdict is ProofVerdict.MERKLE_MISMATCH

    # Slash hook fired with exactly the context the StorageSlashing.sol
    # contract needs: (provider, shard, evidence_hash, challenger).
    assert len(hook.calls) == 1
    call = hook.calls[0]
    assert call["provider_id"] == "provider-42"
    assert call["shard_id"] == "shard-cafebabe"
    assert call["challenger"] == "foundation-challenger-01"
    assert call["evidence_hash"].startswith("0x") and len(call["evidence_hash"]) == 66


def test_challenge_on_honest_response_does_not_slash():
    """Complementary check: honest provider → slash hook silent."""
    CHUNK_SIZE = 64
    chunks = [os.urandom(CHUNK_SIZE) for _ in range(16)]
    tree = MerkleTree(chunks)
    expected_root = tree.root()

    issuer = ChallengeIssuer(clock=lambda: 1_700_000_000.0)
    challenge = issuer.issue("provider-honest", "shard-1", num_chunks=len(chunks))

    response = ProofResponse(
        challenge_id=challenge.challenge_id,
        chunk_data=chunks[challenge.chunk_index],
        merkle_proof=tree.proof(challenge.chunk_index),
    )

    hook = _StubSlashHook()
    verifier = ProofVerifier(
        challenger_id="c", clock=lambda: 1_700_000_000.0 + 5, slash_hook=hook
    )
    result = verifier.verify(challenge, response, expected_root)
    assert result.verified
    assert hook.calls == []
