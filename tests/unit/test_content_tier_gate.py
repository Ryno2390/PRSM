"""
Unit tests — Phase 3.x.1 Task 3 — content tier gate.

Acceptance per design plan §4 Task 3:
  - Tier A pass-through (no TEE required)
  - Tier B requires AESKey + attested TEE context; auth failures surface
  - Tier C requires K-of-N erasure shards + M-of-N Shamir shares + attested TEE
  - Decryption fails outside TEE context (TEEContextRequiredError)

These tests exercise the real cryptographic primitives in
``prsm.storage.encryption / erasure / key_sharing``. No mocks of the
crypto layer — that would be exactly the kind of "deprecate to simpler
testing" the project guidelines forbid.
"""

import os

import pytest

from prsm.compute.inference.content_tier_gate import (
    ContentTierGateError,
    MissingMaterialError,
    TEEContext,
    TEEContextRequiredError,
    TierBMaterial,
    TierCMaterial,
    open_content,
    open_tier_a,
    open_tier_b,
    open_tier_c,
)
from prsm.compute.inference.models import ContentTier
from prsm.compute.tee.models import TEEType
from prsm.storage.encryption import EncryptedPayload, encrypt, generate_key
from prsm.storage.erasure import encode as erasure_encode
from prsm.storage.key_sharing import split_key


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def plaintext():
    return b"the canonical PRSM Tier B/C secret payload - handle with care"


@pytest.fixture
def tier_b_material(plaintext):
    key = generate_key()
    payload = encrypt(plaintext, key)
    return TierBMaterial(payload=payload, key=key)


@pytest.fixture
def tier_c_material(plaintext):
    """Build full Tier C material from `plaintext`.

    Encrypts under a fresh AES key, then splits the *ciphertext* into
    K-of-N erasure shards and the *key* into M-of-N Shamir shares. The
    iv + auth_tag are carried in the material since they're not
    erasure-coded.
    """
    key = generate_key()
    payload = encrypt(plaintext, key)
    erasure_metadata, shards = erasure_encode(payload.ciphertext, k=3, n=5)
    key_shares = split_key(key, m=3, n=5)
    return TierCMaterial(
        erasure_metadata=erasure_metadata,
        erasure_shards=shards,
        key_shares=key_shares,
        iv=payload.iv,
        auth_tag=payload.auth_tag,
        key_id=key.key_id,
    )


@pytest.fixture
def hardware_tee_ctx():
    """An attested hardware TEE context (any of the hardware types qualifies)."""
    return TEEContext(tee_type=TEEType.SGX, allow_software_tee=False)


@pytest.fixture
def software_tee_ctx_disallowed():
    """Software TEE without the dev-allow flag — should fail attestation."""
    return TEEContext(tee_type=TEEType.SOFTWARE, allow_software_tee=False)


@pytest.fixture
def software_tee_ctx_allowed():
    """Software TEE with the dev-allow flag — should pass attestation."""
    return TEEContext(tee_type=TEEType.SOFTWARE, allow_software_tee=True)


@pytest.fixture
def no_tee_ctx():
    """No TEE at all — must always fail Tier B/C attestation."""
    return TEEContext(tee_type=TEEType.NONE, allow_software_tee=True)


# ──────────────────────────────────────────────────────────────────────────
# TEEContext.is_attested
# ──────────────────────────────────────────────────────────────────────────


class TestTEEContextAttestation:
    def test_hardware_tee_is_attested(self):
        for tee in [TEEType.SGX, TEEType.TDX, TEEType.SEV,
                    TEEType.TRUSTZONE, TEEType.SECURE_ENCLAVE]:
            ctx = TEEContext(tee_type=tee)
            assert ctx.is_attested, f"{tee.value} must attest"

    def test_software_tee_disallowed_not_attested(self):
        ctx = TEEContext(tee_type=TEEType.SOFTWARE, allow_software_tee=False)
        assert not ctx.is_attested

    def test_software_tee_allowed_is_attested(self):
        ctx = TEEContext(tee_type=TEEType.SOFTWARE, allow_software_tee=True)
        assert ctx.is_attested

    def test_no_tee_never_attested_even_with_allow_flag(self, no_tee_ctx):
        # The allow_software_tee flag should NOT escalate TEEType.NONE.
        assert not no_tee_ctx.is_attested

    def test_hardware_tee_attested_regardless_of_allow_flag(self):
        # allow_software_tee is irrelevant for hardware-backed TEE types.
        ctx = TEEContext(tee_type=TEEType.SGX, allow_software_tee=False)
        assert ctx.is_attested


# ──────────────────────────────────────────────────────────────────────────
# Tier A — pass-through
# ──────────────────────────────────────────────────────────────────────────


class TestTierA:
    def test_tier_a_pass_through(self, plaintext):
        assert open_tier_a(plaintext) == plaintext

    def test_tier_a_no_tee_needed(self, plaintext):
        # Whole point of Tier A: works regardless of context.
        result = open_content(ContentTier.A, plaintext=plaintext)
        assert result == plaintext

    def test_tier_a_empty_payload(self):
        assert open_tier_a(b"") == b""

    def test_tier_a_dispatch_without_plaintext_raises(self):
        with pytest.raises(MissingMaterialError, match="Tier A"):
            open_content(ContentTier.A)


# ──────────────────────────────────────────────────────────────────────────
# Tier B — encrypted, TEE-gated
# ──────────────────────────────────────────────────────────────────────────


class TestTierB:
    def test_hardware_tee_decrypts_successfully(
        self, tier_b_material, hardware_tee_ctx, plaintext
    ):
        assert open_tier_b(tier_b_material, hardware_tee_ctx) == plaintext

    def test_software_tee_allowed_decrypts(
        self, tier_b_material, software_tee_ctx_allowed, plaintext
    ):
        assert open_tier_b(tier_b_material, software_tee_ctx_allowed) == plaintext

    def test_software_tee_disallowed_raises(
        self, tier_b_material, software_tee_ctx_disallowed
    ):
        with pytest.raises(TEEContextRequiredError, match="hardware-attested"):
            open_tier_b(tier_b_material, software_tee_ctx_disallowed)

    def test_no_tee_raises(self, tier_b_material, no_tee_ctx):
        with pytest.raises(TEEContextRequiredError):
            open_tier_b(tier_b_material, no_tee_ctx)

    def test_tampered_ciphertext_raises(self, tier_b_material, hardware_tee_ctx):
        tampered = EncryptedPayload(
            ciphertext=tier_b_material.payload.ciphertext + b"X",
            iv=tier_b_material.payload.iv,
            auth_tag=tier_b_material.payload.auth_tag,
            key_id=tier_b_material.payload.key_id,
        )
        material = TierBMaterial(payload=tampered, key=tier_b_material.key)
        with pytest.raises(ContentTierGateError, match="decryption failed"):
            open_tier_b(material, hardware_tee_ctx)

    def test_wrong_key_id_raises(self, tier_b_material, hardware_tee_ctx):
        # Generate an unrelated key — its key_id won't match the payload's.
        wrong_key = generate_key()
        material = TierBMaterial(payload=tier_b_material.payload, key=wrong_key)
        with pytest.raises(ContentTierGateError):
            open_tier_b(material, hardware_tee_ctx)

    def test_dispatch_with_tier_b(
        self, tier_b_material, hardware_tee_ctx, plaintext
    ):
        result = open_content(
            ContentTier.B, tier_b=tier_b_material, ctx=hardware_tee_ctx
        )
        assert result == plaintext

    def test_dispatch_tier_b_without_material_raises(self, hardware_tee_ctx):
        with pytest.raises(MissingMaterialError, match="Tier B"):
            open_content(ContentTier.B, ctx=hardware_tee_ctx)

    def test_dispatch_tier_b_without_ctx_raises(self, tier_b_material):
        with pytest.raises(TEEContextRequiredError, match="Tier B"):
            open_content(ContentTier.B, tier_b=tier_b_material)


# ──────────────────────────────────────────────────────────────────────────
# Tier C — erasure-coded + Shamir-split, TEE-gated
# ──────────────────────────────────────────────────────────────────────────


class TestTierC:
    def test_full_quorum_decrypts(
        self, tier_c_material, hardware_tee_ctx, plaintext
    ):
        # All 5 shares + all 5 shards present → must decrypt.
        assert open_tier_c(tier_c_material, hardware_tee_ctx) == plaintext

    def test_minimum_quorum_decrypts(
        self, tier_c_material, hardware_tee_ctx, plaintext
    ):
        # Exactly K=3 shards + exactly M=3 shares should suffice.
        # Use the first three of each.
        material = TierCMaterial(
            erasure_metadata=tier_c_material.erasure_metadata,
            erasure_shards=tier_c_material.erasure_shards[:3],
            key_shares=tier_c_material.key_shares[:3],
            iv=tier_c_material.iv,
            auth_tag=tier_c_material.auth_tag,
            key_id=tier_c_material.key_id,
        )
        assert open_tier_c(material, hardware_tee_ctx) == plaintext

    def test_no_tee_raises(self, tier_c_material, no_tee_ctx):
        with pytest.raises(TEEContextRequiredError):
            open_tier_c(tier_c_material, no_tee_ctx)

    def test_software_tee_disallowed_raises(
        self, tier_c_material, software_tee_ctx_disallowed
    ):
        with pytest.raises(TEEContextRequiredError, match="hardware-attested"):
            open_tier_c(tier_c_material, software_tee_ctx_disallowed)

    def test_insufficient_key_shares_raises(
        self, tier_c_material, hardware_tee_ctx
    ):
        # Only 2 shares, need 3 → MissingMaterialError.
        material = TierCMaterial(
            erasure_metadata=tier_c_material.erasure_metadata,
            erasure_shards=tier_c_material.erasure_shards,
            key_shares=tier_c_material.key_shares[:2],
            iv=tier_c_material.iv,
            auth_tag=tier_c_material.auth_tag,
            key_id=tier_c_material.key_id,
        )
        with pytest.raises(MissingMaterialError, match="key reconstruction"):
            open_tier_c(material, hardware_tee_ctx)

    def test_insufficient_erasure_shards_raises(
        self, tier_c_material, hardware_tee_ctx
    ):
        # Only 2 shards, need 3 → MissingMaterialError on erasure decode.
        material = TierCMaterial(
            erasure_metadata=tier_c_material.erasure_metadata,
            erasure_shards=tier_c_material.erasure_shards[:2],
            key_shares=tier_c_material.key_shares,
            iv=tier_c_material.iv,
            auth_tag=tier_c_material.auth_tag,
            key_id=tier_c_material.key_id,
        )
        with pytest.raises(MissingMaterialError, match="erasure"):
            open_tier_c(material, hardware_tee_ctx)

    def test_corrupt_shard_raises(
        self, tier_c_material, hardware_tee_ctx
    ):
        # Tamper with one shard — its self-verify check inside erasure_decode
        # should raise CorruptShardError, surfacing as MissingMaterialError.
        from prsm.storage.erasure import ErasureShard
        good = list(tier_c_material.erasure_shards)
        bad = ErasureShard(
            index=good[0].index,
            data=good[0].data[:-1] + bytes([good[0].data[-1] ^ 0xFF]),
            sha256=good[0].sha256,  # stale hash → triggers CorruptShardError
        )
        good[0] = bad
        material = TierCMaterial(
            erasure_metadata=tier_c_material.erasure_metadata,
            erasure_shards=good,
            key_shares=tier_c_material.key_shares,
            iv=tier_c_material.iv,
            auth_tag=tier_c_material.auth_tag,
            key_id=tier_c_material.key_id,
        )
        with pytest.raises(MissingMaterialError):
            open_tier_c(material, hardware_tee_ctx)

    def test_tampered_iv_raises_auth_error(
        self, tier_c_material, hardware_tee_ctx
    ):
        # IV tamper → AEAD authentication fails after reconstruction.
        material = TierCMaterial(
            erasure_metadata=tier_c_material.erasure_metadata,
            erasure_shards=tier_c_material.erasure_shards,
            key_shares=tier_c_material.key_shares,
            iv=os.urandom(12),  # fresh, wrong IV
            auth_tag=tier_c_material.auth_tag,
            key_id=tier_c_material.key_id,
        )
        with pytest.raises(ContentTierGateError, match="decryption failed"):
            open_tier_c(material, hardware_tee_ctx)

    def test_wrong_key_id_in_material_raises(
        self, tier_c_material, hardware_tee_ctx
    ):
        # Pretend the material's key_id is something other than what
        # combine_shares will actually reconstruct → pairing check fires.
        material = TierCMaterial(
            erasure_metadata=tier_c_material.erasure_metadata,
            erasure_shards=tier_c_material.erasure_shards,
            key_shares=tier_c_material.key_shares,
            iv=tier_c_material.iv,
            auth_tag=tier_c_material.auth_tag,
            key_id="00000000-0000-0000-0000-000000000000",  # bogus
        )
        with pytest.raises(ContentTierGateError, match="key_id mismatch"):
            open_tier_c(material, hardware_tee_ctx)

    def test_dispatch_with_tier_c(
        self, tier_c_material, hardware_tee_ctx, plaintext
    ):
        result = open_content(
            ContentTier.C, tier_c=tier_c_material, ctx=hardware_tee_ctx
        )
        assert result == plaintext

    def test_dispatch_tier_c_without_material_raises(self, hardware_tee_ctx):
        with pytest.raises(MissingMaterialError, match="Tier C"):
            open_content(ContentTier.C, ctx=hardware_tee_ctx)

    def test_dispatch_tier_c_without_ctx_raises(self, tier_c_material):
        with pytest.raises(TEEContextRequiredError, match="Tier C"):
            open_content(ContentTier.C, tier_c=tier_c_material)


# ──────────────────────────────────────────────────────────────────────────
# Cross-tier — exception types are distinct + dispatcher hygiene
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_tee_context_required_is_gate_error(self):
        assert issubclass(TEEContextRequiredError, ContentTierGateError)

    def test_missing_material_is_gate_error(self):
        assert issubclass(MissingMaterialError, ContentTierGateError)

    def test_unknown_tier_raises_value_error(self, plaintext):
        # ContentTier accepts only A/B/C; passing a stray value bypasses
        # the Enum check at the dispatch layer.
        with pytest.raises(ValueError, match="unknown content tier"):
            # The dispatcher's final branch fires when none of the tier
            # comparisons match. Construct an obviously-invalid value.
            class FakeTier:
                value = "Z"
            open_content(FakeTier(), plaintext=plaintext)
