"""
Unit tests — Phase 3.x.2 Task 2 — manifest sign/verify.

Acceptance per design plan §4 Task 2: sign/verify works against a real
NodeIdentity; tampering caught for every signed field; no parallel crypto.

Per project testing rules, these tests use real Ed25519 keypairs and
real NodeIdentity instances — no crypto mocks. The signing payload is
a wire format; the tests assert against actual signatures.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib

import pytest

from prsm.compute.model_registry.models import (
    ManifestShardEntry,
    ModelManifest,
)
from prsm.compute.model_registry.signing import (
    is_signed,
    sign_manifest,
    verify_manifest,
)
from prsm.node.identity import NodeIdentity, generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _entry(idx: int, *, shape=(8, 16)) -> ManifestShardEntry:
    fake = f"shard-{idx}".encode("utf-8")
    return ManifestShardEntry(
        shard_id=f"sid-{idx}",
        shard_index=idx,
        tensor_shape=shape,
        sha256=hashlib.sha256(fake).hexdigest(),
        size_bytes=len(fake),
    )


def _manifest(*, num_shards: int = 3) -> ModelManifest:
    return ModelManifest(
        model_id="llama-3-8b",
        model_name="Llama 3 8B",
        publisher_node_id="placeholder",  # overwritten by sign_manifest
        total_shards=num_shards,
        shards=tuple(_entry(i) for i in range(num_shards)),
        published_at=1714000000.0,
    )


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.2-test-publisher")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.2-test-impostor")


@pytest.fixture
def signed(identity) -> ModelManifest:
    return sign_manifest(_manifest(), identity)


# ──────────────────────────────────────────────────────────────────────────
# sign_manifest
# ──────────────────────────────────────────────────────────────────────────


class TestSignManifest:
    def test_returns_new_manifest(self, identity):
        original = _manifest()
        signed = sign_manifest(original, identity)
        # Frozen dataclasses; sign_manifest must produce a fresh instance.
        assert signed is not original
        assert original.publisher_signature == b""
        assert signed.publisher_signature != b""

    def test_stamps_publisher_node_id(self, identity):
        signed = sign_manifest(_manifest(), identity)
        assert signed.publisher_node_id == identity.node_id

    def test_signature_is_64_bytes(self, identity):
        # Ed25519 signatures are always 64 bytes
        signed = sign_manifest(_manifest(), identity)
        assert len(signed.publisher_signature) == 64

    def test_signature_is_deterministic_for_same_manifest(self, identity):
        # Ed25519 is deterministic — same payload + same key → same sig
        s1 = sign_manifest(_manifest(), identity)
        s2 = sign_manifest(_manifest(), identity)
        assert s1.publisher_signature == s2.publisher_signature

    def test_signature_differs_across_publishers(self, identity, other_identity):
        s1 = sign_manifest(_manifest(), identity)
        s2 = sign_manifest(_manifest(), other_identity)
        assert s1.publisher_signature != s2.publisher_signature
        assert s1.publisher_node_id != s2.publisher_node_id

    def test_signature_differs_across_models(self, identity):
        m1 = dataclasses.replace(_manifest(), model_id="A")
        m2 = dataclasses.replace(_manifest(), model_id="B")
        s1 = sign_manifest(m1, identity)
        s2 = sign_manifest(m2, identity)
        assert s1.publisher_signature != s2.publisher_signature

    def test_overrides_placeholder_publisher(self, identity):
        # The unsigned manifest had publisher_node_id="placeholder"; signing
        # MUST overwrite it with the signer's actual node_id, else verifiers
        # would have to trust the placeholder.
        m = _manifest()
        assert m.publisher_node_id == "placeholder"
        signed = sign_manifest(m, identity)
        assert signed.publisher_node_id == identity.node_id


# ──────────────────────────────────────────────────────────────────────────
# verify_manifest — happy path
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyHappyPath:
    def test_verifies_against_signing_identity(self, identity, signed):
        assert verify_manifest(signed, identity=identity) is True

    def test_verifies_against_public_key_b64(self, identity, signed):
        assert verify_manifest(signed, public_key_b64=identity.public_key_b64) is True

    def test_verifies_after_dict_roundtrip(self, identity, signed):
        # FilesystemModelRegistry (Task 4) writes manifest.json and reads
        # it back; verification must survive that roundtrip.
        d = signed.to_dict()
        reloaded = ModelManifest.from_dict(d)
        assert verify_manifest(reloaded, identity=identity) is True


# ──────────────────────────────────────────────────────────────────────────
# verify_manifest — failure modes
# ──────────────────────────────────────────────────────────────────────────


class TestVerifyFailureModes:
    def test_unsigned_manifest_returns_false(self, identity):
        m = _manifest()  # publisher_signature=b""
        assert verify_manifest(m, identity=identity) is False

    def test_no_credential_returns_false(self, signed):
        # Caller forgot to pass identity OR public_key_b64
        assert verify_manifest(signed) is False

    def test_wrong_public_key_returns_false(self, signed, other_identity):
        # Right manifest, wrong key → reject
        assert verify_manifest(signed, identity=other_identity) is False
        assert verify_manifest(
            signed, public_key_b64=other_identity.public_key_b64
        ) is False

    def test_malformed_public_key_returns_false(self, signed):
        # Garbage public-key string must NOT raise — fail closed.
        assert verify_manifest(signed, public_key_b64="not-base64-!!!") is False

    def test_truncated_signature_returns_false(self, identity, signed):
        # Tamper: truncate the signature bytes
        bad = dataclasses.replace(
            signed, publisher_signature=signed.publisher_signature[:32]
        )
        assert verify_manifest(bad, identity=identity) is False

    def test_zero_signature_returns_false(self, identity, signed):
        bad = dataclasses.replace(signed, publisher_signature=b"\x00" * 64)
        assert verify_manifest(bad, identity=identity) is False


# ──────────────────────────────────────────────────────────────────────────
# Tampering — every signed field must invalidate the signature
# ──────────────────────────────────────────────────────────────────────────


class TestTamperDetection:
    """Confirms every load-bearing field in signing_payload is covered.

    If any of these test fail, the signing payload is missing a field —
    i.e., that field could be tampered with without breaking the signature.
    Catching this at test time is the whole point.
    """

    def test_tamper_model_id(self, identity, signed):
        bad = dataclasses.replace(signed, model_id="different")
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_model_name(self, identity, signed):
        bad = dataclasses.replace(signed, model_name="Different Name")
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_publisher_node_id(self, identity, signed):
        bad = dataclasses.replace(signed, publisher_node_id="different-publisher")
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_total_shards(self, identity, signed):
        bad = dataclasses.replace(signed, total_shards=signed.total_shards + 1)
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_published_at(self, identity, signed):
        bad = dataclasses.replace(signed, published_at=signed.published_at + 1.0)
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_schema_version(self, identity, signed):
        # Downgrade attack: re-stamp as schema v0 to evade a v1-aware verifier
        bad = dataclasses.replace(signed, schema_version=0)
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_shard_sha256(self, identity, signed):
        # Swap a shard's sha256 digest — the bytes promise breaks
        new_shards = list(signed.shards)
        new_shards[0] = dataclasses.replace(new_shards[0], sha256="ff" * 32)
        bad = dataclasses.replace(signed, shards=tuple(new_shards))
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_shard_id(self, identity, signed):
        new_shards = list(signed.shards)
        new_shards[0] = dataclasses.replace(new_shards[0], shard_id="malicious-id")
        bad = dataclasses.replace(signed, shards=tuple(new_shards))
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_shard_size_bytes(self, identity, signed):
        new_shards = list(signed.shards)
        new_shards[0] = dataclasses.replace(new_shards[0], size_bytes=999999)
        bad = dataclasses.replace(signed, shards=tuple(new_shards))
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_shard_tensor_shape(self, identity, signed):
        new_shards = list(signed.shards)
        new_shards[0] = dataclasses.replace(new_shards[0], tensor_shape=(99, 99))
        bad = dataclasses.replace(signed, shards=tuple(new_shards))
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_shard_reorder_doesnt_matter(self, identity, signed):
        # Shards are canonicalized by index in __post_init__, so a
        # reordered shard list must produce the SAME signing payload
        # and verify successfully.
        reversed_shards = tuple(reversed(signed.shards))
        same = dataclasses.replace(signed, shards=reversed_shards)
        assert verify_manifest(same, identity=identity) is True

    def test_tamper_add_extra_shard(self, identity, signed):
        # Adding a shard changes the per-shard line list AND total_shards
        new_shards = list(signed.shards) + [_entry(99)]
        bad = dataclasses.replace(
            signed, total_shards=len(new_shards), shards=tuple(new_shards)
        )
        assert verify_manifest(bad, identity=identity) is False

    def test_tamper_remove_shard(self, identity, signed):
        new_shards = list(signed.shards)[:-1]
        bad = dataclasses.replace(
            signed, total_shards=len(new_shards), shards=tuple(new_shards)
        )
        assert verify_manifest(bad, identity=identity) is False


# ──────────────────────────────────────────────────────────────────────────
# Signing-payload-excludes-signature property
# ──────────────────────────────────────────────────────────────────────────


class TestSigningExclusion:
    """If the signing payload included publisher_signature, signing would
    be circular: each new signature would change the payload and thus
    require a new signature, ad infinitum. Task 1 already pinned this in
    test_payload_excludes_signature; here we re-verify it from the
    signing-flow side.
    """

    def test_resigning_yields_identical_signature(self, identity):
        # Sign once, then sign again. Because the payload doesn't depend
        # on the signature, both signatures must be identical for
        # deterministic Ed25519.
        s1 = sign_manifest(_manifest(), identity)
        s2 = sign_manifest(s1, identity)  # already signed → re-sign anyway
        assert s1.publisher_signature == s2.publisher_signature

    def test_signature_field_doesnt_affect_payload_bytes(self, identity, signed):
        # The signed manifest's signing_payload must equal the
        # un-signed-but-stamped manifest's signing_payload.
        stamped_only = dataclasses.replace(
            _manifest(), publisher_node_id=identity.node_id
        )
        assert stamped_only.signing_payload() == signed.signing_payload()


# ──────────────────────────────────────────────────────────────────────────
# is_signed predicate
# ──────────────────────────────────────────────────────────────────────────


class TestIsSigned:
    def test_unsigned_returns_false(self):
        assert is_signed(_manifest()) is False

    def test_signed_returns_true(self, signed):
        assert is_signed(signed) is True

    def test_signature_only_no_node_id_returns_false(self):
        # Bizarre construction: signature populated but node_id empty.
        # Treat as unsigned (no provenance).
        m = dataclasses.replace(
            _manifest(),
            publisher_node_id="",
            publisher_signature=b"\xff" * 64,
        )
        assert is_signed(m) is False

    def test_node_id_only_no_signature_returns_false(self, identity):
        m = dataclasses.replace(_manifest(), publisher_node_id=identity.node_id)
        assert is_signed(m) is False

    def test_is_signed_is_NOT_a_crypto_check(self, identity, other_identity):
        # is_signed only checks field-population; verify_manifest is the
        # real check. A manifest signed by A but holding B's node_id
        # would pass is_signed (both fields populated) yet fail
        # verify_manifest — confirms documentation contract.
        signed_by_a = sign_manifest(_manifest(), identity)
        # Tamper the publisher_node_id post-signing
        confused = dataclasses.replace(
            signed_by_a, publisher_node_id=other_identity.node_id
        )
        assert is_signed(confused) is True   # field-level: yes
        assert verify_manifest(confused, identity=identity) is False   # crypto: no


# ──────────────────────────────────────────────────────────────────────────
# Cross-artifact replay protection — domain separation matters
# ──────────────────────────────────────────────────────────────────────────


class TestDomainSeparation:
    def test_manifest_signature_does_not_verify_as_receipt(self, identity, signed):
        # The manifest signing payload starts with b"prsm-model-manifest:v1";
        # InferenceReceipt's signing payload uses a different format.
        # This test confirms a manifest signature can't be replayed
        # against a receipt — even one whose payload happens to share
        # bytes with a manifest payload.
        # We reconstruct the manifest's signing bytes and verify them
        # under the publisher's pubkey directly. Then we confirm that
        # the same bytes prefixed differently (faking a different
        # artifact type) wouldn't verify.
        from prsm.node.identity import verify_signature
        sig_b64 = base64.b64encode(signed.publisher_signature).decode()

        # Real payload verifies
        assert verify_signature(
            identity.public_key_b64, signed.signing_payload(), sig_b64
        ) is True

        # Same signature against a different artifact's payload (e.g.,
        # the manifest payload with the domain tag rewritten) must fail
        original_bytes = signed.signing_payload()
        tampered_domain = b"prsm-inference-receipt:v1" + original_bytes[len(b"prsm-model-manifest:v1"):]
        assert verify_signature(
            identity.public_key_b64, tampered_domain, sig_b64
        ) is False
