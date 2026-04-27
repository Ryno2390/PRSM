"""
Unit tests — Phase 3.x.3 Task 4 — anchor-backed verifier wrappers.

Acceptance per design plan §4 Task 4: happy path (anchor returns key,
verify succeeds), unregistered publisher returns False (anchor returns
None), tampered artifact returns False (anchor returns key, verify
fails). For each of the three artifact types — ModelManifest,
PrivacyBudgetEntry, InferenceReceipt.

Real Ed25519 signatures + real artifact construction; the anchor
client is the only mock (its lookup() returns a predetermined value).
"""

from __future__ import annotations

import dataclasses
import hashlib
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from prsm.compute.inference.models import (
    ContentTier,
    InferenceReceipt,
)
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.model_registry.models import (
    ManifestShardEntry,
    ModelManifest,
)
from prsm.compute.model_registry.signing import sign_manifest
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.security.privacy_budget_persistence.models import (
    GENESIS_PREV_HASH,
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
)
from prsm.security.privacy_budget_persistence.signing import sign_entry
from prsm.security.publisher_key_anchor import (
    AnchorRPCError,
    verify_entry_with_anchor,
    verify_manifest_with_anchor,
    verify_receipt_with_anchor,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.3-task4-publisher")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.3-task4-other")


@pytest.fixture
def anchor(identity):
    """Mock anchor client that resolves identity.node_id → identity.public_key_b64."""
    mock = MagicMock()

    def _lookup(node_id):
        if node_id == identity.node_id:
            return identity.public_key_b64
        return None

    mock.lookup = MagicMock(side_effect=_lookup)
    return mock


# ──────────────────────────────────────────────────────────────────────────
# verify_manifest_with_anchor
# ──────────────────────────────────────────────────────────────────────────


def _build_manifest(num_shards: int = 2) -> ModelManifest:
    shards = []
    for i in range(num_shards):
        data = f"shard-{i}".encode()
        shards.append(
            ManifestShardEntry(
                shard_id=f"sid-{i}",
                shard_index=i,
                tensor_shape=(8, 16),
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    return ModelManifest(
        model_id="llama-3-8b",
        model_name="Llama 3 8B",
        publisher_node_id="placeholder",  # overwritten by sign_manifest
        total_shards=num_shards,
        shards=tuple(shards),
        published_at=1714000000.0,
    )


class TestManifestVerifier:
    def test_happy_path(self, anchor, identity):
        signed = sign_manifest(_build_manifest(), identity)
        assert verify_manifest_with_anchor(signed, anchor) is True
        anchor.lookup.assert_called_once_with(identity.node_id)

    def test_unregistered_publisher_returns_false(self, anchor, other_identity):
        # Sign with other_identity; anchor only knows identity.
        signed = sign_manifest(_build_manifest(), other_identity)
        assert verify_manifest_with_anchor(signed, anchor) is False

    def test_tampered_manifest_returns_false(self, anchor, identity):
        signed = sign_manifest(_build_manifest(), identity)
        # Tamper a signed field; signature now invalid.
        tampered = dataclasses.replace(signed, model_name="Different Name")
        assert verify_manifest_with_anchor(tampered, anchor) is False

    def test_unsigned_manifest_returns_false(self, anchor):
        # An unsigned manifest carries empty bytes; even with the right
        # anchored key, verification is False.
        unsigned = _build_manifest()
        # publisher_node_id is "placeholder" → anchor returns None →
        # short-circuits to False before even checking signature.
        assert verify_manifest_with_anchor(unsigned, anchor) is False

    def test_anchor_rpc_failure_propagates(self, identity):
        signed = sign_manifest(_build_manifest(), identity)
        broken_anchor = MagicMock()
        broken_anchor.lookup = MagicMock(
            side_effect=AnchorRPCError("rpc unreachable")
        )
        with pytest.raises(AnchorRPCError):
            verify_manifest_with_anchor(signed, broken_anchor)


# ──────────────────────────────────────────────────────────────────────────
# verify_entry_with_anchor
# ──────────────────────────────────────────────────────────────────────────


def _build_entry() -> PrivacyBudgetEntry:
    return PrivacyBudgetEntry(
        sequence_number=0,
        entry_type=PrivacyBudgetEntryType.SPEND,
        node_id="placeholder",  # overwritten by sign_entry
        epsilon=8.0,
        operation="inference",
        model_id="llama-3-8b",
        timestamp=1714000000.0,
        prev_entry_hash=GENESIS_PREV_HASH,
    )


class TestEntryVerifier:
    def test_happy_path(self, anchor, identity):
        signed = sign_entry(_build_entry(), identity)
        assert verify_entry_with_anchor(signed, anchor) is True
        anchor.lookup.assert_called_once_with(identity.node_id)

    def test_unregistered_publisher_returns_false(self, anchor, other_identity):
        signed = sign_entry(_build_entry(), other_identity)
        assert verify_entry_with_anchor(signed, anchor) is False

    def test_tampered_entry_returns_false(self, anchor, identity):
        signed = sign_entry(_build_entry(), identity)
        # Reduce ε on disk → operator dodges budget. Must fail.
        tampered = dataclasses.replace(signed, epsilon=signed.epsilon / 2)
        assert verify_entry_with_anchor(tampered, anchor) is False

    def test_unsigned_entry_returns_false(self, anchor):
        unsigned = _build_entry()
        assert verify_entry_with_anchor(unsigned, anchor) is False

    def test_anchor_rpc_failure_propagates(self, identity):
        signed = sign_entry(_build_entry(), identity)
        broken_anchor = MagicMock()
        broken_anchor.lookup = MagicMock(
            side_effect=AnchorRPCError("rpc unreachable")
        )
        with pytest.raises(AnchorRPCError):
            verify_entry_with_anchor(signed, broken_anchor)


# ──────────────────────────────────────────────────────────────────────────
# verify_receipt_with_anchor
# ──────────────────────────────────────────────────────────────────────────


def _build_receipt() -> InferenceReceipt:
    return InferenceReceipt(
        job_id="job-abc",
        request_id="req-xyz",
        model_id="llama-3-8b",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
        epsilon_spent=8.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"\x00" * 64,
        output_hash=hashlib.sha256(b"output").digest(),
        duration_seconds=0.5,
        cost_ftns=Decimal("0.10"),
        # settler_node_id stamped by sign_receipt
    )


class TestReceiptVerifier:
    def test_happy_path(self, anchor, identity):
        signed = sign_receipt(_build_receipt(), identity)
        assert verify_receipt_with_anchor(signed, anchor) is True
        # Receipts are signed by the SETTLING node — wrapper looks up
        # settler_node_id, not any publisher field.
        anchor.lookup.assert_called_once_with(identity.node_id)

    def test_unregistered_settler_returns_false(self, anchor, other_identity):
        signed = sign_receipt(_build_receipt(), other_identity)
        assert verify_receipt_with_anchor(signed, anchor) is False

    def test_tampered_receipt_returns_false(self, anchor, identity):
        signed = sign_receipt(_build_receipt(), identity)
        tampered = dataclasses.replace(signed, cost_ftns=Decimal("999.0"))
        assert verify_receipt_with_anchor(tampered, anchor) is False

    def test_unsigned_receipt_returns_false(self, anchor):
        unsigned = _build_receipt()
        # settler_node_id is "" by default → anchor returns None.
        assert verify_receipt_with_anchor(unsigned, anchor) is False

    def test_anchor_rpc_failure_propagates(self, identity):
        signed = sign_receipt(_build_receipt(), identity)
        broken_anchor = MagicMock()
        broken_anchor.lookup = MagicMock(
            side_effect=AnchorRPCError("rpc unreachable")
        )
        with pytest.raises(AnchorRPCError):
            verify_receipt_with_anchor(signed, broken_anchor)


# ──────────────────────────────────────────────────────────────────────────
# Cross-artifact wrappers consistently use the right id field
# ──────────────────────────────────────────────────────────────────────────


class TestCorrectIdField:
    """Each wrapper looks up the correct id field on its artifact type.
    Documents the mapping so a future refactor that renames a field
    breaks this test rather than silently calling lookup with the
    wrong value."""

    def test_manifest_uses_publisher_node_id(self, identity):
        signed = sign_manifest(_build_manifest(), identity)
        anchor = MagicMock()
        anchor.lookup = MagicMock(return_value=identity.public_key_b64)
        verify_manifest_with_anchor(signed, anchor)
        anchor.lookup.assert_called_once_with(signed.publisher_node_id)
        assert signed.publisher_node_id == identity.node_id

    def test_entry_uses_node_id(self, identity):
        signed = sign_entry(_build_entry(), identity)
        anchor = MagicMock()
        anchor.lookup = MagicMock(return_value=identity.public_key_b64)
        verify_entry_with_anchor(signed, anchor)
        anchor.lookup.assert_called_once_with(signed.node_id)
        assert signed.node_id == identity.node_id

    def test_receipt_uses_settler_node_id(self, identity):
        signed = sign_receipt(_build_receipt(), identity)
        anchor = MagicMock()
        anchor.lookup = MagicMock(return_value=identity.public_key_b64)
        verify_receipt_with_anchor(signed, anchor)
        anchor.lookup.assert_called_once_with(signed.settler_node_id)
        assert signed.settler_node_id == identity.node_id


# ──────────────────────────────────────────────────────────────────────────
# Cross-artifact replay protection still holds end-to-end
# ──────────────────────────────────────────────────────────────────────────


class TestCrossArtifactReplayProtection:
    """A signature over one artifact type must not verify against
    another type, even when the anchor returns the right key. This
    is the cross-artifact domain-separation property that the
    underlying verifiers enforce; the anchor wrappers must NOT defeat
    it."""

    def test_manifest_sig_does_not_verify_as_entry(self, anchor, identity):
        manifest = sign_manifest(_build_manifest(), identity)
        # Build an entry whose sig comes from the manifest. Even with
        # the right anchored key, verify_entry rejects: the entry's
        # signing payload uses a different domain prefix.
        forged_entry = dataclasses.replace(
            sign_entry(_build_entry(), identity),
            signature=manifest.publisher_signature,
        )
        assert verify_entry_with_anchor(forged_entry, anchor) is False

    def test_entry_sig_does_not_verify_as_manifest(self, anchor, identity):
        entry = sign_entry(_build_entry(), identity)
        forged_manifest = dataclasses.replace(
            sign_manifest(_build_manifest(), identity),
            publisher_signature=entry.signature,
        )
        assert verify_manifest_with_anchor(forged_manifest, anchor) is False

    def test_receipt_sig_does_not_verify_as_manifest(self, anchor, identity):
        receipt = sign_receipt(_build_receipt(), identity)
        forged_manifest = dataclasses.replace(
            sign_manifest(_build_manifest(), identity),
            publisher_signature=receipt.settler_signature,
        )
        assert verify_manifest_with_anchor(forged_manifest, anchor) is False
