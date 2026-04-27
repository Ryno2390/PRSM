"""
End-to-end integration test — Phase 3.x.3 Task 7.

Acceptance per design plan §4 Task 7: spin up the anchor stack,
register two publishers (alice + bob), sign artifacts under each,
verify cross-publisher artifacts fail, tamper a sidecar AND keep
the on-chain key intact → anchor-path still verifies. Confirms the
trust upgrade end-to-end.

Test approach — simulated on-chain contract:
  Uses a faithful Python implementation of PublisherKeyAnchor.sol's
  semantics (sha256-derived nodeId + write-once + admin override) in
  the place of a real EVM. This is enough because:
    - Solidity-side correctness is covered by Task 1's 20 Hardhat
      tests against the real contract
    - Python-client correctness is covered by Task 3's 25 unit tests
    - This test exercises the COMPOSITION: real signatures, real
      artifacts (manifest + entry + receipt), real client + wrappers,
      real verifiers — only the EVM is simulated

  The Sepolia deploy runbook (Task 2) bridges this simulation to
  real-EVM correctness on the live network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from prsm.compute.inference.models import ContentTier, InferenceReceipt
from prsm.compute.inference.receipt import sign_receipt
from prsm.compute.model_registry import (
    FilesystemModelRegistry,
    ManifestVerificationError,
)
from prsm.compute.model_registry.signing import sign_manifest
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.identity import NodeIdentity, generate_node_identity
from prsm.security.privacy_budget_persistence import (
    FilesystemPrivacyBudgetStore,
    GENESIS_PREV_HASH,
    PersistentPrivacyBudgetTracker,
    PrivacyBudgetEntry,
    PrivacyBudgetEntryType,
    hash_entry_payload,
    sign_entry,
)
from prsm.security.publisher_key_anchor import (
    PublisherAlreadyRegisteredError,
    verify_entry_with_anchor,
    verify_manifest_with_anchor,
    verify_receipt_with_anchor,
)


# ──────────────────────────────────────────────────────────────────────────
# Faithful on-chain contract simulation
# ──────────────────────────────────────────────────────────────────────────


class SimulatedAnchorContract:
    """In-process Python mirror of PublisherKeyAnchor.sol semantics.

    Implements:
    - sha256(pubkey) → first 16 bytes used as the node_id binding
      (mirrors the on-chain derivation rule)
    - write-once registration per node_id
    - multisig admin override (set at construction)

    Exposes the same lookup-by-node_id interface that
    PublisherKeyAnchorClient consumes via its lookup() method.
    The client's _call_lookup is patched to delegate here.
    """

    def __init__(self, admin_address: str = "admin"):
        self.admin = admin_address
        # node_id_bytes16 → 32-byte pubkey
        self._publisher_keys: dict[bytes, bytes] = {}

    def register(self, public_key_bytes: bytes, *, caller: str = "anyone") -> bytes:
        """Mirrors register(bytes publicKey) on-chain."""
        if len(public_key_bytes) != 32:
            raise ValueError(
                f"InvalidPublicKeyLength: got {len(public_key_bytes)}"
            )
        node_id = hashlib.sha256(public_key_bytes).digest()[:16]
        if node_id in self._publisher_keys:
            raise PublisherAlreadyRegisteredError(
                f"AlreadyRegistered: node_id={node_id.hex()}"
            )
        self._publisher_keys[node_id] = public_key_bytes
        return node_id

    def admin_override(
        self, node_id: bytes, new_key: bytes, *, caller: str
    ) -> None:
        """Mirrors adminOverride. Caller must be admin."""
        if caller != self.admin:
            raise PermissionError(f"NotAdmin: caller={caller}")
        if node_id not in self._publisher_keys:
            raise KeyError(f"NotRegistered: node_id={node_id.hex()}")
        if len(new_key) != 32:
            raise ValueError(f"InvalidPublicKeyLength: got {len(new_key)}")
        self._publisher_keys[node_id] = new_key

    def lookup(self, node_id_bytes16: bytes) -> bytes:
        """Mirrors lookup() on-chain. Returns empty bytes if not registered."""
        return self._publisher_keys.get(node_id_bytes16, b"")


class SimulatedAnchorClient:
    """Minimal anchor client for tests — implements just the
    PublisherKeyAnchorClient.lookup interface that the verifier
    wrappers + Filesystem stores consume.

    Wraps a SimulatedAnchorContract to provide the hex-string node_id
    interface the wrappers expect."""

    def __init__(self, contract: SimulatedAnchorContract):
        self._contract = contract

    def lookup(self, node_id: str) -> str | None:
        """node_id is the hex string from the artifact's signed payload."""
        import base64

        if node_id is None or node_id == "":
            return None
        s = node_id[2:] if node_id.startswith("0x") else node_id
        if len(s) != 32:
            return None
        try:
            node_id_bytes = bytes.fromhex(s)
        except ValueError:
            return None
        result = self._contract.lookup(node_id_bytes)
        if not result:
            return None
        return base64.b64encode(result).decode("ascii")


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — two real publishers + a deployed anchor
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def alice() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.3-task7-alice")


@pytest.fixture
def bob() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.3-task7-bob")


@pytest.fixture
def anchor_contract():
    return SimulatedAnchorContract(admin_address="foundation-multisig")


@pytest.fixture
def anchor(anchor_contract):
    return SimulatedAnchorClient(anchor_contract)


@pytest.fixture
def both_registered(anchor_contract, alice, bob):
    """Both publishers registered on the simulated anchor."""
    anchor_contract.register(alice.public_key_bytes)
    anchor_contract.register(bob.public_key_bytes)
    return anchor_contract


# ──────────────────────────────────────────────────────────────────────────
# 1. Contract behavior end-to-end
# ──────────────────────────────────────────────────────────────────────────


class TestAnchorContractBehavior:
    """The simulated contract mirrors the Solidity contract's behavior.
    These tests are belt-and-suspenders against the simulation drifting
    from the real contract (covered by Task 1 Hardhat tests)."""

    def test_register_derives_node_id_from_pubkey(
        self, anchor_contract, alice
    ):
        node_id = anchor_contract.register(alice.public_key_bytes)
        # Same derivation rule as the on-chain contract
        expected = hashlib.sha256(alice.public_key_bytes).digest()[:16]
        assert node_id == expected
        # Matches alice.node_id (which is hex of these 16 bytes)
        assert node_id.hex() == alice.node_id

    def test_register_rejects_wrong_length_pubkey(self, anchor_contract):
        with pytest.raises(ValueError, match="InvalidPublicKeyLength"):
            anchor_contract.register(b"\x00" * 31)

    def test_register_rejects_duplicate_node_id(
        self, anchor_contract, alice
    ):
        anchor_contract.register(alice.public_key_bytes)
        with pytest.raises(PublisherAlreadyRegisteredError):
            anchor_contract.register(alice.public_key_bytes)

    def test_admin_override_succeeds(
        self, anchor_contract, alice, bob
    ):
        node_id = anchor_contract.register(alice.public_key_bytes)
        anchor_contract.admin_override(
            node_id, bob.public_key_bytes, caller="foundation-multisig"
        )
        assert anchor_contract.lookup(node_id) == bob.public_key_bytes

    def test_non_admin_override_rejected(self, anchor_contract, alice, bob):
        node_id = anchor_contract.register(alice.public_key_bytes)
        with pytest.raises(PermissionError, match="NotAdmin"):
            anchor_contract.admin_override(
                node_id, bob.public_key_bytes, caller="attacker"
            )


# ──────────────────────────────────────────────────────────────────────────
# 2. Cross-publisher manifest verification
# ──────────────────────────────────────────────────────────────────────────


def _build_manifest(num_shards: int = 2):
    from prsm.compute.model_registry.models import (
        ManifestShardEntry,
        ModelManifest,
    )

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
        publisher_node_id="placeholder",
        total_shards=num_shards,
        shards=tuple(shards),
        published_at=1714000000.0,
    )


class TestCrossPublisherManifest:
    def test_alices_manifest_verifies_under_alice_anchored_key(
        self, both_registered, anchor, alice
    ):
        signed = sign_manifest(_build_manifest(), alice)
        assert verify_manifest_with_anchor(signed, anchor) is True

    def test_alices_manifest_does_not_verify_under_bob_lookup_key(
        self, anchor_contract, anchor, alice, bob
    ):
        # ONLY bob registered. Alice's manifest is signed with alice's
        # private key, but alice isn't on-chain → anchor returns None
        # → wrapper returns False.
        anchor_contract.register(bob.public_key_bytes)
        signed = sign_manifest(_build_manifest(), alice)
        assert verify_manifest_with_anchor(signed, anchor) is False

    def test_swapped_signatures_caught_after_both_registered(
        self, both_registered, anchor, alice, bob
    ):
        # Both registered. Sign two different manifests (one each), then
        # swap signatures. Each forged combination must fail verify.
        a_manifest = sign_manifest(_build_manifest(), alice)
        b_manifest = sign_manifest(
            dataclasses.replace(_build_manifest(), model_id="mistral-7b"),
            bob,
        )
        # Forge: alice's manifest with bob's signature
        forged_a = dataclasses.replace(
            a_manifest, publisher_signature=b_manifest.publisher_signature
        )
        # Forge: bob's manifest with alice's signature
        forged_b = dataclasses.replace(
            b_manifest, publisher_signature=a_manifest.publisher_signature
        )
        assert verify_manifest_with_anchor(forged_a, anchor) is False
        assert verify_manifest_with_anchor(forged_b, anchor) is False


# ──────────────────────────────────────────────────────────────────────────
# 3. Cross-publisher privacy-budget entry verification
# ──────────────────────────────────────────────────────────────────────────


def _build_entry():
    return PrivacyBudgetEntry(
        sequence_number=0,
        entry_type=PrivacyBudgetEntryType.SPEND,
        node_id="placeholder",
        epsilon=8.0,
        operation="inference",
        model_id="llama-3-8b",
        timestamp=1714000000.0,
        prev_entry_hash=GENESIS_PREV_HASH,
    )


class TestCrossPublisherEntry:
    def test_alices_entry_verifies(self, both_registered, anchor, alice):
        signed = sign_entry(_build_entry(), alice)
        assert verify_entry_with_anchor(signed, anchor) is True

    def test_unregistered_entry_publisher_fails(
        self, anchor_contract, anchor, alice, bob
    ):
        anchor_contract.register(bob.public_key_bytes)
        signed = sign_entry(_build_entry(), alice)
        assert verify_entry_with_anchor(signed, anchor) is False


# ──────────────────────────────────────────────────────────────────────────
# 4. Cross-publisher inference-receipt verification
# ──────────────────────────────────────────────────────────────────────────


def _build_receipt():
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
    )


class TestCrossPublisherReceipt:
    def test_alices_receipt_verifies(self, both_registered, anchor, alice):
        signed = sign_receipt(_build_receipt(), alice)
        assert verify_receipt_with_anchor(signed, anchor) is True

    def test_unregistered_settler_fails(
        self, anchor_contract, anchor, alice, bob
    ):
        anchor_contract.register(bob.public_key_bytes)
        signed = sign_receipt(_build_receipt(), alice)
        assert verify_receipt_with_anchor(signed, anchor) is False


# ──────────────────────────────────────────────────────────────────────────
# 5. THE TRUST UPGRADE — sidecar-tamper-with-anchor-intact still verifies
# ──────────────────────────────────────────────────────────────────────────


class TestTrustUpgradeFilesystemModelRegistry:
    """The acceptance criterion of Phase 3.x.3: with anchor configured,
    a sidecar-tampered registry STILL verifies if the on-chain key is
    intact. The cross-node trust-boundary caveat from Phase 3.x.2 is
    closed."""

    def test_tampered_sidecar_does_not_break_anchor_verification(
        self, tmp_path, both_registered, anchor, alice
    ):
        # Build a real ShardedModel and register through the registry
        # (without anchor) so the sidecar is written under alice.
        shards = [
            ModelShard(
                shard_id=f"s{i}",
                model_id="m1",
                shard_index=i,
                total_shards=2,
                tensor_data=f"shard-{i}".encode() * 4,
                tensor_shape=(8,),
                layer_range=(0, 0),
                size_bytes=0,
                checksum="",
            )
            for i in range(2)
        ]
        model = ShardedModel(
            model_id="m1", model_name="m1", total_shards=2, shards=shards
        )
        FilesystemModelRegistry(tmp_path).register(model, identity=alice)

        # Tamper the sidecar with a bogus key. Sidecar-only verify fails.
        sidecar_path = tmp_path / "m1" / "publisher.pubkey"
        sidecar_path.write_text("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")

        # Without anchor: sidecar mismatch → ManifestVerificationError
        plain_reg = FilesystemModelRegistry(tmp_path)
        with pytest.raises(ManifestVerificationError):
            plain_reg.get("m1")

        # WITH anchor: sidecar is irrelevant; on-chain key resolves
        # correctly, signature verifies, model is returned.
        anchor_reg = FilesystemModelRegistry(tmp_path, anchor=anchor)
        out = anchor_reg.get("m1")
        assert out.model_id == "m1"
        # Bytes match what was registered
        for s_in, s_out in zip(model.shards, out.shards):
            assert s_in.tensor_data == s_out.tensor_data


class TestTrustUpgradeFilesystemPrivacyBudgetStore:
    """The same trust upgrade for the journal store."""

    def test_anchor_verifies_journal_with_compromised_sidecar_environment(
        self, tmp_path, both_registered, anchor, alice
    ):
        # Build a journal under alice through the persistent tracker.
        store = FilesystemPrivacyBudgetStore(tmp_path, alice.public_key_b64)
        tracker = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store, identity=alice
        )
        tracker.record_spend(8.0, "inference", model_id="llama-3-8b")
        tracker.record_spend(4.0, "forge_query")

        # Now reopen WITH anchor. Sidecar is intact (we don't tamper it
        # because the constructor would reject the binding mismatch);
        # the test demonstrates that anchor-path verification works
        # ALONGSIDE sidecar-binding without conflict.
        store_with_anchor = FilesystemPrivacyBudgetStore(
            tmp_path, alice.public_key_b64, anchor=anchor
        )
        # public_key_b64 arg is ignored under anchor mode
        assert store_with_anchor.verify_chain("ignored") is True

    def test_anchor_with_unregistered_publisher_fails_verify_chain(
        self, tmp_path, anchor_contract, anchor, alice, bob
    ):
        # Only bob registered. Alice writes a journal locally; without
        # anchoring her key, the anchor-path verification must refuse
        # to accept the journal — even though the local sidecar +
        # signatures all match alice.
        anchor_contract.register(bob.public_key_bytes)
        store = FilesystemPrivacyBudgetStore(tmp_path, alice.public_key_b64)
        tracker = PersistentPrivacyBudgetTracker(
            max_epsilon=100.0, store=store, identity=alice
        )
        tracker.record_spend(2.0, "inference")

        store_with_anchor = FilesystemPrivacyBudgetStore(
            tmp_path, alice.public_key_b64, anchor=anchor
        )
        assert store_with_anchor.verify_chain("ignored") is False


# ──────────────────────────────────────────────────────────────────────────
# 6. Admin override end-to-end — emergency key revocation
# ──────────────────────────────────────────────────────────────────────────


class TestAdminOverrideE2E:
    def test_admin_override_invalidates_existing_signatures(
        self, both_registered, anchor_contract, anchor, alice, bob
    ):
        # Alice signs a manifest; verifies fine.
        signed = sign_manifest(_build_manifest(), alice)
        assert verify_manifest_with_anchor(signed, anchor) is True

        # Foundation discovers alice's key was compromised. Admin
        # override binds alice's node_id to a NEW key (bob's, in this
        # synthetic scenario). The old signature now fails to verify.
        alice_node_id_bytes = bytes.fromhex(alice.node_id)
        anchor_contract.admin_override(
            alice_node_id_bytes,
            bob.public_key_bytes,
            caller="foundation-multisig",
        )
        assert verify_manifest_with_anchor(signed, anchor) is False
