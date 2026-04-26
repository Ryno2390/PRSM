"""
Unit tests — Phase 3.x.2 Task 3 — ModelRegistry ABC + InMemoryModelRegistry.

Acceptance per design plan §4 Task 3: drop-in replacement for the
Phase 3.x.1 dict-based pattern; the existing test_supported_models_lists_registry
shape passes against InMemoryModelRegistry.

Real ShardedModel + ModelShard with deterministic numpy bytes; real
NodeIdentity Ed25519 signing — no crypto/numerics mocks.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from prsm.compute.model_registry import (
    InMemoryModelRegistry,
    ManifestVerificationError,
    ModelAlreadyRegisteredError,
    ModelNotFoundError,
    ModelRegistryError,
    manifest_from_model,
    sign_manifest,
)
from prsm.compute.model_registry.registry import _hash_shard
from prsm.compute.model_sharding.models import ModelShard, ShardedModel
from prsm.node.identity import NodeIdentity, generate_node_identity


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — real ShardedModel + real NodeIdentity
# ──────────────────────────────────────────────────────────────────────────


def _make_shard(model_id: str, index: int, total: int, rows: int = 4, cols: int = 8) -> ModelShard:
    rng = np.random.default_rng(seed=2000 + index)
    tensor = rng.standard_normal(size=(rows, cols))
    data = tensor.tobytes()
    return ModelShard(
        shard_id=f"{model_id}-shard-{index}",
        model_id=model_id,
        shard_index=index,
        total_shards=total,
        tensor_data=data,
        tensor_shape=(rows, cols),
        layer_range=(0, 0),
        size_bytes=len(data),
        checksum=hashlib.sha256(data).hexdigest(),
    )


def _make_model(model_id: str = "test-llama", num_shards: int = 3) -> ShardedModel:
    shards = [_make_shard(model_id, i, num_shards) for i in range(num_shards)]
    return ShardedModel(
        model_id=model_id,
        model_name=f"{model_id} display",
        total_shards=num_shards,
        shards=shards,
    )


@pytest.fixture
def identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.2-task3-publisher")


@pytest.fixture
def other_identity() -> NodeIdentity:
    return generate_node_identity(display_name="phase3.x.2-task3-impostor")


@pytest.fixture
def registry() -> InMemoryModelRegistry:
    return InMemoryModelRegistry()


@pytest.fixture
def model() -> ShardedModel:
    return _make_model()


# ──────────────────────────────────────────────────────────────────────────
# Helper — manifest_from_model
# ──────────────────────────────────────────────────────────────────────────


class TestManifestFromModel:
    def test_builds_entry_per_shard(self, model):
        m = manifest_from_model(model, publisher_node_id="pub")
        assert len(m.shards) == model.total_shards
        assert m.total_shards == model.total_shards

    def test_sha256_commits_to_actual_bytes(self, model):
        m = manifest_from_model(model, publisher_node_id="pub")
        for shard, entry in zip(
            sorted(model.shards, key=lambda s: s.shard_index), m.shards
        ):
            assert entry.sha256 == hashlib.sha256(shard.tensor_data).hexdigest()
            assert entry.size_bytes == len(shard.tensor_data)

    def test_unsigned_by_default(self, model):
        m = manifest_from_model(model, publisher_node_id="pub")
        assert m.publisher_signature == b""

    def test_publisher_node_id_stamped(self, model):
        m = manifest_from_model(model, publisher_node_id="alice-node")
        assert m.publisher_node_id == "alice-node"

    def test_explicit_published_at_used(self, model):
        m = manifest_from_model(model, publisher_node_id="pub", published_at=42.0)
        assert m.published_at == 42.0


# ──────────────────────────────────────────────────────────────────────────
# register — write path
# ──────────────────────────────────────────────────────────────────────────


class TestRegister:
    def test_basic_register_returns_signed_manifest(self, registry, model, identity):
        manifest = registry.register(model, identity=identity)
        assert manifest.model_id == model.model_id
        assert manifest.publisher_node_id == identity.node_id
        assert len(manifest.publisher_signature) == 64

    def test_register_stores_model(self, registry, model, identity):
        registry.register(model, identity=identity)
        assert model.model_id in registry.list_models()

    def test_duplicate_register_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        with pytest.raises(ModelAlreadyRegisteredError, match=model.model_id):
            registry.register(model, identity=identity)

    def test_duplicate_register_blocks_different_publisher(
        self, registry, model, identity, other_identity
    ):
        # First-write-wins per node — even a different publisher can't
        # overwrite the existing registration in this registry.
        registry.register(model, identity=identity)
        with pytest.raises(ModelAlreadyRegisteredError):
            registry.register(model, identity=other_identity)

    def test_multiple_models_can_register(self, registry, identity):
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        registry.register(m1, identity=identity)
        registry.register(m2, identity=identity)
        assert sorted(registry.list_models()) == ["m1", "m2"]

    def test_register_signs_under_supplied_identity(
        self, registry, model, identity, other_identity
    ):
        # Two different identities → distinct signatures
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        s1 = registry.register(m1, identity=identity)
        s2 = registry.register(m2, identity=other_identity)
        assert s1.publisher_node_id != s2.publisher_node_id


# ──────────────────────────────────────────────────────────────────────────
# get — read path with verification
# ──────────────────────────────────────────────────────────────────────────


class TestGet:
    def test_get_returns_same_model(self, registry, model, identity):
        registry.register(model, identity=identity)
        retrieved = registry.get(model.model_id)
        # Same shard contents (object equality not required; byte-equality is).
        assert retrieved.model_id == model.model_id
        assert len(retrieved.shards) == len(model.shards)
        for s_in, s_out in zip(
            sorted(model.shards, key=lambda s: s.shard_index),
            sorted(retrieved.shards, key=lambda s: s.shard_index),
        ):
            assert s_in.tensor_data == s_out.tensor_data

    def test_get_unknown_raises_not_found(self, registry):
        with pytest.raises(ModelNotFoundError, match="not-registered"):
            registry.get("not-registered")

    def test_get_after_signature_tamper_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        # Reach into private state to flip a byte in the stored signature
        signed = registry._manifests[model.model_id]
        bad_sig = bytes(
            [signed.publisher_signature[0] ^ 0xFF]
        ) + signed.publisher_signature[1:]
        registry._manifests[model.model_id] = dataclasses.replace(
            signed, publisher_signature=bad_sig
        )
        with pytest.raises(ManifestVerificationError, match="signature"):
            registry.get(model.model_id)

    def test_get_after_shard_byte_tamper_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        # Flip a byte in the stored shard's tensor_data
        stored = registry._models[model.model_id]
        s0 = stored.shards[0]
        s0_tampered = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=s0.shard_index, total_shards=s0.total_shards,
            tensor_data=bytes([s0.tensor_data[0] ^ 0xFF]) + s0.tensor_data[1:],
            tensor_shape=s0.tensor_shape, layer_range=s0.layer_range,
            size_bytes=s0.size_bytes, checksum=s0.checksum,
        )
        stored.shards[0] = s0_tampered
        with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
            registry.get(model.model_id)

    def test_get_after_shard_size_tamper_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        stored = registry._models[model.model_id]
        s0 = stored.shards[0]
        # Truncate tensor_data — sha256 also changes, so the sha256
        # check fires first. Construct a case where size is wrong but
        # sha256 happens to match by truncating zero-bytes — easier to
        # just check that truncation raises (whichever path).
        s0_short = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=s0.shard_index, total_shards=s0.total_shards,
            tensor_data=s0.tensor_data[:-1],
            tensor_shape=s0.tensor_shape, layer_range=s0.layer_range,
            size_bytes=s0.size_bytes, checksum=s0.checksum,
        )
        stored.shards[0] = s0_short
        with pytest.raises(ManifestVerificationError):
            registry.get(model.model_id)

    def test_get_after_shard_removal_raises(self, registry, model, identity):
        registry.register(model, identity=identity)
        stored = registry._models[model.model_id]
        # Drop a shard from the model (but not the manifest)
        del stored.shards[-1]
        with pytest.raises(ManifestVerificationError, match="shards"):
            registry.get(model.model_id)

    def test_get_after_shard_index_swap_raises(self, registry, model, identity):
        # Two shards with swapped indices but original bytes → sha256
        # check on each slot fails because slot 0 now holds shard-1
        # bytes whose sha256 doesn't match slot 0's manifest entry.
        registry.register(model, identity=identity)
        stored = registry._models[model.model_id]
        s0, s1 = stored.shards[0], stored.shards[1]
        stored.shards[0] = ModelShard(
            shard_id=s0.shard_id, model_id=s0.model_id,
            shard_index=0, total_shards=s0.total_shards,
            tensor_data=s1.tensor_data,
            tensor_shape=s1.tensor_shape, layer_range=s1.layer_range,
            size_bytes=s1.size_bytes, checksum=s1.checksum,
        )
        with pytest.raises(ManifestVerificationError):
            registry.get(model.model_id)


# ──────────────────────────────────────────────────────────────────────────
# list_models / get_manifest
# ──────────────────────────────────────────────────────────────────────────


class TestListAndGetManifest:
    def test_list_empty(self, registry):
        assert registry.list_models() == []

    def test_list_sorted(self, registry, identity):
        registry.register(_make_model("zz"), identity=identity)
        registry.register(_make_model("aa"), identity=identity)
        registry.register(_make_model("mm"), identity=identity)
        assert registry.list_models() == ["aa", "mm", "zz"]

    def test_get_manifest_returns_signed_manifest(self, registry, model, identity):
        registry.register(model, identity=identity)
        m = registry.get_manifest(model.model_id)
        assert m.model_id == model.model_id
        assert len(m.publisher_signature) == 64

    def test_get_manifest_unknown_raises(self, registry):
        with pytest.raises(ModelNotFoundError):
            registry.get_manifest("missing")


# ──────────────────────────────────────────────────────────────────────────
# verify — audit-only
# ──────────────────────────────────────────────────────────────────────────


class TestVerify:
    def test_verify_passes_for_clean_registration(self, registry, model, identity):
        registry.register(model, identity=identity)
        assert registry.verify(model.model_id) is True

    def test_verify_returns_false_for_unknown(self, registry):
        assert registry.verify("nope") is False

    def test_verify_returns_false_after_tamper(self, registry, model, identity):
        registry.register(model, identity=identity)
        registry._models[model.model_id].shards[0] = ModelShard(
            shard_id="x", model_id=model.model_id,
            shard_index=0, total_shards=model.total_shards,
            tensor_data=b"different bytes",
            tensor_shape=(1,), layer_range=(0, 0),
            size_bytes=15, checksum="",
        )
        assert registry.verify(model.model_id) is False

    def test_verify_does_not_swallow_unknown_errors(self, registry, model, identity):
        # If the registry raises something OTHER than the documented
        # exception types, verify must NOT silently turn it into False
        # — that would mask bugs.
        registry.register(model, identity=identity)

        original_get = registry.get
        def boom(model_id):
            raise RuntimeError("registry corrupted")
        registry.get = boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="corrupted"):
            registry.verify(model.model_id)
        registry.get = original_get  # type: ignore[method-assign]


# ──────────────────────────────────────────────────────────────────────────
# Drop-in for the Phase 3.x.1 dict pattern (acceptance per §4 Task 3)
# ──────────────────────────────────────────────────────────────────────────


class TestPhase3x1DropIn:
    def test_supported_models_lists_registry_shape(self, registry, identity):
        # The Phase 3.x.1 executor's TestConstruction::test_supported_models_lists_registry
        # asserts ex.supported_models() == sorted(model_ids). The registry's
        # list_models() must produce the same shape.
        registry.register(_make_model("test-llama"), identity=identity)
        assert registry.list_models() == ["test-llama"]

    def test_register_and_get_round_trip(self, registry, model, identity):
        # Mirrors the Phase 3.x.1 dict pattern: registered models
        # round-trip through get() unchanged.
        registry.register(model, identity=identity)
        out = registry.get(model.model_id)
        assert out.model_id == model.model_id
        assert out.total_shards == model.total_shards


# ──────────────────────────────────────────────────────────────────────────
# Exception hierarchy contract
# ──────────────────────────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_all_inherit_from_base(self):
        assert issubclass(ModelNotFoundError, ModelRegistryError)
        assert issubclass(ModelAlreadyRegisteredError, ModelRegistryError)
        assert issubclass(ManifestVerificationError, ModelRegistryError)

    def test_all_distinguishable(self):
        # Distinct types so callers can `except ModelNotFoundError`
        # without catching tampering errors.
        assert ModelNotFoundError is not ManifestVerificationError
        assert ModelAlreadyRegisteredError is not ManifestVerificationError


# ──────────────────────────────────────────────────────────────────────────
# Internal _hash_shard helper — pinned so future "optimization" can't
# silently desync from the manifest writer
# ──────────────────────────────────────────────────────────────────────────


class TestHashShard:
    def test_matches_sha256_of_tensor_data(self):
        shard = _make_shard("m", 0, 1)
        assert _hash_shard(shard) == hashlib.sha256(shard.tensor_data).hexdigest()
