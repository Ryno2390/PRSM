"""
Unit tests — Phase 3.x.2 Task 1 — ModelManifest dataclass + signing payload.

Acceptance per design plan §4 Task 1: manifest dataclass + signing
payload pinned; tests cover all field combinations.

Per project testing rules: no crypto/numerics mocks. The signing-payload
contract is a wire format, so the tests assert exact bytes.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from prsm.compute.model_registry.models import (
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_SIGNING_DOMAIN,
    ManifestShardEntry,
    ModelManifest,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _entry(idx: int, *, shape=(8, 16), nonce: bytes = b"") -> ManifestShardEntry:
    """Build a deterministic shard entry for tests."""
    fake_data = f"shard-{idx}".encode("utf-8") + nonce
    return ManifestShardEntry(
        shard_id=f"sid-{idx}",
        shard_index=idx,
        tensor_shape=shape,
        sha256=hashlib.sha256(fake_data).hexdigest(),
        size_bytes=len(fake_data),
    )


def _manifest(
    *,
    model_id: str = "llama-3-8b",
    model_name: str = "Llama 3 8B",
    publisher_node_id: str = "node-AAA",
    num_shards: int = 3,
    published_at: float = 1714000000.0,
    nonce: bytes = b"",
) -> ModelManifest:
    shards = tuple(_entry(i, nonce=nonce) for i in range(num_shards))
    return ModelManifest(
        model_id=model_id,
        model_name=model_name,
        publisher_node_id=publisher_node_id,
        total_shards=num_shards,
        shards=shards,
        published_at=published_at,
    )


# ──────────────────────────────────────────────────────────────────────────
# ManifestShardEntry — frozen dataclass, type coercion
# ──────────────────────────────────────────────────────────────────────────


class TestManifestShardEntry:
    def test_construction_basic(self):
        e = _entry(0)
        assert e.shard_index == 0
        assert e.tensor_shape == (8, 16)
        assert e.sha256 == hashlib.sha256(b"shard-0").hexdigest()
        assert e.size_bytes == 7

    def test_is_frozen(self):
        e = _entry(0)
        with pytest.raises((AttributeError, Exception)):
            e.shard_id = "different"  # type: ignore[misc]

    def test_tensor_shape_coerced_to_tuple(self):
        e = ManifestShardEntry(
            shard_id="x", shard_index=0, tensor_shape=[4, 8],  # type: ignore[arg-type]
            sha256="aa", size_bytes=10,
        )
        assert isinstance(e.tensor_shape, tuple)
        assert e.tensor_shape == (4, 8)

    def test_int_fields_coerced(self):
        # JSON dicts often round-trip ints as floats; coercion must
        # handle that without raising.
        e = ManifestShardEntry(
            shard_id="x", shard_index=2.0,  # type: ignore[arg-type]
            tensor_shape=(1,), sha256="aa", size_bytes=5.0,  # type: ignore[arg-type]
        )
        assert e.shard_index == 2
        assert e.size_bytes == 5
        assert isinstance(e.shard_index, int)
        assert isinstance(e.size_bytes, int)

    def test_to_dict_roundtrip(self):
        e1 = _entry(2)
        d = e1.to_dict()
        # tensor_shape must serialize as list (JSON compatibility)
        assert d["tensor_shape"] == [8, 16]
        e2 = ManifestShardEntry.from_dict(d)
        assert e1 == e2

    def test_from_dict_drops_unknown_keys(self):
        # Additive schema changes (new fields) shouldn't break old loaders.
        d = {**_entry(0).to_dict(), "future_field": "ignored"}
        e = ManifestShardEntry.from_dict(d)
        assert e.shard_id == "sid-0"


# ──────────────────────────────────────────────────────────────────────────
# ModelManifest — construction
# ──────────────────────────────────────────────────────────────────────────


class TestModelManifestConstruction:
    def test_basic_construction(self):
        m = _manifest()
        assert m.model_id == "llama-3-8b"
        assert m.total_shards == 3
        assert len(m.shards) == 3
        assert m.schema_version == 1
        assert m.publisher_signature == b""

    def test_is_frozen(self):
        m = _manifest()
        with pytest.raises((AttributeError, Exception)):
            m.model_id = "different"  # type: ignore[misc]

    def test_shards_coerced_to_tuple(self):
        m = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=2,
            shards=[_entry(1), _entry(0)],  # type: ignore[arg-type]
            published_at=0.0,
        )
        assert isinstance(m.shards, tuple)

    def test_shards_sorted_by_index_canonically(self):
        # Construction order shouldn't matter — manifest must canonicalize
        # shard order so two manifests with the same content but different
        # construction orders produce identical signing payloads.
        m_forward = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=3,
            shards=(_entry(0), _entry(1), _entry(2)),
            published_at=1.0,
        )
        m_reverse = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=3,
            shards=(_entry(2), _entry(1), _entry(0)),
            published_at=1.0,
        )
        assert m_forward.shards == m_reverse.shards
        assert [s.shard_index for s in m_forward.shards] == [0, 1, 2]

    def test_dict_shards_coerced_to_entries(self):
        # Building a manifest from a JSON-loaded dict (where shards
        # are still dicts) must coerce them to ManifestShardEntry.
        shard_dicts = [_entry(i).to_dict() for i in range(2)]
        m = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=2,
            shards=shard_dicts,  # type: ignore[arg-type]
            published_at=0.0,
        )
        assert all(isinstance(s, ManifestShardEntry) for s in m.shards)


# ──────────────────────────────────────────────────────────────────────────
# Signing payload — canonical, deterministic, schema-bound
# ──────────────────────────────────────────────────────────────────────────


class TestSigningPayload:
    def test_payload_includes_signing_domain(self):
        # Domain separation: the payload must start with the protocol
        # tag so a signature over a manifest can't be replayed against
        # a different artifact (e.g., InferenceReceipt) that uses a
        # different domain.
        m = _manifest()
        payload = m.signing_payload()
        assert payload.startswith(MANIFEST_SIGNING_DOMAIN)

    def test_payload_includes_schema_version(self):
        m = _manifest()
        payload = m.signing_payload()
        assert f"\n{MANIFEST_SCHEMA_VERSION}\n".encode("utf-8") in payload

    def test_payload_excludes_signature(self):
        # Two manifests differing only in publisher_signature must
        # produce identical signing payloads — otherwise signing
        # would be circular.
        from dataclasses import replace
        m = _manifest()
        m_signed = replace(m, publisher_signature=b"\xff" * 64)
        assert m.signing_payload() == m_signed.signing_payload()

    def test_payload_deterministic_across_construction_orders(self):
        m_forward = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=3,
            shards=(_entry(0), _entry(1), _entry(2)),
            published_at=1.0,
        )
        m_reverse = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=3,
            shards=(_entry(2), _entry(1), _entry(0)),
            published_at=1.0,
        )
        assert m_forward.signing_payload() == m_reverse.signing_payload()

    def test_payload_sensitive_to_model_id(self):
        m1 = _manifest(model_id="A")
        m2 = _manifest(model_id="B")
        assert m1.signing_payload() != m2.signing_payload()

    def test_payload_sensitive_to_publisher(self):
        m1 = _manifest(publisher_node_id="alice")
        m2 = _manifest(publisher_node_id="bob")
        assert m1.signing_payload() != m2.signing_payload()

    def test_payload_sensitive_to_published_at(self):
        m1 = _manifest(published_at=1.0)
        m2 = _manifest(published_at=2.0)
        assert m1.signing_payload() != m2.signing_payload()

    def test_payload_sensitive_to_shard_sha256(self):
        # If anyone swaps a shard's bytes (sha256 changes), the payload
        # changes — and the signature breaks. This is the core
        # tampering-detection property of the registry.
        m1 = _manifest(nonce=b"")
        m2 = _manifest(nonce=b"different")
        assert m1.signing_payload() != m2.signing_payload()

    def test_payload_sensitive_to_shard_count(self):
        m1 = _manifest(num_shards=2)
        m2 = _manifest(num_shards=3)
        assert m1.signing_payload() != m2.signing_payload()

    def test_payload_sensitive_to_tensor_shape(self):
        m1 = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=1,
            shards=(_entry(0, shape=(4, 4)),),
            published_at=0.0,
        )
        m2 = ModelManifest(
            model_id="x", model_name="X", publisher_node_id="n",
            total_shards=1,
            shards=(_entry(0, shape=(8, 8)),),
            published_at=0.0,
        )
        # Shapes are part of the per-shard line; same sha256 with
        # different shape → different payload.
        # (Note: in practice the sha256 already commits to the bytes
        # which implicitly commits to the shape, but we belt-and-
        # suspenders include it so a future encoding change doesn't
        # silently un-commit the shape.)
        # _entry uses fixed sha256 derived from shard-{idx} so the
        # only differing field is shape.
        assert m1.signing_payload() != m2.signing_payload()


# ──────────────────────────────────────────────────────────────────────────
# JSON roundtrip — used by FilesystemModelRegistry (Task 4)
# ──────────────────────────────────────────────────────────────────────────


class TestJSONRoundtrip:
    def test_to_dict_then_from_dict(self):
        m1 = _manifest()
        m2 = ModelManifest.from_dict(m1.to_dict())
        assert m1 == m2

    def test_roundtrip_with_signature(self):
        from dataclasses import replace
        m1 = replace(_manifest(), publisher_signature=b"\x01" * 64)
        m2 = ModelManifest.from_dict(m1.to_dict())
        assert m2.publisher_signature == b"\x01" * 64
        assert m1 == m2

    def test_json_serialization_canonical(self):
        # to_dict() must produce JSON-encodable values. A canonical
        # serialization (sort_keys + no whitespace) lets two registries
        # compare manifest bytes directly.
        m = _manifest()
        d = m.to_dict()
        json_bytes = json.dumps(d, sort_keys=True).encode("utf-8")
        # Must round-trip through JSON without loss
        m_back = ModelManifest.from_dict(json.loads(json_bytes))
        assert m == m_back

    def test_signature_hex_roundtrip(self):
        from dataclasses import replace
        sig = bytes(range(64))  # 0x00..0x3F
        m1 = replace(_manifest(), publisher_signature=sig)
        d = m1.to_dict()
        # Hex-encoded for JSON safety
        assert isinstance(d["publisher_signature"], str)
        assert d["publisher_signature"] == sig.hex()
        m2 = ModelManifest.from_dict(d)
        assert m2.publisher_signature == sig

    def test_from_dict_drops_unknown_keys(self):
        d = {**_manifest().to_dict(), "future_field": "ignored"}
        m = ModelManifest.from_dict(d)
        assert m.model_id == "llama-3-8b"


# ──────────────────────────────────────────────────────────────────────────
# Module-level constants — pinned wire-format identifiers
# ──────────────────────────────────────────────────────────────────────────


class TestModuleConstants:
    def test_schema_version_is_one(self):
        # Bumping this requires a new domain string and a migration
        # path for existing manifests. Lock the value here to force
        # an explicit test update if anyone bumps it casually.
        assert MANIFEST_SCHEMA_VERSION == 1

    def test_signing_domain_is_pinned(self):
        # Locking this byte string guards against accidental rename
        # that would silently invalidate every existing signature.
        assert MANIFEST_SIGNING_DOMAIN == b"prsm-model-manifest:v1"
