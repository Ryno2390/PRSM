"""
PRSM-PROV-1 Item 3 Task 6 — ContentUploader auto-registration.

Covers ContentUploader._register_local_embedding directly:
  - Skip conditions (no index / no model_id / no provenance_hash)
  - Successful registration produces a Ed25519-verifiable record
  - Round-trip through LocalEmbeddingIndex preserves fields
  - Registration error does NOT raise out of the upload-critical path
  - Vector bytes match the dimension and are byte-exact float32
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from prsm.network.embedding_dht.local_index import (
    LocalEmbeddingIndex,
)
from prsm.network.embedding_dht.protocol import canonical_signing_payload
from prsm.node.content_uploader import ContentUploader
from prsm.node.identity import generate_node_identity


# ---- helpers --------------------------------------------------------


def _make_uploader(
    *,
    embedding_index=None,
    embedding_model_id=None,
):
    """Build a ContentUploader with the minimum fixtures needed to
    exercise the registration path. Other dependencies are stubbed —
    we only test _register_local_embedding here."""
    identity = generate_node_identity("test-node")
    gossip = MagicMock()
    ledger = MagicMock()
    uploader = ContentUploader(
        identity=identity,
        gossip=gossip,
        ledger=ledger,
        embedding_model_id=embedding_model_id,
        embedding_index=embedding_index,
    )
    return uploader, identity


def _make_local_index(tmp_path: Path) -> LocalEmbeddingIndex:
    p = tmp_path / "idx"
    p.mkdir(parents=True, exist_ok=True)
    return LocalEmbeddingIndex(p)


# ---- skip conditions ------------------------------------------------


def test_skips_when_no_embedding_index(tmp_path):
    uploader, _ = _make_uploader(
        embedding_index=None,
        embedding_model_id="m1",
    )
    # Should be a clean no-op — no exception, no side effects.
    uploader._register_local_embedding(
        np.array([1.0, 0.0], dtype=np.float32),
        provenance_hash_hex="0xdead",
    )


def test_skips_when_no_model_id(tmp_path):
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id=None,
    )
    uploader._register_local_embedding(
        np.array([1.0, 0.0], dtype=np.float32),
        provenance_hash_hex="0xdead",
    )
    # Index should be empty: no record was registered.
    assert idx.list_keys() == []


def test_skips_when_no_provenance_hash(tmp_path):
    """No on-chain anchor → peers cannot verify our signature → MUST
    NOT register the embedding for cross-node serving."""
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    uploader._register_local_embedding(
        np.array([1.0, 0.0], dtype=np.float32),
        provenance_hash_hex=None,
    )
    assert idx.list_keys() == []


def test_skips_when_provenance_hash_empty_string(tmp_path):
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    uploader._register_local_embedding(
        np.array([1.0, 0.0], dtype=np.float32),
        provenance_hash_hex="",
    )
    assert idx.list_keys() == []


# ---- successful registration ----------------------------------------


def test_registration_persists_record(tmp_path):
    idx = _make_local_index(tmp_path)
    uploader, identity = _make_uploader(
        embedding_index=idx,
        embedding_model_id="sentence-transformers/all-MiniLM-L6-v2",
    )
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    uploader._register_local_embedding(vec, provenance_hash_hex="0xabc")

    fetched = idx.lookup("0xabc", "sentence-transformers/all-MiniLM-L6-v2")
    assert fetched is not None
    assert fetched.creator_id == identity.node_id
    assert fetched.dimension == 4
    assert fetched.dtype == "float32"


def test_registered_signature_verifies_with_real_ed25519(tmp_path):
    """The signature MUST verify against the canonical signing
    payload using the creator's Ed25519 public key. This is the
    core invariant T3.6 establishes — peers fetching this record
    will run exactly this check via EmbeddingDHTClient."""
    idx = _make_local_index(tmp_path)
    uploader, identity = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    vec = np.array([0.6, -0.8], dtype=np.float32)
    uploader._register_local_embedding(vec, provenance_hash_hex="0xfeed")

    record = idx.lookup("0xfeed", "m1")
    assert record is not None

    vector_bytes = base64.b64decode(record.vector_b64, validate=True)
    payload = canonical_signing_payload(
        content_hash=record.content_hash,
        model_id=record.model_id,
        dimension=record.dimension,
        dtype=record.dtype,
        vector_bytes=vector_bytes,
        created_at=record.created_at,
    )
    pubkey = Ed25519PublicKey.from_public_bytes(identity.public_key_bytes)
    sig_bytes = base64.b64decode(record.signature_b64, validate=True)
    # Raises InvalidSignature if not valid — clean pass means OK.
    pubkey.verify(sig_bytes, payload)


def test_vector_bytes_are_exact_float32(tmp_path):
    """The vector that comes back must be byte-exactly the same
    float32 sequence we put in. Defends against float64 leakage."""
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    # Use values that would round differently in float64 if leaked.
    vec = np.array([0.123456789, -0.987654321, 1e-20, 1e20], dtype=np.float32)
    uploader._register_local_embedding(vec, provenance_hash_hex="0x1")

    record = idx.lookup("0x1", "m1")
    decoded = np.frombuffer(
        base64.b64decode(record.vector_b64, validate=True), dtype=np.float32,
    )
    np.testing.assert_array_equal(decoded, vec)


def test_idempotent_re_register_replaces_record(tmp_path):
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    uploader._register_local_embedding(v1, provenance_hash_hex="0x1")
    uploader._register_local_embedding(v2, provenance_hash_hex="0x1")

    record = idx.lookup("0x1", "m1")
    decoded = np.frombuffer(
        base64.b64decode(record.vector_b64, validate=True), dtype=np.float32,
    )
    np.testing.assert_array_equal(decoded, v2)


# ---- robustness ------------------------------------------------------


def test_registration_error_does_not_raise(tmp_path):
    """If the underlying index.register raises (disk full, corrupt,
    whatever), we MUST NOT propagate — the upload critical path is
    not allowed to fail because cross-node serve setup hit an issue."""
    idx = MagicMock(spec=["register"])
    idx.register.side_effect = OSError("disk full")
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    # Must not raise.
    uploader._register_local_embedding(
        np.array([1.0, 0.0], dtype=np.float32),
        provenance_hash_hex="0xfeed",
    )
    idx.register.assert_called_once()


def test_malformed_embedding_shape_skipped(tmp_path):
    """A 2D embedding (caller bug) must be skipped without crashing
    the upload path — log + skip, never raise."""
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    # 2D array — would normally indicate caller bug. Graceful skip.
    bad = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    uploader._register_local_embedding(bad, provenance_hash_hex="0xfeed")
    assert idx.list_keys() == []


def test_zero_dimension_embedding_skipped(tmp_path):
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="m1",
    )
    uploader._register_local_embedding(
        np.array([], dtype=np.float32),
        provenance_hash_hex="0xfeed",
    )
    assert idx.list_keys() == []


# ---- cross-model partition contract ---------------------------------


def test_registration_uses_configured_model_id(tmp_path):
    """The model_id stored in the record MUST be the one configured
    on the uploader, not anything inferred from the vector itself —
    cross-model partition relies on the uploader being authoritative."""
    idx = _make_local_index(tmp_path)
    uploader, _ = _make_uploader(
        embedding_index=idx,
        embedding_model_id="openai/text-embedding-3-small",
    )
    uploader._register_local_embedding(
        np.array([1.0] + [0.0] * 1535, dtype=np.float32),
        provenance_hash_hex="0xfeed",
    )
    record = idx.lookup("0xfeed", "openai/text-embedding-3-small")
    assert record is not None
    assert record.model_id == "openai/text-embedding-3-small"
    # Same content_hash under a DIFFERENT model_id = miss.
    assert idx.lookup("0xfeed", "some/other-model") is None
