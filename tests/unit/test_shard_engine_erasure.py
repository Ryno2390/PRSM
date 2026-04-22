"""Unit tests for prsm.storage.shard_engine ERASURE mode.

Per docs/2026-04-22-phase7-storage-design-plan.md §6 Task 2.

Covers the new ShardingMode.ERASURE branch added in Task 2. The legacy
REPLICATION behaviour is covered by the existing shard_engine tests and
is only spot-checked here for regression.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from prsm.storage.blob_store import BlobStore
from prsm.storage.exceptions import ShardIntegrityError
from prsm.storage.erasure import InsufficientShardsError
from prsm.storage.models import (
    ContentHash,
    ErasureParams,
    ShardingMode,
    ShardManifest,
)
from prsm.storage.shard_engine import ShardEngine


@pytest.fixture
def blob_store(tmp_path):
    return BlobStore(data_dir=str(tmp_path / "blobs"))


@pytest.fixture
def engine(blob_store):
    return ShardEngine(blob_store)


# -----------------------------------------------------------------------------
# Replication regression — sanity
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replication_mode_default_still_works(engine):
    data = b"Tier A plaintext" * 100
    manifest = await engine.split(data, owner_node_id="node-A", replication_factor=3)
    assert manifest.sharding_mode is ShardingMode.REPLICATION
    assert manifest.erasure_params is None
    assert await engine.reassemble(manifest) == data


# -----------------------------------------------------------------------------
# Erasure — split
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_erasure_split_produces_n_shards(engine):
    data = b"Erasure-mode content" * 200
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    assert manifest.sharding_mode is ShardingMode.ERASURE
    assert manifest.erasure_params is not None
    assert manifest.erasure_params.k == 6
    assert manifest.erasure_params.n == 10
    assert len(manifest.shard_hashes) == 10


@pytest.mark.asyncio
async def test_erasure_split_records_payload_sha256(engine):
    import hashlib
    data = b"payload for hash check"
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    assert manifest.erasure_params.payload_sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.asyncio
async def test_erasure_split_custom_k_n(engine):
    data = b"custom params"
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
        erasure_k=3,
        erasure_n=5,
    )
    assert manifest.erasure_params.k == 3
    assert manifest.erasure_params.n == 5
    assert len(manifest.shard_hashes) == 5


# -----------------------------------------------------------------------------
# Erasure — reassemble
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_erasure_reassemble_from_all_shards(engine):
    data = b"The quick brown fox " * 500
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    recovered = await engine.reassemble(manifest)
    assert recovered == data


@pytest.mark.asyncio
async def test_erasure_reassemble_from_k_of_n_shards(blob_store):
    """Kill n-k shards from the blob store; reassembly must still work."""
    engine = ShardEngine(blob_store)
    data = os.urandom(4096)
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )

    # Simulate 4 shard losses (provider churn) by deleting their blob files.
    for i in (0, 3, 7, 9):
        blob_path = blob_store._path_for(manifest.shard_hashes[i])
        os.unlink(blob_path)

    recovered = await engine.reassemble(manifest)
    assert recovered == data


@pytest.mark.asyncio
async def test_erasure_reassemble_fails_below_k(blob_store):
    engine = ShardEngine(blob_store)
    data = os.urandom(1024)
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    # Kill 5 shards (one below k=6).
    for i in range(5):
        os.unlink(blob_store._path_for(manifest.shard_hashes[i]))

    with pytest.raises(InsufficientShardsError):
        await engine.reassemble(manifest)


@pytest.mark.asyncio
async def test_erasure_reassemble_rejects_missing_params(engine):
    """A manifest marked ERASURE but with no erasure_params is malformed."""
    data = b"x"
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    # Simulate corruption — strip the params.
    manifest.erasure_params = None

    with pytest.raises(ShardIntegrityError):
        await engine.reassemble(manifest)


# -----------------------------------------------------------------------------
# Manifest JSON round-trip
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_erasure_manifest_json_roundtrip(engine):
    data = b"JSON round-trip for erasure manifest" * 50
    manifest = await engine.split(
        data,
        owner_node_id="node-A",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    encoded = manifest.to_json()
    decoded = ShardManifest.from_json(encoded)

    assert decoded.sharding_mode is ShardingMode.ERASURE
    assert decoded.erasure_params is not None
    assert decoded.erasure_params.k == manifest.erasure_params.k
    assert decoded.erasure_params.n == manifest.erasure_params.n
    assert decoded.erasure_params.payload_bytes == manifest.erasure_params.payload_bytes
    assert decoded.erasure_params.payload_sha256 == manifest.erasure_params.payload_sha256


@pytest.mark.asyncio
async def test_replication_manifest_json_omits_new_fields(engine):
    """Backwards-compat: REPLICATION manifests JSON should NOT include
    erasure_params / sharding_mode so existing deserialisers keep working."""
    import json

    data = b"Tier A"
    manifest = await engine.split(
        data, owner_node_id="n", replication_factor=1
    )
    body = json.loads(manifest.to_json())
    assert "erasure_params" not in body
    assert "sharding_mode" not in body


def test_from_json_legacy_manifest_defaults_to_replication():
    """Old manifests (no sharding_mode field) must deserialise as
    REPLICATION mode with no erasure_params."""
    # Hex format: 2-char algorithm prefix (01 = SHA256) + 64 hex digest.
    hash_hex = "01" + "a" * 64
    legacy = (
        '{"content_hash":"' + hash_hex + '","shard_hashes":[],'
        '"total_size":0,"shard_size":262144,"algorithm_id":1,'
        '"created_at":0,"replication_factor":1,"owner_node_id":"n"}'
    )
    manifest = ShardManifest.from_json(legacy)
    assert manifest.sharding_mode is ShardingMode.REPLICATION
    assert manifest.erasure_params is None


# -----------------------------------------------------------------------------
# Tolerance envelope
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_erasure_tolerates_exactly_n_minus_k_losses(blob_store):
    """Plan §7 acceptance pinned: k=6,n=10 tolerates 4 lost shards."""
    engine = ShardEngine(blob_store)
    data = os.urandom(2048)
    manifest = await engine.split(
        data,
        owner_node_id="n",
        replication_factor=1,
        sharding_mode=ShardingMode.ERASURE,
    )
    # Delete exactly 4 shards.
    for i in range(4):
        os.unlink(blob_store._path_for(manifest.shard_hashes[i]))

    recovered = await engine.reassemble(manifest)
    assert recovered == data
