"""
Sprint 4 Phase 4: Content Retrieval API Tests
===============================================

Tests for the request_content() cross-node download functionality
in ContentUploader, including provider discovery, inline/gateway
transfer modes, content hash verification, and error handling.
"""

import asyncio
import base64
import hashlib
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Set

from prsm.node.content_uploader import ContentUploader, UploadedContent
from prsm.node.content_index import ContentRecord


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_identity():
    """Create a mock node identity."""
    identity = MagicMock()
    identity.node_id = "node-A"
    identity.public_key_b64 = "mock-pubkey-b64"
    identity.sign = MagicMock(return_value="mock-signature")
    return identity


@pytest.fixture
def mock_gossip():
    """Create a mock gossip protocol."""
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    return gossip


@pytest.fixture
def mock_ledger():
    """Create a mock ledger."""
    ledger = MagicMock()
    ledger.credit = AsyncMock()
    return ledger


@pytest.fixture
def mock_transport():
    """Create a mock transport layer."""
    transport = MagicMock()
    transport.on_message = MagicMock()
    transport.send_to_peer = AsyncMock(return_value=True)
    transport.identity = MagicMock()
    transport.identity.node_id = "node-A"
    return transport


@pytest.fixture
def mock_content_index():
    """Create a mock content index."""
    index = MagicMock()
    return index


@pytest.fixture
def uploader(mock_identity, mock_gossip, mock_ledger, mock_transport, mock_content_index):
    """Create a ContentUploader with all mock dependencies."""
    cu = ContentUploader(
        identity=mock_identity,
        gossip=mock_gossip,
        ledger=mock_ledger,
        transport=mock_transport,
        content_index=mock_content_index,
    )
    return cu


# =============================================================================
# HELPER
# =============================================================================

def make_content_record(cid: str, providers: set, content_hash: str = "") -> ContentRecord:
    """Create a ContentRecord for testing."""
    return ContentRecord(
        cid=cid,
        filename="test.txt",
        size_bytes=100,
        content_hash=content_hash,
        creator_id="creator-node",
        providers=providers,
    )


# =============================================================================
# TEST: request_content() - provider discovery
# =============================================================================

class TestProviderDiscovery:
    """Test that request_content() correctly discovers and contacts providers."""

    @pytest.mark.asyncio
    async def test_no_transport_returns_none(self, uploader):
        """Without transport, request_content should return None."""
        uploader.transport = None
        result = await uploader.request_content("QmTestCid")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_providers_returns_none(self, uploader, mock_content_index):
        """When no providers are known, return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmTestCid", providers=set()
        )
        result = await uploader.request_content("QmTestCid")
        assert result is None

    @pytest.mark.asyncio
    async def test_self_provider_skipped(self, uploader, mock_content_index):
        """If only provider is self, skip and return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmTestCid", providers={"node-A"}  # self
        )
        result = await uploader.request_content("QmTestCid")
        assert result is None

    @pytest.mark.asyncio
    async def test_local_content_returned_without_network(self, uploader, mock_content_index):
        """If we have the content locally (uploaded_content), try IPFS cat first."""
        uploader.uploaded_content["QmLocal"] = UploadedContent(
            cid="QmLocal",
            filename="local.txt",
            size_bytes=5,
            content_hash="abc",
            creator_id="node-A",
        )
        uploader._ipfs_cat = AsyncMock(return_value=b"hello")

        result = await uploader.request_content("QmLocal")

        assert result == b"hello"
        uploader._ipfs_cat.assert_called_once_with("QmLocal")
        # Should not have contacted the content index
        mock_content_index.lookup.assert_not_called()


# =============================================================================
# TEST: Inline transfer mode
# =============================================================================

class TestInlineTransfer:
    """Test content retrieval via inline base64 transfer."""

    @pytest.mark.asyncio
    async def test_inline_content_decoded(self, uploader, mock_content_index, mock_transport):
        """Inline base64 content should be decoded and returned."""
        test_content = b"Hello, PRSM network!"
        content_hash = hashlib.sha256(test_content).hexdigest()

        mock_content_index.lookup.return_value = make_content_record(
            "QmTest", providers={"node-B"}, content_hash=content_hash
        )

        # Simulate: when send_to_peer is called, resolve the pending future
        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            # Simulate the provider responding
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "subtype": "content_response",
                    "request_id": request_id,
                    "cid": "QmTest",
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(test_content).decode(),
                    "content_hash": content_hash,
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmTest", timeout=5.0)

        assert result == test_content

    @pytest.mark.asyncio
    async def test_hash_verification_rejects_mismatch(self, uploader, mock_content_index, mock_transport):
        """Content with wrong hash should be rejected."""
        test_content = b"Tampered content"
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        mock_content_index.lookup.return_value = make_content_record(
            "QmBad", providers={"node-B"}, content_hash=wrong_hash
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(test_content).decode(),
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmBad", timeout=5.0)

        # Should return None because hash doesn't match
        assert result is None

    @pytest.mark.asyncio
    async def test_hash_verification_skipped_when_disabled(self, uploader, mock_content_index, mock_transport):
        """When verify_hash=False, content is returned even with mismatched hash."""
        test_content = b"Any content"
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

        mock_content_index.lookup.return_value = make_content_record(
            "QmAny", providers={"node-B"}, content_hash=wrong_hash
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": True,
                    "transfer_mode": "inline",
                    "data_b64": base64.b64encode(test_content).decode(),
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmAny", timeout=5.0, verify_hash=False)

        assert result == test_content


# =============================================================================
# TEST: Gateway transfer mode
# =============================================================================

class TestGatewayTransfer:
    """Test content retrieval via IPFS gateway URL."""

    @pytest.mark.asyncio
    async def test_gateway_content_fetched(self, uploader, mock_content_index, mock_transport):
        """Gateway URL content should be fetched and returned."""
        test_content = b"Large file content from gateway"
        content_hash = hashlib.sha256(test_content).hexdigest()

        mock_content_index.lookup.return_value = make_content_record(
            "QmGateway", providers={"node-B"}, content_hash=content_hash
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": True,
                    "transfer_mode": "gateway",
                    "gateway_url": "http://127.0.0.1:8080/ipfs/QmGateway",
                })
            return True

        mock_transport.send_to_peer = fake_send
        uploader._fetch_from_gateway = AsyncMock(return_value=test_content)

        result = await uploader.request_content("QmGateway", timeout=5.0)

        assert result == test_content
        uploader._fetch_from_gateway.assert_called_once_with(
            "http://127.0.0.1:8080/ipfs/QmGateway"
        )


# =============================================================================
# TEST: Timeout and error handling
# =============================================================================

class TestErrorHandling:
    """Test timeout and error scenarios in content retrieval."""

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, uploader, mock_content_index, mock_transport):
        """If provider doesn't respond within timeout, return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmSlow", providers={"node-B"}
        )

        # Don't resolve the future — simulates no response
        mock_transport.send_to_peer = AsyncMock(return_value=True)

        result = await uploader.request_content("QmSlow", timeout=0.5)

        assert result is None

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self, uploader, mock_content_index, mock_transport):
        """If provider responds with found=False, return None."""
        mock_content_index.lookup.return_value = make_content_record(
            "QmMissing", providers={"node-B"}
        )

        async def fake_send(peer_id, msg):
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                future.set_result({
                    "found": False,
                    "cid": "QmMissing",
                })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmMissing", timeout=5.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self, uploader, mock_content_index, mock_transport):
        """If first provider fails, try the next one."""
        test_content = b"From second provider"
        content_hash = hashlib.sha256(test_content).hexdigest()

        mock_content_index.lookup.return_value = make_content_record(
            "QmFallback", providers={"node-B", "node-C"}, content_hash=content_hash
        )

        call_count = 0

        async def fake_send(peer_id, msg):
            nonlocal call_count
            call_count += 1
            request_id = msg.payload.get("request_id", "")
            future = uploader._pending_requests.get(request_id)
            if future and not future.done():
                if call_count == 1:
                    # First provider says not found
                    future.set_result({"found": False})
                else:
                    # Second provider has it
                    future.set_result({
                        "found": True,
                        "transfer_mode": "inline",
                        "data_b64": base64.b64encode(test_content).decode(),
                    })
            return True

        mock_transport.send_to_peer = fake_send

        result = await uploader.request_content("QmFallback", timeout=5.0)

        assert result == test_content
        assert call_count == 2


# =============================================================================
# TEST: Response handler
# =============================================================================

class TestResponseHandler:
    """Test the _handle_content_response future resolution."""

    def test_response_resolves_pending_future(self, uploader):
        """A content_response should resolve the matching pending future."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        uploader._pending_requests["req-123"] = future

        msg = MagicMock()
        msg.payload = {
            "subtype": "content_response",
            "request_id": "req-123",
            "found": True,
            "transfer_mode": "inline",
            "data_b64": base64.b64encode(b"test").decode(),
        }

        uploader._handle_content_response(msg)

        assert future.done()
        assert future.result()["found"] is True
        loop.close()

    def test_unknown_request_id_ignored(self, uploader):
        """A response for an unknown request_id should be silently ignored."""
        msg = MagicMock()
        msg.payload = {
            "subtype": "content_response",
            "request_id": "unknown-id",
            "found": True,
        }

        # Should not raise
        uploader._handle_content_response(msg)

    def test_already_done_future_not_overwritten(self, uploader):
        """If the future is already resolved, don't overwrite it."""
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        future.set_result({"found": False})
        uploader._pending_requests["req-done"] = future

        msg = MagicMock()
        msg.payload = {
            "request_id": "req-done",
            "found": True,
        }

        uploader._handle_content_response(msg)

        # Should still have the original result
        assert future.result()["found"] is False
        loop.close()


# =============================================================================
# TEST: Semantic deduplication (_SemanticIndex + upload path wiring)
# =============================================================================

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")

from prsm.node.content_uploader import _SemanticIndex


class TestSemanticIndex:
    """Unit tests for the _SemanticIndex class."""

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_empty_index_returns_none(self):
        idx = _SemanticIndex()
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert idx.find_nearest(vec) is None

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_store_and_retrieve(self):
        idx = _SemanticIndex()
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        idx.store("QmA", vec, "creator-1")
        result = idx.find_nearest(vec)
        assert result is not None
        cid, sim, creator = result
        assert cid == "QmA"
        assert abs(sim - 1.0) < 1e-5
        assert creator == "creator-1"

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_finds_most_similar(self):
        idx = _SemanticIndex()
        idx.store("QmA", np.array([1.0, 0.0, 0.0], dtype=np.float32), "c1")
        idx.store("QmB", np.array([0.0, 1.0, 0.0], dtype=np.float32), "c2")
        # Query close to QmA
        query = np.array([0.99, 0.14, 0.0], dtype=np.float32)
        cid, sim, _ = idx.find_nearest(query)
        assert cid == "QmA"
        assert sim > 0.9

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_zero_vector_returns_none(self):
        idx = _SemanticIndex()
        idx.store("QmA", np.array([1.0, 0.0, 0.0], dtype=np.float32), "c1")
        result = idx.find_nearest(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert result is None

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "sem_idx.json"
        idx = _SemanticIndex(persist_path=path)
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        idx.store("QmPersist", vec, "creator-x")

        # Reload from disk
        idx2 = _SemanticIndex(persist_path=path)
        result = idx2.find_nearest(vec)
        assert result is not None
        assert result[0] == "QmPersist"
        assert result[2] == "creator-x"

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    def test_len(self):
        idx = _SemanticIndex()
        assert len(idx) == 0
        idx.store("QmA", np.array([1.0, 0.0], dtype=np.float32), "c1")
        assert len(idx) == 1
        idx.store("QmB", np.array([0.0, 1.0], dtype=np.float32), "c2")
        assert len(idx) == 2


class TestUploadSemanticWiring:
    """Tests for semantic deduplication wired into ContentUploader.upload()."""

    @pytest.fixture
    def uploader_with_embedding(self, mock_identity, mock_gossip, mock_ledger,
                                mock_transport, mock_content_index):
        """ContentUploader with a fake synchronous embedding_fn."""
        async def fake_embed(text: str):
            # Returns a deterministic unit vector based on text content
            vec = np.zeros(8, dtype=np.float32)
            for i, ch in enumerate(text[:8]):
                vec[i] = float(ord(ch))
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else vec

        cu = ContentUploader(
            identity=mock_identity,
            gossip=mock_gossip,
            ledger=mock_ledger,
            transport=mock_transport,
            content_index=mock_content_index,
            embedding_fn=fake_embed,
        )
        return cu

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    async def test_embedding_stored_after_upload(self, uploader_with_embedding):
        """After a successful upload, the embedding should be in the semantic index."""
        cu = uploader_with_embedding
        cu._ipfs_add = AsyncMock(return_value="QmFakeHash1")

        content = b"This is a unique research paper about quantum computing and entanglement."
        result = await cu.upload(content, filename="paper.txt")

        assert result is not None
        assert result.embedding_id == "emb:QmFakeHash1"
        assert len(cu._semantic_index) == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    async def test_near_duplicate_auto_registers_derivative(self, uploader_with_embedding):
        """A near-duplicate upload should be auto-registered with the matching CID as parent."""
        cu = uploader_with_embedding

        # First upload — establish the original
        cu._ipfs_add = AsyncMock(return_value="QmOriginal")
        original_text = b"The quick brown fox jumps over the lazy dog, a classic test sentence."
        await cu.upload(original_text, filename="original.txt")

        # Manually inject a very similar (but not identical) embedding into the index
        # to simulate a near-duplicate without relying on the fake_embed being 0.92+ similar
        original_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        duplicate_vec = np.array([0.999, 0.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Normalise and plant
        original_vec /= np.linalg.norm(original_vec)
        duplicate_vec /= np.linalg.norm(duplicate_vec)
        cu._semantic_index._index["QmSeed"] = (original_vec, "creator-original")

        # Override the embedding_fn to return the near-duplicate vector
        async def near_dup_embed(text: str):
            return duplicate_vec.copy()

        cu._embedding_fn = near_dup_embed
        cu._ipfs_add = AsyncMock(return_value="QmDerivative")

        result = await cu.upload(b"A slightly modified version of the original text, with some extra words added.", filename="dup.txt")

        assert result is not None
        assert result.near_duplicate_of == "QmSeed"
        assert result.near_duplicate_similarity is not None
        assert result.near_duplicate_similarity >= _SemanticIndex.DERIVATIVE_THRESHOLD
        # QmSeed should have been auto-added as a parent
        assert "QmSeed" in result.parent_cids

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    async def test_unrelated_upload_has_no_near_duplicate(self, uploader_with_embedding):
        """Content that is semantically unrelated should not trigger the derivative path."""
        cu = uploader_with_embedding

        # Plant an orthogonal vector in the index
        orthogonal = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        cu._semantic_index._index["QmUnrelated"] = (orthogonal, "other-creator")

        async def unrelated_embed(text: str):
            # Returns vector close to [1,0,0,...] — far from [0,0,1,...]
            v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return v

        cu._embedding_fn = unrelated_embed
        cu._ipfs_add = AsyncMock(return_value="QmNewContent")

        result = await cu.upload(b"Completely different content about economics.", filename="econ.txt")

        assert result is not None
        assert result.near_duplicate_of is None
        assert result.near_duplicate_similarity is None
        assert "QmUnrelated" not in result.parent_cids

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    async def test_no_embedding_fn_skips_semantic_check(self, mock_identity, mock_gossip,
                                                        mock_ledger, mock_transport,
                                                        mock_content_index):
        """Without an embedding_fn, upload succeeds with no semantic metadata."""
        cu = ContentUploader(
            identity=mock_identity,
            gossip=mock_gossip,
            ledger=mock_ledger,
            transport=mock_transport,
            content_index=mock_content_index,
            # No embedding_fn
        )
        cu._ipfs_add = AsyncMock(return_value="QmNoEmbed")

        result = await cu.upload(b"Some content that will not be embedded.", filename="nobed.txt")

        assert result is not None
        assert result.embedding_id is None
        assert result.near_duplicate_of is None
        assert len(cu._semantic_index) == 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")
    async def test_get_stats_includes_semantic_fields(self, uploader_with_embedding):
        """get_stats() should report semantic_index_size and embedding_fn_active."""
        cu = uploader_with_embedding
        cu._ipfs_add = AsyncMock(return_value="QmStat")

        await cu.upload(b"Long enough content to generate an embedding for stats testing.", filename="stats.txt")
        stats = cu.get_stats()

        assert "semantic_index_size" in stats
        assert stats["semantic_index_size"] >= 1
        assert stats["embedding_fn_active"] is True
        assert "derivative_works_detected" in stats


# =============================================================================
# TEST: LocalLedger provenance persistence (Task 2)
# =============================================================================

import json as _json

from prsm.node.local_ledger import LocalLedger, TransactionType


class TestLocalLedgerProvenance:
    """Tests for upsert_provenance / get_provenance / list_provenance_by_creator."""

    @pytest_asyncio.fixture
    async def ledger(self):
        """In-memory LocalLedger initialised with the new provenance_chains table."""
        l = LocalLedger(db_path=":memory:")
        await l.initialize()
        yield l
        await l.close()

    @pytest.mark.asyncio
    async def test_upsert_and_get_provenance(self, ledger):
        """upsert_provenance stores a record; get_provenance retrieves it."""
        data = {
            "cid": "QmTest001",
            "content_hash": "abc123",
            "creator_id": "creator-node-1",
            "creator_public_key": "pubkey-x",
            "filename": "paper.txt",
            "size_bytes": 1024,
            "royalty_rate": 0.05,
            "parent_cids": [],
            "signature": "sig-abc",
            "embedding_id": "emb:QmTest001",
            "near_duplicate_of": None,
            "metadata": {"topic": "science"},
        }
        await ledger.upsert_provenance(data)
        result = await ledger.get_provenance("QmTest001")

        assert result is not None
        assert result["cid"] == "QmTest001"
        assert result["creator_id"] == "creator-node-1"
        assert result["royalty_rate"] == 0.05
        assert result["embedding_id"] == "emb:QmTest001"
        assert result["metadata"] == {"topic": "science"}

    @pytest.mark.asyncio
    async def test_get_provenance_missing_returns_none(self, ledger):
        """get_provenance returns None for unknown CIDs."""
        result = await ledger.get_provenance("QmNotHere")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(self, ledger):
        """Re-upserting the same CID with updated fields overwrites, not duplicates."""
        base = {"cid": "QmIdempotent", "creator_id": "alice", "royalty_rate": 0.01}
        await ledger.upsert_provenance(base)
        await ledger.upsert_provenance({**base, "royalty_rate": 0.05})

        result = await ledger.get_provenance("QmIdempotent")
        assert result["royalty_rate"] == 0.05

    @pytest.mark.asyncio
    async def test_list_provenance_by_creator(self, ledger):
        """list_provenance_by_creator returns all records for a creator."""
        for i in range(3):
            await ledger.upsert_provenance({
                "cid": f"QmCreator{i}",
                "creator_id": "bob",
                "royalty_rate": 0.01,
            })
        # Different creator — should not appear
        await ledger.upsert_provenance({"cid": "QmOther", "creator_id": "carol"})

        records = await ledger.list_provenance_by_creator("bob")
        assert len(records) == 3
        cids = {r["cid"] for r in records}
        assert cids == {"QmCreator0", "QmCreator1", "QmCreator2"}

    @pytest.mark.asyncio
    async def test_list_provenance_empty_creator(self, ledger):
        """Unknown creator returns an empty list."""
        records = await ledger.list_provenance_by_creator("nobody")
        assert records == []

    @pytest.mark.asyncio
    async def test_upsert_missing_cid_is_ignored(self, ledger):
        """upsert_provenance silently ignores data with no cid."""
        await ledger.upsert_provenance({"creator_id": "ghost"})
        # No error raised; nothing stored (can't meaningfully verify without CID)


# =============================================================================
# TEST: ContentIndex provenance gossip handlers (Task 2)
# =============================================================================

from prsm.node.content_index import ContentIndex
from prsm.node.gossip import (
    GOSSIP_PROVENANCE_QUERY,
    GOSSIP_PROVENANCE_REGISTER,
    GOSSIP_PROVENANCE_RESPONSE,
)


class TestContentIndexProvenance:
    """Unit tests for the provenance gossip handlers in ContentIndex."""

    @pytest.fixture
    def mock_ledger_for_index(self):
        ledger = MagicMock()
        ledger.upsert_provenance = AsyncMock()
        ledger.get_provenance = AsyncMock(return_value=None)
        return ledger

    @pytest.fixture
    def index(self, mock_gossip, mock_ledger_for_index):
        ci = ContentIndex(gossip=mock_gossip, ledger=mock_ledger_for_index)
        return ci

    @pytest.mark.asyncio
    async def test_on_provenance_register_calls_upsert(self, index, mock_ledger_for_index):
        """_on_provenance_register persists the data to the ledger."""
        data = {"cid": "QmReg1", "creator_id": "alice", "royalty_rate": 0.02}
        await index._on_provenance_register(GOSSIP_PROVENANCE_REGISTER, data, "peer-1")
        mock_ledger_for_index.upsert_provenance.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_on_provenance_register_no_ledger_is_noop(self, mock_gossip):
        """With no ledger, _on_provenance_register does nothing (no crash)."""
        ci = ContentIndex(gossip=mock_gossip, ledger=None)
        data = {"cid": "QmReg2", "creator_id": "alice"}
        await ci._on_provenance_register(GOSSIP_PROVENANCE_REGISTER, data, "peer-1")
        # no assertion needed — just confirm no exception

    @pytest.mark.asyncio
    async def test_on_provenance_register_missing_cid_is_noop(self, index, mock_ledger_for_index):
        """Data without a cid should not call upsert."""
        await index._on_provenance_register(GOSSIP_PROVENANCE_REGISTER, {"creator_id": "x"}, "p")
        mock_ledger_for_index.upsert_provenance.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_provenance_query_responds_when_known(self, index, mock_gossip, mock_ledger_for_index):
        """When the ledger has the record, _on_provenance_query publishes a response."""
        prov = {"cid": "QmKnown", "creator_id": "bob"}
        mock_ledger_for_index.get_provenance = AsyncMock(return_value=prov)

        await index._on_provenance_query(
            GOSSIP_PROVENANCE_QUERY,
            {"cid": "QmKnown", "requester_id": "req-node"},
            "req-node",
        )

        mock_gossip.publish.assert_called_once()
        call_args = mock_gossip.publish.call_args
        assert call_args[0][0] == GOSSIP_PROVENANCE_RESPONSE
        payload = call_args[0][1]
        assert payload["cid"] == "QmKnown"
        assert payload["provenance"] == prov

    @pytest.mark.asyncio
    async def test_on_provenance_query_silent_when_unknown(self, index, mock_gossip, mock_ledger_for_index):
        """When the ledger doesn't have the record, no response is published."""
        mock_ledger_for_index.get_provenance = AsyncMock(return_value=None)

        await index._on_provenance_query(
            GOSSIP_PROVENANCE_QUERY,
            {"cid": "QmUnknown", "requester_id": "req-node"},
            "req-node",
        )

        mock_gossip.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_provenance_response_persists_and_resolves_future(self, index, mock_ledger_for_index):
        """_on_provenance_response caches to ledger and resolves a pending future."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        index._pending_provenance["QmResp"] = future

        prov = {"cid": "QmResp", "creator_id": "charlie"}
        await index._on_provenance_response(
            GOSSIP_PROVENANCE_RESPONSE,
            {"cid": "QmResp", "provenance": prov},
            "peer-x",
        )

        mock_ledger_for_index.upsert_provenance.assert_called_once_with(prov)
        assert future.done()
        assert future.result() == prov

    @pytest.mark.asyncio
    async def test_get_provenance_local_ledger_hit(self, index, mock_ledger_for_index):
        """get_provenance returns from ledger immediately without broadcasting."""
        prov = {"cid": "QmLocal", "creator_id": "dave"}
        mock_ledger_for_index.get_provenance = AsyncMock(return_value=prov)

        result = await index.get_provenance("QmLocal")
        assert result == prov
        mock_gossip_obj = index.gossip
        mock_gossip_obj.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_provenance_cross_node_timeout(self, index, mock_ledger_for_index):
        """When ledger misses and no peer responds, get_provenance times out and returns None."""
        mock_ledger_for_index.get_provenance = AsyncMock(return_value=None)

        result = await index.get_provenance("QmMissing", timeout=0.05)
        assert result is None


# =============================================================================
# TEST: ProvenanceVectorIntegration FTNS royalty crediting (Task 2)
# =============================================================================

from prsm.data.vector_store.provenance import ProvenanceVectorIntegration
from prsm.data.vector_store.base import ContentMatch, ContentType


class TestProvenanceVectorRoyalties:
    """Tests for FTNS royalty crediting in ProvenanceVectorIntegration."""

    def make_match(
        self,
        cid: str = "QmMatch1",
        creator_id: str = "creator-A",
        similarity: float = 0.90,
        royalty_rate: float = 0.10,
    ) -> ContentMatch:
        return ContentMatch(
            content_cid=cid,
            similarity_score=similarity,
            creator_id=creator_id,
            royalty_rate=royalty_rate,
            content_type=ContentType.TEXT,
        )

    @pytest.fixture
    def mock_vector_store(self):
        vs = MagicMock()
        vs.search_similar_content = AsyncMock(return_value=[])
        return vs

    @pytest.fixture
    def mock_royalty_ledger(self):
        ledger = MagicMock()
        ledger.credit = AsyncMock()
        return ledger

    @pytest.mark.asyncio
    async def test_royalty_credited_to_creator(self, mock_vector_store, mock_royalty_ledger):
        """credit() is called with the right wallet and amount."""
        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=mock_royalty_ledger,
            node_id="querying-node",
        )
        match = self.make_match(similarity=0.80, royalty_rate=0.10)
        await pvi._track_content_access([match], user_id="user-X")

        mock_royalty_ledger.credit.assert_called_once()
        call_kwargs = mock_royalty_ledger.credit.call_args[1]
        assert call_kwargs["wallet_id"] == "creator-A"
        expected_royalty = round(0.80 * 0.10, 6)
        assert abs(call_kwargs["amount"] - expected_royalty) < 1e-9

    @pytest.mark.asyncio
    async def test_no_self_royalty(self, mock_vector_store, mock_royalty_ledger):
        """No credit when creator_id equals the querying node_id."""
        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=mock_royalty_ledger,
            node_id="creator-A",  # same as creator
        )
        match = self.make_match(creator_id="creator-A")
        await pvi._track_content_access([match], user_id="some-user")
        mock_royalty_ledger.credit.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_self_royalty_user(self, mock_vector_store, mock_royalty_ledger):
        """No credit when creator_id equals the querying user_id."""
        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=mock_royalty_ledger,
            node_id="other-node",
        )
        match = self.make_match(creator_id="user-X")
        await pvi._track_content_access([match], user_id="user-X")
        mock_royalty_ledger.credit.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_royalty_rate_skipped(self, mock_vector_store, mock_royalty_ledger):
        """Content with royalty_rate=0 generates no credit call."""
        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=mock_royalty_ledger,
            node_id="node-x",
        )
        match = self.make_match(royalty_rate=0.0)
        await pvi._track_content_access([match], user_id="user-Y")
        mock_royalty_ledger.credit.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_ledger_still_logs_usage(self, mock_vector_store):
        """Without a ledger, usage is still recorded in memory (no crash)."""
        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=None,
            node_id="node-x",
        )
        match = self.make_match()
        await pvi._track_content_access([match], user_id="user-Z")
        assert len(pvi.usage_log) == 1
        assert pvi.usage_log[0]["content_cid"] == "QmMatch1"

    @pytest.mark.asyncio
    async def test_multiple_matches_all_credited(self, mock_vector_store, mock_royalty_ledger):
        """Each match in the result list gets its own credit call."""
        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=mock_royalty_ledger,
            node_id="node-x",
        )
        matches = [
            self.make_match(cid="QmA", creator_id="alice", similarity=0.95, royalty_rate=0.05),
            self.make_match(cid="QmB", creator_id="bob", similarity=0.80, royalty_rate=0.10),
        ]
        await pvi._track_content_access(matches, user_id="user-Z")
        assert mock_royalty_ledger.credit.call_count == 2

    @pytest.mark.asyncio
    async def test_search_with_royalty_tracking_integration(self, mock_vector_store, mock_royalty_ledger):
        """search_with_royalty_tracking() performs search and credits royalties."""
        match = self.make_match(similarity=0.85, royalty_rate=0.10)
        mock_vector_store.search_similar_content = AsyncMock(return_value=[match])

        pvi = ProvenanceVectorIntegration(
            vector_store=mock_vector_store,
            ledger=mock_royalty_ledger,
            node_id="node-x",
        )
        import numpy as np
        query = np.zeros(8, dtype=np.float32)
        results = await pvi.search_with_royalty_tracking(query, user_id="user-Z")

        assert len(results) == 1
        mock_royalty_ledger.credit.assert_called_once()


# =============================================================================
# TEST: ContentVerificationSystem._load_provenance_chain (Task 2)
# =============================================================================

from prsm.data.ipfs.content_verification import ContentVerificationSystem, create_verification_system


class TestContentVerificationProvenance:
    """Tests for _load_provenance_chain() with injected provenance_lookup_fn."""

    @pytest.fixture
    def mock_ipfs_client(self):
        client = MagicMock()
        client.get_content = AsyncMock(return_value=b"data")
        client.add_content = AsyncMock(return_value="QmFakeIPFS")
        return client

    @pytest.mark.asyncio
    async def test_no_lookup_fn_returns_none(self, mock_ipfs_client):
        """Without a provenance_lookup_fn, _load_provenance_chain returns None."""
        cvs = ContentVerificationSystem(mock_ipfs_client, provenance_lookup_fn=None)
        result = await cvs._load_provenance_chain("QmAny")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_fn_called_and_chain_built(self, mock_ipfs_client):
        """When lookup_fn returns data, a ProvenanceChain is synthesised and cached."""
        import time as _time
        prov_data = {
            "cid": "QmChain1",
            "creator_id": "researcher-A",
            "content_hash": "abc",
            "filename": "study.txt",
            "parent_cids": [],
            "signature": "sig-x",
            "registered_at": _time.time(),
        }

        async def lookup(cid):
            return prov_data if cid == "QmChain1" else None

        cvs = ContentVerificationSystem(mock_ipfs_client, provenance_lookup_fn=lookup)
        chain = await cvs._load_provenance_chain("QmChain1")

        assert chain is not None
        assert chain.content_cid == "QmChain1"
        assert len(chain.events) == 1
        assert chain.events[0].actor_id == "researcher-A"
        # Should be cached for subsequent calls
        assert "QmChain1" in cvs.provenance_chains

    @pytest.mark.asyncio
    async def test_lookup_fn_returning_none_gives_none(self, mock_ipfs_client):
        """When lookup_fn returns None (unknown CID), chain is None."""
        async def lookup(cid):
            return None

        cvs = ContentVerificationSystem(mock_ipfs_client, provenance_lookup_fn=lookup)
        result = await cvs._load_provenance_chain("QmGhost")
        assert result is None

    @pytest.mark.asyncio
    async def test_lookup_fn_exception_gives_none(self, mock_ipfs_client):
        """Exceptions from lookup_fn are caught; None is returned gracefully."""
        async def bad_lookup(cid):
            raise RuntimeError("network error")

        cvs = ContentVerificationSystem(mock_ipfs_client, provenance_lookup_fn=bad_lookup)
        result = await cvs._load_provenance_chain("QmBad")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_skips_lookup(self, mock_ipfs_client):
        """get_provenance_chain() returns the cached chain without calling lookup_fn again."""
        calls = []

        async def lookup(cid):
            calls.append(cid)
            import time as _time
            return {"cid": cid, "creator_id": "x", "registered_at": _time.time()}

        cvs = ContentVerificationSystem(mock_ipfs_client, provenance_lookup_fn=lookup)
        # First call — populates cache via get_provenance_chain (the public API)
        chain1 = await cvs.get_provenance_chain("QmCached")
        assert chain1 is not None
        assert len(calls) == 1

        # Second call — should hit the in-memory cache, NOT call lookup_fn again
        chain2 = await cvs.get_provenance_chain("QmCached")
        assert chain2 is chain1
        assert len(calls) == 1

    def test_create_verification_system_passes_lookup_fn(self, mock_ipfs_client):
        """create_verification_system() forwards provenance_lookup_fn to the instance."""
        async def lookup(cid):
            return None

        cvs = create_verification_system(mock_ipfs_client, provenance_lookup_fn=lookup)
        assert cvs._provenance_lookup_fn is lookup

    def test_create_verification_system_no_lookup_fn(self, mock_ipfs_client):
        """create_verification_system() without lookup_fn gives None for _provenance_lookup_fn."""
        cvs = create_verification_system(mock_ipfs_client)
        assert cvs._provenance_lookup_fn is None
