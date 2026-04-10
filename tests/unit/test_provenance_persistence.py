"""
Unit tests for provenance persistence functionality.

Tests verify that:
1. Uploads persist provenance records to the database
2. DB write failures are non-blocking
3. Hydration loads records from DB correctly
4. Hydration does not overwrite in-memory records
5. record_access updates DB access stats
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from prsm.node.content_uploader import ContentUploader, UploadedContent


class TestProvenancePersistence:
    """Test suite for provenance persistence."""

    @pytest.fixture
    def mock_identity(self):
        """Create a mock identity for testing."""
        identity = MagicMock()
        identity.node_id = "test_node_123"
        identity.public_key_b64 = "dGVzdF9wdWJsaWNfa2V5X2Jhc2U2NA=="  # base64 encoded test key
        identity.sign = MagicMock(return_value="test_signature_hex")
        return identity

    @pytest.fixture
    def mock_gossip(self):
        """Create a mock gossip service."""
        return AsyncMock()

    @pytest.fixture
    def mock_ledger(self):
        """Create a mock ledger service."""
        return AsyncMock()

    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport for testing."""
        return AsyncMock()

    @pytest.fixture
    async def content_uploader(self, mock_identity, mock_gossip, mock_ledger, mock_transport):
        """Create a ContentUploader instance for testing."""
        uploader = ContentUploader(
            identity=mock_identity,
            gossip=mock_gossip,
            ledger=mock_ledger,
            ipfs_api_url="http://127.0.0.1:5001",
            transport=mock_transport
        )
        # Mock the internal IPFS session to avoid actual HTTP calls
        uploader._ipfs_session = AsyncMock()
        return uploader

    @pytest.mark.asyncio
    async def test_upload_persists_to_db(self, content_uploader):
        """Verify _persist_provenance is called after successful upload."""
        # Mock ProvenanceQueries.upsert_provenance
        with patch('prsm.core.database.ProvenanceQueries.upsert_provenance', new_callable=AsyncMock) as mock_upsert:
            mock_upsert.return_value = True
            
            # Mock the IPFS add operation directly
            with patch.object(content_uploader, '_ipfs_add', new_callable=AsyncMock) as mock_add:
                mock_add.return_value = "QmTestCID123456789"
                
                # Create a test file
                test_content = b"test model weights content"
                
                # Upload the content (creator_id comes from identity.node_id)
                result = await content_uploader.upload(
                    content=test_content,
                    filename="test_model.bin"
                )
                
                # Verify upload succeeded
                assert result is not None
                assert result.content_id in content_uploader.uploaded_content
                
                # Verify ProvenanceQueries.upsert_provenance was called
                mock_upsert.assert_called_once()
                call_args = mock_upsert.call_args[0][0]
                assert call_args["content_id"] == result.content_id
                assert call_args["creator_id"] == "test_node_123"  # from mock_identity.node_id
                assert call_args["filename"] == "test_model.bin"

    @pytest.mark.asyncio
    async def test_persist_failure_is_non_blocking(self, content_uploader):
        """Verify DB write failure does not prevent upload from succeeding."""
        # Mock ProvenanceQueries.upsert_provenance to raise an exception
        with patch('prsm.core.database.ProvenanceQueries.upsert_provenance', new_callable=AsyncMock) as mock_upsert:
            mock_upsert.side_effect = Exception("Database connection failed")
            
            # Mock the IPFS add operation directly
            with patch.object(content_uploader, '_ipfs_add', new_callable=AsyncMock) as mock_add:
                mock_add.return_value = "QmTestCID123456789"
                
                # Create a test file
                test_content = b"test model weights content"
                
                # Upload the content - should still succeed despite DB failure
                result = await content_uploader.upload(
                    content=test_content,
                    filename="test_model.bin"
                )
                
                # Verify upload succeeded even though DB write failed
                assert result is not None
                assert result.content_id in content_uploader.uploaded_content
                assert content_uploader.uploaded_content[result.content_id].filename == "test_model.bin"

    @pytest.mark.asyncio
    async def test_hydrate_loads_records_from_db(self, content_uploader):
        """Verify _hydrate_from_db populates uploaded_content from DB rows."""
        # Mock ProvenanceQueries.load_all_for_node to return test records
        mock_records = [
            {
                "content_id": "QmTestCID111111111",
                "filename": "model_v1.bin",
                "size_bytes": 1024,
                "content_hash": "abc123def456",
                "creator_id": "test_node_123",
                "created_at": 1700000000.0,
                "provenance_signature": "sig1",
                "royalty_rate": 0.02,
                "parent_content_ids": [],
                "access_count": 5,
                "total_royalties": 10.5,
                "is_sharded": False,
                "manifest_content_id": None,
                "total_shards": 0,
                "embedding_id": None,
                "near_duplicate_of": None,
                "near_duplicate_similarity": None,
            },
            {
                "content_id": "QmTestCID222222222",
                "filename": "model_v2.bin",
                "size_bytes": 2048,
                "content_hash": "def789ghi012",
                "creator_id": "test_node_123",
                "created_at": 1700100000.0,
                "provenance_signature": "sig2",
                "royalty_rate": 0.03,
                "parent_content_ids": ["QmParentCID123"],
                "access_count": 10,
                "total_royalties": 25.0,
                "is_sharded": True,
                "manifest_content_id": "QmManifestCID123",
                "total_shards": 3,
                "embedding_id": "emb123",
                "near_duplicate_of": None,
                "near_duplicate_similarity": None,
            },
        ]
        
        with patch('prsm.core.database.ProvenanceQueries.load_all_for_node', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_records
            
            # Call hydration
            loaded_count = await content_uploader._hydrate_from_db()
            
            # Verify correct number of records loaded
            assert loaded_count == 2
            
            # Verify both CIDs are in uploaded_content
            assert "QmTestCID111111111" in content_uploader.uploaded_content
            assert "QmTestCID222222222" in content_uploader.uploaded_content
            
            # Verify fields match the mock records
            record1 = content_uploader.uploaded_content["QmTestCID111111111"]
            assert record1.filename == "model_v1.bin"
            assert record1.access_count == 5
            assert record1.total_royalties == 10.5
            
            record2 = content_uploader.uploaded_content["QmTestCID222222222"]
            assert record2.filename == "model_v2.bin"
            assert record2.parent_content_ids == ["QmParentCID123"]
            assert record2.is_sharded is True

    @pytest.mark.asyncio
    async def test_hydrate_skips_existing_in_memory_records(self, content_uploader):
        """Verify hydration does not overwrite in-memory records."""
        # Pre-populate uploaded_content with a CID having access_count=5
        existing_cid = "QmExistingCID123456"
        content_uploader.uploaded_content[existing_cid] = UploadedContent(
            content_id=existing_cid,
            filename="existing_model.bin",
            size_bytes=512,
            content_hash="existing_hash",
            creator_id="test_node_123",
            created_at=1700000000.0,
            provenance_signature="existing_sig",
            royalty_rate=0.01,
            parent_content_ids=[],
            access_count=5,  # This should NOT be overwritten
            total_royalties=15.0,
            is_sharded=False,
            manifest_content_id=None,
            total_shards=0,
            embedding_id=None,
            near_duplicate_of=None,
            near_duplicate_similarity=None,
        )
        
        # Mock DB returning same CID with access_count=0
        mock_records = [
            {
                "content_id": existing_cid,
                "filename": "existing_model.bin",
                "size_bytes": 512,
                "content_hash": "existing_hash",
                "creator_id": "test_node_123",
                "created_at": 1700000000.0,
                "provenance_signature": "existing_sig",
                "royalty_rate": 0.01,
                "parent_content_ids": [],
                "access_count": 0,  # Different from in-memory value
                "total_royalties": 0.0,  # Different from in-memory value
                "is_sharded": False,
                "manifest_content_id": None,
                "total_shards": 0,
                "embedding_id": None,
                "near_duplicate_of": None,
                "near_duplicate_similarity": None,
            },
        ]
        
        with patch('prsm.core.database.ProvenanceQueries.load_all_for_node', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_records
            
            # Call hydration
            loaded_count = await content_uploader._hydrate_from_db()
            
            # Verify no records were loaded (skipped existing)
            assert loaded_count == 0
            
            # Verify in-memory record was NOT overwritten
            assert content_uploader.uploaded_content[existing_cid].access_count == 5
            assert content_uploader.uploaded_content[existing_cid].total_royalties == 15.0

    @pytest.mark.asyncio
    async def test_record_access_updates_db(self, content_uploader):
        """Verify record_access calls update_access_stats after royalty credit."""
        # Pre-populate uploaded_content with a CID
        test_cid = "QmTestAccessCID123"
        content_uploader.uploaded_content[test_cid] = UploadedContent(
            content_id=test_cid,
            filename="test_model.bin",
            size_bytes=1024,
            content_hash="test_hash",
            creator_id="test_node_123",
            created_at=1700000000.0,
            provenance_signature="test_sig",
            royalty_rate=0.02,  # 2% royalty rate
            parent_content_ids=[],
            access_count=0,
            total_royalties=0.0,
            is_sharded=False,
            manifest_content_id=None,
            total_shards=0,
            embedding_id=None,
            near_duplicate_of=None,
            near_duplicate_similarity=None,
        )
        
        # Mock ledger.credit to succeed
        content_uploader.ledger.credit = AsyncMock(return_value=True)
        
        # Mock ProvenanceQueries.update_access_stats
        with patch('prsm.core.database.ProvenanceQueries.update_access_stats', new_callable=AsyncMock) as mock_update:
            mock_update.return_value = True
            
            # Call record_access
            await content_uploader.record_access(test_cid, "requesting_node_456")
            
            # Verify update_access_stats was called with correct parameters
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args.kwargs["cid"] == test_cid
            assert call_args.kwargs["access_count_delta"] == 1
            assert call_args.kwargs["royalty_delta"] == 0.02  # royalty_rate
