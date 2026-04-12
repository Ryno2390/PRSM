"""
Unit tests for royalty pipeline functionality.

Tests verify that:
1. Single creator royalty triggers platform FTNS transfer
2. Platform transfer failure is non-blocking
3. Multilevel royalty fires multiple platform transfers
4. Gossip source royalty triggers platform transfer
5. Insufficient balance is logged at INFO level, not raised as error
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from decimal import Decimal

from prsm.node.content_uploader import ContentUploader, UploadedContent
from prsm.node.local_ledger import TransactionType


class TestRoyaltyPipeline:
    """Test suite for royalty pipeline platform FTNS transfers."""

    @pytest.fixture
    def mock_identity(self):
        """Create a mock identity for testing."""
        identity = MagicMock()
        identity.node_id = "creator_node_123"
        identity.public_key_b64 = "dGVzdF9wdWJsaWNfa2V5X2Jhc2U2NA=="
        identity.sign = MagicMock(return_value="test_signature_hex")
        return identity

    @pytest.fixture
    def mock_gossip(self):
        """Create a mock gossip service."""
        return AsyncMock()

    @pytest.fixture
    def mock_ledger(self):
        """Create a mock ledger service."""
        ledger = AsyncMock()
        # Mock the credit method to return a transaction object
        ledger.credit = AsyncMock(return_value=MagicMock(tx_id="tx_123"))
        return ledger

    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport for testing."""
        return AsyncMock()

    @pytest.fixture
    def mock_content_index(self):
        """Create a mock content index."""
        return MagicMock()

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
    async def test_single_creator_royalty_triggers_platform_transfer(self, content_uploader, mock_ledger):
        """Verify record_access() with single creator fires platform FTNS transfer."""
        # Setup: Add content to uploaded_content
        cid = "QmTestCID123456789abcdef"
        accessor_id = "accessor_node_456"
        royalty_rate = 0.05
        
        content = UploadedContent(
            content_id=cid,
            filename="test_model.bin",
            size_bytes=1024,
            content_hash="abc123hash",
            creator_id=content_uploader.identity.node_id,
            royalty_rate=royalty_rate,
        )
        content_uploader.uploaded_content[cid] = content

        # Mock AtomicFTNSService.transfer_tokens_atomic
        mock_transfer_result = MagicMock()
        mock_transfer_result.success = True
        mock_transfer_result.transaction_id = "tx_platform_123"

        with patch('prsm.economy.tokenomics.atomic_ftns_service.AtomicFTNSService') as MockAtomicFTNSService:
            mock_service_instance = AsyncMock()
            mock_service_instance.transfer_tokens_atomic = AsyncMock(return_value=mock_transfer_result)
            MockAtomicFTNSService.return_value = mock_service_instance

            # Mock _update_provenance_access to avoid DB calls
            with patch.object(content_uploader, '_update_provenance_access', new_callable=AsyncMock):
                # Execute record_access
                await content_uploader.record_access(cid, accessor_id)

            # Verify local ledger credit was called
            mock_ledger.credit.assert_called_once()
            call_args = mock_ledger.credit.call_args
            assert call_args[1]["wallet_id"] == content_uploader.identity.node_id
            assert call_args[1]["amount"] == royalty_rate
            assert call_args[1]["tx_type"] == TransactionType.CONTENT_ROYALTY

            # Verify platform FTNS transfer was called
            mock_service_instance.transfer_tokens_atomic.assert_called_once()
            transfer_call_args = mock_service_instance.transfer_tokens_atomic.call_args
            assert transfer_call_args[1]["from_user_id"] == accessor_id
            assert transfer_call_args[1]["to_user_id"] == content_uploader.identity.node_id
            assert float(transfer_call_args[1]["amount"]) == royalty_rate

    @pytest.mark.asyncio
    async def test_platform_transfer_failure_is_non_blocking(self, content_uploader, mock_ledger):
        """Verify platform FTNS failure does not prevent record_access from completing."""
        # Setup: Add content to uploaded_content
        cid = "QmTestCID123456789abcdef"
        accessor_id = "accessor_node_456"
        royalty_rate = 0.05
        
        content = UploadedContent(
            content_id=cid,
            filename="test_model.bin",
            size_bytes=1024,
            content_hash="abc123hash",
            creator_id=content_uploader.identity.node_id,
            royalty_rate=royalty_rate,
        )
        content_uploader.uploaded_content[cid] = content

        # Mock AtomicFTNSService.transfer_tokens_atomic to fail
        mock_transfer_result = MagicMock()
        mock_transfer_result.success = False
        mock_transfer_result.error_message = "Database connection failed"

        with patch('prsm.economy.tokenomics.atomic_ftns_service.AtomicFTNSService') as MockAtomicFTNSService:
            mock_service_instance = AsyncMock()
            mock_service_instance.transfer_tokens_atomic = AsyncMock(return_value=mock_transfer_result)
            MockAtomicFTNSService.return_value = mock_service_instance

            # Mock _update_provenance_access to avoid DB calls
            with patch.object(content_uploader, '_update_provenance_access', new_callable=AsyncMock):
                # Execute record_access - should NOT raise
                await content_uploader.record_access(cid, accessor_id)

            # Verify local ledger credit was still called (not affected by platform failure)
            mock_ledger.credit.assert_called_once()

            # Verify content royalties were updated (local ledger succeeded)
            assert content.total_royalties == royalty_rate

    @pytest.mark.asyncio
    async def test_multilevel_royalty_fires_multiple_platform_transfers(self, content_uploader, mock_ledger, mock_content_index):
        """Verify derivative content with local source creator fires 2 platform transfers."""
        # Setup: Add derivative content with parent CID
        parent_cid = "QmParentCID123456789"
        derivative_cid = "QmDerivativeCID123456"
        accessor_id = "accessor_node_456"
        royalty_rate = 0.10
        
        # Parent content (source) - same node
        parent_content = UploadedContent(
            content_id=parent_cid,
            filename="source_model.bin",
            size_bytes=1024,
            content_hash="parent_hash",
            creator_id=content_uploader.identity.node_id,
            royalty_rate=0.05,
        )
        content_uploader.uploaded_content[parent_cid] = parent_content

        # Derivative content
        derivative_content = UploadedContent(
            content_id=derivative_cid,
            filename="derivative_model.bin",
            size_bytes=2048,
            content_hash="derivative_hash",
            creator_id=content_uploader.identity.node_id,
            royalty_rate=royalty_rate,
            parent_cids=[parent_cid],
        )
        content_uploader.uploaded_content[derivative_cid] = derivative_content

        # Mock content_index to resolve parent creator
        mock_parent_record = MagicMock()
        mock_parent_record.creator_id = content_uploader.identity.node_id
        mock_content_index.lookup = MagicMock(return_value=mock_parent_record)
        content_uploader.content_index = mock_content_index

        # Mock AtomicFTNSService.transfer_tokens_atomic
        mock_transfer_result = MagicMock()
        mock_transfer_result.success = True

        with patch('prsm.economy.tokenomics.atomic_ftns_service.AtomicFTNSService') as MockAtomicFTNSService:
            mock_service_instance = AsyncMock()
            mock_service_instance.transfer_tokens_atomic = AsyncMock(return_value=mock_transfer_result)
            MockAtomicFTNSService.return_value = mock_service_instance

            # Mock _update_provenance_access
            with patch.object(content_uploader, '_update_provenance_access', new_callable=AsyncMock):
                # Execute record_access
                await content_uploader.record_access(derivative_cid, accessor_id)

            # Verify 2 platform FTNS transfers were called:
            # 1. Derivative share (70%)
            # 2. Source share (25% - since parent creator is on same node)
            assert mock_service_instance.transfer_tokens_atomic.call_count == 2

            # Check first call (derivative share)
            first_call = mock_service_instance.transfer_tokens_atomic.call_args_list[0]
            expected_derivative_share = royalty_rate * 0.70
            assert float(first_call[1]["amount"]) == pytest.approx(expected_derivative_share, rel=1e-3)

            # Check second call (source share)
            second_call = mock_service_instance.transfer_tokens_atomic.call_args_list[1]
            expected_source_share = royalty_rate * 0.25
            assert float(second_call[1]["amount"]) == pytest.approx(expected_source_share, rel=1e-3)

    @pytest.mark.asyncio
    async def test_gossip_source_royalty_triggers_platform_transfer(self, content_uploader, mock_ledger):
        """Verify _on_content_access() as source creator fires platform FTNS transfer."""
        # Setup: Add source content that will be referenced by a derivative
        source_cid = "QmSourceCID123456789"
        derivative_cid = "QmDerivativeCID123456"
        accessor_id = "accessor_node_456"
        creator_id = "other_node_789"  # Different node created the derivative
        royalty_rate = 0.10
        
        # Source content on this node
        source_content = UploadedContent(
            content_id=source_cid,
            filename="source_model.bin",
            size_bytes=1024,
            content_hash="source_hash",
            creator_id=content_uploader.identity.node_id,
            royalty_rate=0.05,
        )
        content_uploader.uploaded_content[source_cid] = source_content

        # Mock AtomicFTNSService.transfer_tokens_atomic
        mock_transfer_result = MagicMock()
        mock_transfer_result.success = True

        with patch('prsm.economy.tokenomics.atomic_ftns_service.AtomicFTNSService') as MockAtomicFTNSService:
            mock_service_instance = AsyncMock()
            mock_service_instance.transfer_tokens_atomic = AsyncMock(return_value=mock_transfer_result)
            MockAtomicFTNSService.return_value = mock_service_instance

            # Create gossip message data
            gossip_data = {
                "content_id": derivative_cid,
                "accessor_id": accessor_id,
                "creator_id": creator_id,
                "royalty_rate": royalty_rate,
                "parent_cids": [source_cid],  # Our content is a parent
            }

            # Execute _on_content_access (origin is different from our node)
            await content_uploader._on_content_access(
                subtype="content_access",
                data=gossip_data,
                origin=creator_id
            )

            # Verify platform FTNS transfer was called for source royalty
            mock_service_instance.transfer_tokens_atomic.assert_called_once()
            transfer_call = mock_service_instance.transfer_tokens_atomic.call_args
            assert transfer_call[1]["from_user_id"] == accessor_id
            assert transfer_call[1]["to_user_id"] == content_uploader.identity.node_id
            # Source pool is 25% of royalty rate
            expected_amount = royalty_rate * 0.25
            assert float(transfer_call[1]["amount"]) == pytest.approx(expected_amount, rel=1e-3)

    @pytest.mark.asyncio
    async def test_insufficient_balance_logged_not_raised(self, content_uploader, mock_ledger, caplog):
        """Verify accessor with zero FTNS is logged at INFO level, not raised as error."""
        import logging
        
        # Setup: Add content to uploaded_content
        cid = "QmTestCID123456789abcdef"
        accessor_id = "accessor_node_456"
        royalty_rate = 0.05
        
        content = UploadedContent(
            content_id=cid,
            filename="test_model.bin",
            size_bytes=1024,
            content_hash="abc123hash",
            creator_id=content_uploader.identity.node_id,
            royalty_rate=royalty_rate,
        )
        content_uploader.uploaded_content[cid] = content

        # Mock AtomicFTNSService.transfer_tokens_atomic to return insufficient balance error
        mock_transfer_result = MagicMock()
        mock_transfer_result.success = False
        mock_transfer_result.error_message = "Insufficient balance: 0 < 0.05"

        with patch('prsm.economy.tokenomics.atomic_ftns_service.AtomicFTNSService') as MockAtomicFTNSService:
            mock_service_instance = AsyncMock()
            mock_service_instance.transfer_tokens_atomic = AsyncMock(return_value=mock_transfer_result)
            MockAtomicFTNSService.return_value = mock_service_instance

            # Mock _update_provenance_access
            with patch.object(content_uploader, '_update_provenance_access', new_callable=AsyncMock):
                # Execute record_access with log capture
                with caplog.at_level(logging.INFO):
                    await content_uploader.record_access(cid, accessor_id)

            # Verify no exception was raised
            # Verify local ledger credit was still called
            mock_ledger.credit.assert_called_once()

            # Verify INFO level log message about deferred royalty
            # The log should contain "Platform royalty deferred" or similar
            info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
            assert any("Platform royalty deferred" in msg or "deferred" in msg.lower() for msg in info_messages), \
                f"Expected INFO log about deferred royalty, got: {info_messages}"
