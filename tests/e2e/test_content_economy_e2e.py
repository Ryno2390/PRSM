"""
End-to-End Tests for Phase 4 Content Economy
============================================

Tests the complete flow:
1. Node A uploads content with royalty tracking
2. Node B retrieves content and pays royalties
3. Replication is tracked and enforced
4. Payments are recorded in ledgers

Requires:
- IPFS daemon running (or mocked)
- Two test nodes with separate data directories
"""

import asyncio
import os
import tempfile
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from prsm.node.node import PRSMNode, NodeConfig
from prsm.node.content_economy import (
    ContentEconomy,
    RoyaltyModel,
    PaymentStatus,
    ReplicationStatus,
)
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.identity import NodeIdentity


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test node data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_ipfs():
    """Mock IPFS client for testing without a running daemon."""
    with patch("prsm.node.storage_provider.StorageProvider._detect_ipfs") as mock_detect:
        mock_detect.return_value = True
        
        with patch("prsm.node.storage_provider.StorageProvider.pin_content") as mock_pin:
            mock_pin.return_value = True
            
            with patch("prsm.node.storage_provider.StorageProvider.verify_pin") as mock_verify:
                mock_verify.return_value = True
                
                with patch("prsm.node.content_uploader.ContentUploader._ipfs_add") as mock_add:
                    mock_add.return_value = ("QmTestCID123456789", 1024)
                    
                    with patch("prsm.node.content_provider.ContentProvider._ipfs_cat") as mock_cat:
                        mock_cat.return_value = b"Test content for E2E testing"
                        
                        yield {
                            "detect": mock_detect,
                            "pin": mock_pin,
                            "verify": mock_verify,
                            "add": mock_add,
                            "cat": mock_cat,
                        }


@pytest.fixture
async def node_a(temp_data_dir, mock_ipfs):
    """Create test node A (primary/content creator)."""
    config = NodeConfig(
        data_dir=os.path.join(temp_data_dir, "node-a"),
        p2p_port=19001,
        api_port=18000,
        roles=["full"],
        storage_gb=10,
        min_replicas=2,
        royalty_model="phase4",
    )
    os.makedirs(config.data_dir, exist_ok=True)
    
    node = PRSMNode(config)
    await node.initialize()
    await node.start()
    
    yield node
    
    await node.stop()


@pytest.fixture
async def node_b(temp_data_dir, mock_ipfs):
    """Create test node B (secondary/content accessor)."""
    config = NodeConfig(
        data_dir=os.path.join(temp_data_dir, "node-b"),
        p2p_port=19002,
        api_port=18001,
        roles=["full"],
        storage_gb=10,
        min_replicas=2,
        royalty_model="phase4",
    )
    os.makedirs(config.data_dir, exist_ok=True)
    
    node = PRSMNode(config)
    await node.initialize()
    await node.start()
    
    yield node
    
    await node.stop()


# ── E2E Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_content_upload_tracks_replication(node_a):
    """Test that content upload creates replication tracking."""
    content_economy = node_a.content_economy
    assert content_economy is not None
    
    # Upload content via content uploader
    content_uploader = node_a.content_uploader
    assert content_uploader is not None
    
    # Upload test content
    result = await content_uploader.upload(
        content=b"Test content for E2E replication tracking",
        filename="test.txt",
        royalty_rate=0.05,
        replicas=2,
    )
    
    assert result is not None
    assert result.cid is not None
    
    # Check replication tracking was created
    status = content_economy._replication_status.get(result.cid)
    assert status is not None
    assert status.min_replicas == 2
    assert status.current_replicas >= 1  # At least we have it


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_content_access_triggers_payment(node_a):
    """Test that content access triggers FTNS payment."""
    content_economy = node_a.content_economy
    ledger = node_a.ledger
    
    # Get initial balance
    initial_balance = await ledger.get_balance(node_a.identity.node_id)
    
    # Process content access
    payment = await content_economy.process_content_access(
        cid="QmTestCID123456789",
        accessor_id=node_a.identity.node_id,
        content_metadata={
            "royalty_rate": 0.05,
            "creator_id": "creator-test-123",
            "parent_cids": [],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    assert payment.amount == Decimal("0.05")
    
    # Check balance decreased
    new_balance = await ledger.get_balance(node_a.identity.node_id)
    assert new_balance < initial_balance


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_royalty_distribution_derivative_work(node_a):
    """Test royalty distribution for derivative content."""
    content_economy = node_a.content_economy
    content_economy.royalty_model = RoyaltyModel.PHASE4
    
    # Process payment for derivative work
    payment = await content_economy.process_content_access(
        cid="QmDerivativeCID",
        accessor_id=node_a.identity.node_id,
        content_metadata={
            "royalty_rate": 0.10,
            "creator_id": "derivative-creator",
            "parent_cids": ["QmOriginalCID"],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    assert payment.royalty_model == RoyaltyModel.PHASE4
    
    # Check distributions
    assert len(payment.royalty_distributions) > 0
    
    # Should have network fee
    network_dist = [d for d in payment.royalty_distributions if d["type"] == "network_fee"]
    assert len(network_dist) > 0
    
    # Total should approximately equal the payment amount
    total_distributed = sum(d["amount"] for d in payment.royalty_distributions)
    assert abs(total_distributed - float(payment.amount)) < 0.01


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_replication_status_update(node_a):
    """Test replication status updates correctly."""
    content_economy = node_a.content_economy
    
    # Create initial tracking
    await content_economy.track_content_upload(
        cid="QmReplicationTest",
        size_bytes=1024,
        replicas_requested=3,
    )
    
    status = content_economy._replication_status.get("QmReplicationTest")
    assert status is not None
    assert status.current_replicas == 1
    
    # Simulate another provider
    await content_economy.update_replication_status(
        cid="QmReplicationTest",
        provider_id="provider-node-xyz",
        has_content=True,
    )
    
    assert status.current_replicas == 2
    assert "provider-node-xyz" in status.providers
    
    # Simulate provider removing content
    await content_economy.update_replication_status(
        cid="QmReplicationTest",
        provider_id="provider-node-xyz",
        has_content=False,
    )
    
    assert status.current_replicas == 1
    assert "provider-node-xyz" not in status.providers


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_semantic_search_disabled_without_vector_store(node_a):
    """Test that semantic search returns empty when no vector store configured."""
    content_economy = node_a.content_economy
    
    # No vector store configured
    assert content_economy.vector_store is None
    
    results = await content_economy.semantic_search(
        query="test query",
        limit=10,
    )
    
    assert results == []


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_content_economy_stats(node_a):
    """Test content economy statistics."""
    content_economy = node_a.content_economy
    
    # Track some content
    await content_economy.track_content_upload("cid-1", 1024, 2)
    await content_economy.track_content_upload("cid-2", 2048, 3)
    
    stats = content_economy.get_stats()
    
    assert stats["tracked_content"] == 2
    assert stats["royalty_model"] == "phase4"
    assert stats["min_replicas"] == 2
    assert stats["vector_store_enabled"] is False


# ── Multi-Node Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
async def test_cross_node_content_retrieval(node_a, node_b):
    """Test content retrieval between two nodes.
    
    This test:
    1. Node A uploads content
    2. Node B requests content
    3. Payment is processed
    """
    # Node A uploads content
    result = await node_a.content_uploader.upload(
        content=b"Cross-node test content",
        filename="cross-node-test.txt",
        royalty_rate=0.05,
        replicas=1,
    )
    
    assert result is not None
    
    # Register content in Node A's content provider
    node_a.content_provider._local_content[result.cid] = {
        "royalty_rate": 0.05,
        "creator_id": node_a.identity.node_id,
        "parent_cids": [],
        "filename": "cross-node-test.txt",
    }
    
    # Node B's content economy processes the access
    initial_balance_b = await node_b.ledger.get_balance(node_b.identity.node_id)
    
    payment = await node_b.content_economy.process_content_access(
        cid=result.cid,
        accessor_id=node_b.identity.node_id,
        content_metadata={
            "royalty_rate": 0.05,
            "creator_id": node_a.identity.node_id,
            "parent_cids": [],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    
    # Node B's balance should decrease
    new_balance_b = await node_b.ledger.get_balance(node_b.identity.node_id)
    assert new_balance_b < initial_balance_b


# ── Integration with Storage Provider ──────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_storage_proof_updates_replication(node_a, mock_ipfs):
    """Test that storage proof success updates replication status."""
    content_economy = node_a.content_economy
    storage_provider = node_a.storage_provider
    
    if not storage_provider:
        pytest.skip("Storage provider not available")
    
    # Track content
    cid = "QmStorageProofTest"
    await content_economy.track_content_upload(cid, 1024, 2)
    
    # Simulate successful storage proof
    await content_economy.update_replication_status(
        cid=cid,
        provider_id=node_a.identity.node_id,
        has_content=True,
    )
    
    status = content_economy._replication_status.get(cid)
    assert status is not None
    assert node_a.identity.node_id in status.providers


# ── On-Chain Integration ───────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("RUN_ONCHAIN_TESTS") != "1",
    reason="Set RUN_ONCHAIN_TESTS=1 to run on-chain tests"
)
async def test_onchain_escrow_for_content_payment(node_a):
    """Test on-chain escrow for content access payment.
    
    Requires:
    - FTNS contract deployed on Base mainnet
    - Private key with FTNS balance
    - RUN_ONCHAIN_TESTS=1 environment variable
    """
    content_economy = node_a.content_economy
    ftns_ledger = content_economy.ftns_ledger
    
    if not ftns_ledger:
        pytest.skip("On-chain FTNS ledger not available")
    
    # Process payment with on-chain escrow
    payment = await content_economy.process_content_access(
        cid="QmOnChainTest",
        accessor_id=node_a.identity.node_id,
        content_metadata={
            "royalty_rate": 0.001,  # Small amount for testing
            "creator_id": "creator-test",
            "parent_cids": [],
        },
    )
    
    assert payment.status == PaymentStatus.COMPLETED
    
    # Check escrow transaction
    if payment.escrow_tx_hash:
        assert payment.escrow_tx_hash.startswith("0x")


# ── Performance Tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
async def test_concurrent_content_uploads(node_a):
    """Test handling multiple concurrent content uploads."""
    content_uploader = node_a.content_uploader
    
    # Upload 10 pieces of content concurrently
    tasks = [
        content_uploader.upload(
            content=f"Concurrent test content {i}".encode(),
            filename=f"concurrent-{i}.txt",
            royalty_rate=0.01,
            replicas=1,
        )
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(successes) == 10
    
    # Each should have unique CID
    cids = [r.cid for r in successes]
    assert len(set(cids)) == 10


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.slow
async def test_concurrent_payments(node_a):
    """Test handling multiple concurrent payment requests."""
    content_economy = node_a.content_economy
    
    # Process 20 payments concurrently
    tasks = [
        content_economy.process_content_access(
            cid=f"QmConcurrentTest{i}",
            accessor_id=node_a.identity.node_id,
            content_metadata={
                "royalty_rate": 0.01,
                "creator_id": f"creator-{i % 5}",  # Multiple creators
                "parent_cids": [],
            },
        )
        for i in range(20)
    ]
    
    payments = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should succeed
    successes = [p for p in payments if not isinstance(p, Exception)]
    assert len(successes) == 20
    
    # All should have unique payment IDs
    payment_ids = [p.payment_id for p in successes]
    assert len(set(payment_ids)) == 20


# ── Error Handling Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_payment_with_insufficient_balance(node_a):
    """Test payment fails gracefully with insufficient balance."""
    content_economy = node_a.content_economy
    
    # Create a new wallet with minimal balance
    await node_a.ledger.create_wallet("poor-node", "Poor Node")
    
    # Try to process expensive payment
    payment = await content_economy.process_content_access(
        cid="QmExpensiveContent",
        accessor_id="poor-node",
        content_metadata={
            "royalty_rate": 100.0,  # Expensive
            "creator_id": "creator-test",
            "parent_cids": [],
        },
    )
    
    assert payment.status == PaymentStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_replication_below_minimum_requests_more(node_a, mock_ipfs):
    """Test that replication below minimum triggers request for more replicas."""
    content_economy = node_a.content_economy
    gossip = node_a.gossip
    
    # Track content with min_replicas=3 but only 1 current
    await content_economy.track_content_upload("QmUnderReplicated", 1024, 3)
    
    status = content_economy._replication_status.get("QmUnderReplicated")
    assert status.current_replicas < status.min_replicas
    
    # Trigger check
    await content_economy._check_replication_needs("QmUnderReplicated", status)
    
    # Should have pending request
    assert status.pending_requests > 0
