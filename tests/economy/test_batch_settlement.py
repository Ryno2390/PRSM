"""Tests for BatchSettlementManager."""
import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    SettlementMode,
    PendingTransfer,
)


@dataclass
class FakeTransaction:
    tx_id: str = "tx-001"
    from_wallet: str = "node-abc"
    to_wallet: str = "0x1234567890abcdef1234567890abcdef12345678"
    amount: float = 0.5
    description: str = "test transfer"


@dataclass
class FakeTxRecord:
    tx_hash: str = "0xdeadbeef"
    status: str = "confirmed"
    block_number: int = 12345


@pytest.fixture
def mock_ledger():
    ledger = MagicMock()
    ledger._is_initialized = True
    ledger._connected_address = "0xAABBCCDDEE1234567890AABBCCDDEE1234567890"
    ledger.transfer = AsyncMock(return_value=FakeTxRecord())
    return ledger


@pytest.fixture
def manager(mock_ledger):
    return BatchSettlementManager(
        ftns_ledger=mock_ledger,
        node_id="node-abc",
        connected_address="0xAABBCCDDEE1234567890AABBCCDDEE1234567890",
        mode=SettlementMode.MANUAL,  # no auto-flush for testing
        flush_threshold=100.0,       # high threshold to prevent auto-flush
    )


@pytest.mark.asyncio
async def test_enqueue_valid_transaction(manager):
    """Valid 0x-addressed transaction should be queued."""
    tx = FakeTransaction()
    result = await manager.enqueue(tx)
    assert result is True
    assert len(manager._queue) == 1
    assert manager._queue[0].amount == 0.5


@pytest.mark.asyncio
async def test_enqueue_skips_internal_wallets(manager):
    """Transfers to non-0x wallets (escrow, named) should be skipped."""
    tx = FakeTransaction(to_wallet="escrow-abc123")
    result = await manager.enqueue(tx)
    assert result is False
    assert len(manager._queue) == 0


@pytest.mark.asyncio
async def test_enqueue_resolves_node_id(manager):
    """Transfers to the node's own ID should resolve to on-chain address."""
    tx = FakeTransaction(to_wallet="node-abc")
    result = await manager.enqueue(tx)
    assert result is True
    assert manager._queue[0].to_wallet == manager._connected_address


@pytest.mark.asyncio
async def test_enqueue_dedup(manager):
    """Same tx_id should not be queued twice."""
    tx = FakeTransaction()
    await manager.enqueue(tx)
    await manager.enqueue(tx)  # duplicate
    assert len(manager._queue) == 1


@pytest.mark.asyncio
async def test_enqueue_skips_zero_amount(manager):
    """Zero-amount transfers should be skipped."""
    tx = FakeTransaction(amount=0)
    result = await manager.enqueue(tx)
    assert result is False


@pytest.mark.asyncio
async def test_flush_empty_queue(manager):
    """Flushing an empty queue should return zero counts."""
    result = await manager.flush()
    assert result.settled_count == 0
    assert result.net_transfers == 0


@pytest.mark.asyncio
async def test_flush_single_transfer(manager, mock_ledger):
    """Single queued transfer should produce one on-chain tx."""
    tx = FakeTransaction(amount=1.0)
    await manager.enqueue(tx)
    result = await manager.flush()

    assert result.settled_count == 1
    assert result.net_transfers == 1
    assert result.total_amount == 1.0
    assert len(result.tx_hashes) == 1
    mock_ledger.transfer.assert_called_once()


@pytest.mark.asyncio
async def test_flush_nets_multiple_to_same_address(manager, mock_ledger):
    """Multiple transfers to the same address should be netted into one."""
    for i in range(5):
        tx = FakeTransaction(tx_id=f"tx-{i}", amount=0.2)
        await manager.enqueue(tx)

    result = await manager.flush()
    assert result.settled_count == 5
    assert result.net_transfers == 1  # all netted into one
    assert abs(result.total_amount - 1.0) < 0.001
    mock_ledger.transfer.assert_called_once()

    # The netted amount should be 1.0
    call_args = mock_ledger.transfer.call_args
    assert abs(call_args.kwargs["amount_ftns"] - 1.0) < 0.001


@pytest.mark.asyncio
async def test_flush_clears_queue(manager):
    """Queue should be empty after flush."""
    await manager.enqueue(FakeTransaction())
    await manager.flush()
    assert len(manager._queue) == 0


@pytest.mark.asyncio
async def test_flush_records_history(manager):
    """Flush results should be recorded in settlement history."""
    await manager.enqueue(FakeTransaction())
    await manager.flush()

    history = manager.get_history()
    assert len(history) == 1
    assert history[0]["settled_count"] == 1


@pytest.mark.asyncio
async def test_stats(manager):
    """Stats should reflect current queue state."""
    await manager.enqueue(FakeTransaction(amount=0.5))
    stats = manager.get_stats()

    assert stats["mode"] == "manual"
    assert stats["queue_size"] == 1
    assert stats["pending_amount"] == 0.5
    assert stats["total_settled"] == 0


@pytest.mark.asyncio
async def test_stats_after_flush(manager):
    """Stats should update after flush."""
    await manager.enqueue(FakeTransaction(amount=0.5))
    await manager.flush()
    stats = manager.get_stats()

    assert stats["queue_size"] == 0
    assert stats["pending_amount"] == 0
    assert stats["total_settled"] == 1


@pytest.mark.asyncio
async def test_gas_savings_tracking(manager, mock_ledger):
    """Gas savings should be tracked when netting reduces transfers."""
    for i in range(10):
        await manager.enqueue(FakeTransaction(tx_id=f"tx-{i}", amount=0.1))

    await manager.flush()
    stats = manager.get_stats()

    # 10 transfers netted to 1 = 9 gas txs saved
    assert stats["gas_txs_saved"] == 9


@pytest.mark.asyncio
async def test_flush_handles_transfer_failure(manager, mock_ledger):
    """Transfer failures should be captured in errors, not crash."""
    mock_ledger.transfer.side_effect = Exception("RPC timeout")

    await manager.enqueue(FakeTransaction())
    result = await manager.flush()

    assert result.settled_count == 1
    assert len(result.errors) == 1
    assert "RPC timeout" in result.errors[0]


@pytest.mark.asyncio
async def test_threshold_auto_flush(mock_ledger):
    """Threshold mode should auto-flush when pending exceeds threshold."""
    manager = BatchSettlementManager(
        ftns_ledger=mock_ledger,
        node_id="node-abc",
        connected_address="0xAABBCCDDEE1234567890AABBCCDDEE1234567890",
        mode=SettlementMode.THRESHOLD,
        flush_threshold=0.5,
    )

    # This should trigger auto-flush (0.6 > 0.5 threshold)
    tx = FakeTransaction(amount=0.6)
    await manager.enqueue(tx)

    # Give the auto-flush task a moment to run
    await asyncio.sleep(0.1)

    # Queue should be flushed
    assert len(manager._queue) == 0
    assert manager._total_settled == 1


@pytest.mark.asyncio
async def test_get_pending(manager):
    """get_pending should return formatted pending transfers."""
    await manager.enqueue(FakeTransaction())
    pending = manager.get_pending()

    assert len(pending) == 1
    assert "tx_id" in pending[0]
    assert "amount" in pending[0]
    assert "age_seconds" in pending[0]
