"""Unit tests for the payment_escrow module."""

import asyncio
import pytest
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import PaymentEscrow, EscrowStatus


@pytest.fixture
async def escrow_env():
    """Create a ledger + escrow with two wallets."""
    ledger = LocalLedger(db_path=":memory:")
    await ledger.initialize()
    await ledger.create_wallet("node-a", "Alice")
    await ledger.create_wallet("node-b", "Bob")
    await ledger.credit("node-a", 100.0, TransactionType.WELCOME_GRANT, "start")
    escrow = PaymentEscrow(ledger=ledger, node_id="node-a")
    yield {"ledger": ledger, "escrow": escrow}
    await ledger.close()


@pytest.mark.asyncio
async def test_create_escrow_locks_funds(escrow_env):
    ledger = escrow_env["ledger"]
    esc = escrow_env["escrow"]
    before = await ledger.get_balance("node-a")
    entry = await esc.create_escrow("job-1", 10.0, "node-a")
    assert entry is not None
    assert entry.amount == 10.0
    after = await ledger.get_balance("node-a")
    assert after == pytest.approx(before - 10.0)
    escrow_bal = await ledger.get_balance(f"escrow-{entry.escrow_id}")
    assert escrow_bal == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_create_escrow_insufficient_balance(escrow_env):
    esc = escrow_env["escrow"]
    entry = await esc.create_escrow("job-2", 999.0, "node-a")
    assert entry is None


@pytest.mark.asyncio
async def test_release_escrow_pays_provider(escrow_env):
    ledger = escrow_env["ledger"]
    esc = escrow_env["escrow"]
    await esc.create_escrow("job-1", 10.0, "node-a")
    tx = await esc.release_escrow("job-1", "node-b", consensus_reached=True)
    assert tx is not None
    bob_balance = await ledger.get_balance("node-b")
    assert bob_balance == pytest.approx(10.0)
    alice_balance = await ledger.get_balance("node-a")
    assert alice_balance == pytest.approx(90.0)


@pytest.mark.asyncio
async def test_release_escrow_partial_refund(escrow_env):
    ledger = escrow_env["ledger"]
    esc = escrow_env["escrow"]
    await esc.create_escrow("job-1", 10.0, "node-a")
    tx = await esc.release_escrow("job-1", "node-b", consensus_reached=True, partial_amount=7.0)
    # Provider gets partial, requester gets the rest
    bob_balance = await ledger.get_balance("node-b")
    alice_balance = await ledger.get_balance("node-a")
    assert bob_balance == pytest.approx(7.0)
    assert alice_balance == pytest.approx(93.0)  # 100 - 10 (lock) + 3 (refund)


@pytest.mark.asyncio
async def test_refund_escrow(escrow_env):
    ledger = escrow_env["ledger"]
    esc = escrow_env["escrow"]
    await esc.create_escrow("job-1", 10.0, "node-a")
    result = await esc.refund_escrow("job-1", reason="Test refund")
    assert result is True
    alice_balance = await ledger.get_balance("node-a")
    assert alice_balance == pytest.approx(100.0)  # Full refund


@pytest.mark.asyncio
async def test_get_escrow_by_job(escrow_env):
    esc = escrow_env["escrow"]
    await esc.create_escrow("job-abc", 5.0, "node-a")
    entry = esc.get_escrow("job-abc")
    assert entry is not None
    assert entry.status == EscrowStatus.PENDING


@pytest.mark.asyncio
async def test_escrow_stats(escrow_env):
    esc = escrow_env["escrow"]
    await esc.create_escrow("job-1", 5.0, "node-a")
    await esc.create_escrow("job-2", 3.0, "node-a")
    stats = esc.get_stats()
    assert stats["total_escrows"] == 2
    assert stats["total_locked_ftns"] == pytest.approx(8.0)
