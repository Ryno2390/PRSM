"""Unit tests for PaymentEscrow state-machine idempotency.

Phase 2 Task 3. Verifies the state-machine guards RemoteShardDispatcher
relies on:
  - create_escrow debits requester, credits escrow holding wallet
  - release_escrow transfers to provider, marks RELEASED
  - refund_escrow returns to requester, marks REFUNDED
  - double-release on RELEASED is a no-op with warning (self-idempotent)
  - double-refund on REFUNDED is a no-op with warning (self-idempotent)
  - release-after-refund raises EscrowAlreadyFinalizedError
  - refund-after-release raises EscrowAlreadyFinalizedError
  - release on unknown job_id returns None (legacy)
  - refund on unknown job_id returns False (legacy)

Test uses the real PaymentEscrow API (job_id-keyed) — see
docs/2026-04-12-phase2-remote-compute-plan.md Task 3.
"""
from __future__ import annotations

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import (
    EscrowAlreadyFinalizedError,
    EscrowStatus,
    PaymentEscrow,
)


@pytest.fixture
async def ledger():
    led = LocalLedger(":memory:")
    await led.initialize()
    await led.create_wallet("alice", "Alice")
    await led.create_wallet("bob", "Bob")
    await led.credit(
        wallet_id="alice",
        amount=100.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="seed",
    )
    yield led
    if led._db is not None:
        await led._db.close()


@pytest.fixture
async def escrow(ledger):
    return PaymentEscrow(ledger=ledger, node_id="alice")


@pytest.mark.asyncio
async def test_create_escrow_debits_requester(ledger, escrow):
    """create_escrow moves FTNS from requester to escrow holding wallet."""
    assert await ledger.get_balance("alice") == 100.0

    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    assert entry is not None

    assert await ledger.get_balance("alice") == 90.0
    assert entry.status == EscrowStatus.PENDING
    assert entry.amount == 10.0


@pytest.mark.asyncio
async def test_release_escrow_transfers_to_provider(ledger, escrow):
    """release_escrow transfers the held amount to provider and marks RELEASED."""
    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )

    tx = await escrow.release_escrow("job-1", "bob")
    assert tx is not None

    assert await ledger.get_balance("bob") == 10.0
    assert entry.status == EscrowStatus.RELEASED


@pytest.mark.asyncio
async def test_refund_escrow_returns_to_requester(ledger, escrow):
    """refund_escrow returns the held amount to original requester and marks REFUNDED."""
    assert await ledger.get_balance("alice") == 100.0
    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    assert await ledger.get_balance("alice") == 90.0

    ok = await escrow.refund_escrow("job-1", reason="test")
    assert ok is True

    assert await ledger.get_balance("alice") == 100.0
    assert entry.status == EscrowStatus.REFUNDED


@pytest.mark.asyncio
async def test_release_escrow_idempotent(ledger, escrow, caplog):
    """Double-release on same job_id is a no-op with warning; doesn't double-pay."""
    await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    first = await escrow.release_escrow("job-1", "bob")
    assert first is not None
    assert await ledger.get_balance("bob") == 10.0

    with caplog.at_level("WARNING"):
        second = await escrow.release_escrow("job-1", "bob")

    assert second is None
    assert await ledger.get_balance("bob") == 10.0
    assert any("already released" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_refund_escrow_idempotent(ledger, escrow, caplog):
    """Double-refund on same job_id is a no-op with warning; doesn't double-refund."""
    await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    ok1 = await escrow.refund_escrow("job-1", reason="first")
    assert ok1 is True
    assert await ledger.get_balance("alice") == 100.0

    with caplog.at_level("WARNING"):
        ok2 = await escrow.refund_escrow("job-1", reason="second")

    assert ok2 is True
    assert await ledger.get_balance("alice") == 100.0
    assert any("already refunded" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_release_after_refund_raises(ledger, escrow):
    """Calling release_escrow on a REFUNDED escrow raises EscrowAlreadyFinalizedError."""
    await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    await escrow.refund_escrow("job-1", reason="test")

    with pytest.raises(EscrowAlreadyFinalizedError):
        await escrow.release_escrow("job-1", "bob")


@pytest.mark.asyncio
async def test_refund_after_release_raises(ledger, escrow):
    """Calling refund_escrow on a RELEASED escrow raises EscrowAlreadyFinalizedError."""
    await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    await escrow.release_escrow("job-1", "bob")

    with pytest.raises(EscrowAlreadyFinalizedError):
        await escrow.refund_escrow("job-1", reason="test")


@pytest.mark.asyncio
async def test_release_unknown_job_returns_none(escrow):
    """release_escrow on an unknown job_id returns None (legacy)."""
    result = await escrow.release_escrow("no-such-job", "bob")
    assert result is None


@pytest.mark.asyncio
async def test_refund_unknown_job_returns_false(escrow):
    """refund_escrow on an unknown job_id returns False (legacy)."""
    result = await escrow.refund_escrow("no-such-job")
    assert result is False
