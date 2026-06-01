"""Sprint 907 — PaymentEscrow concurrency safety (money-path robustness).

The money-path adversarial review (2026-06-01) confirmed three CRITICAL
escrow races: release_escrow / release_escrow_split / refund_escrow all
scan for the PENDING escrow, then `await` ledger ops BEFORE writing the
terminal status — a check-then-act across await with no lock. Two
concurrent operations on the same job both observe PENDING during the
awaits, both pay out, and the escrow wallet goes NEGATIVE (FTNS minted
from nothing). Same class as sp898/sp899, in the escrow layer.

These tests reproduce the races against the real LocalLedger +
PaymentEscrow (no mocks) and assert the money is paid AT MOST ONCE and
the escrow wallet never goes negative. The fix: a per-job_id asyncio.Lock
held across the whole release/refund/split body, so the loser re-evaluates
after the winner has fully committed its terminal status.
"""
from __future__ import annotations

import asyncio

import pytest

from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.payment_escrow import PaymentEscrow, EscrowStatus


@pytest.fixture
async def ledger():
    led = LocalLedger(":memory:")
    await led.initialize()
    await led.create_wallet("alice", "Alice")
    await led.create_wallet("bob", "Bob")
    await led.credit(
        wallet_id="alice", amount=100.0,
        tx_type=TransactionType.WELCOME_GRANT, description="seed",
    )
    yield led
    if led._db is not None:
        await led._db.close()


@pytest.fixture
async def escrow(ledger):
    return PaymentEscrow(ledger=ledger, node_id="alice")


async def _bal(ledger, w):
    return await ledger.get_balance(w)


@pytest.mark.asyncio
async def test_concurrent_double_release_pays_provider_once(ledger, escrow):
    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    ew = f"escrow-{entry.escrow_id}"

    await asyncio.gather(
        escrow.release_escrow("job-1", "bob"),
        escrow.release_escrow("job-1", "bob"),
        return_exceptions=True,
    )

    # Provider paid exactly once; escrow wallet never negative (no mint).
    assert await _bal(ledger, "bob") == 10.0
    assert await _bal(ledger, ew) >= 0.0
    assert await _bal(ledger, ew) == 0.0
    # Total FTNS conserved at the seeded 100 with no over-payout.
    assert await _bal(ledger, "alice") == 90.0


@pytest.mark.asyncio
async def test_concurrent_release_and_refund_single_winner(ledger, escrow):
    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    ew = f"escrow-{entry.escrow_id}"

    await asyncio.gather(
        escrow.release_escrow("job-1", "bob"),
        escrow.refund_escrow("job-1", reason="race"),
        return_exceptions=True,
    )

    bob = await _bal(ledger, "bob")
    alice = await _bal(ledger, "alice")
    # Exactly one terminal outcome — NOT both (which is the bug:
    # bob=10 AND alice=100, escrow=-10).
    released = (bob == 10.0 and alice == 90.0)
    refunded = (bob == 0.0 and alice == 100.0)
    assert released or refunded, f"both-won race: bob={bob} alice={alice}"
    assert await _bal(ledger, ew) >= 0.0


@pytest.mark.asyncio
async def test_concurrent_split_release_pays_recipients_once(ledger, escrow):
    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    ew = f"escrow-{entry.escrow_id}"

    await asyncio.gather(
        escrow.release_escrow_split("job-1", [("bob", 10.0)]),
        escrow.release_escrow_split("job-1", [("bob", 10.0)]),
        return_exceptions=True,
    )

    assert await _bal(ledger, "bob") == 10.0
    assert await _bal(ledger, ew) >= 0.0
    assert await _bal(ledger, ew) == 0.0


@pytest.mark.asyncio
async def test_serial_release_still_works_and_is_idempotent(ledger, escrow):
    """Regression: the lock must not break the normal serial path."""
    entry = await escrow.create_escrow(
        job_id="job-1", amount=10.0, requester_id="alice",
    )
    tx = await escrow.release_escrow("job-1", "bob")
    assert tx is not None
    assert entry.status == EscrowStatus.RELEASED
    assert await _bal(ledger, "bob") == 10.0
    # second release is a no-op (already RELEASED)
    again = await escrow.release_escrow("job-1", "bob")
    assert again is None
    assert await _bal(ledger, "bob") == 10.0
