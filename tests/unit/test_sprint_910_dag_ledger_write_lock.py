"""Sprint 910 — DAGLedger connection-wide write lock (money-path robustness).

The money-path review confirmed a HIGH: DAGLedger (the DEFAULT ledger
backend) uses a shared aiosqlite connection and a connection-GLOBAL
`SAVEPOINT balance_check`. The sprint-487 per-wallet lock only serializes
same-wallet writes, so a COMMIT from ANY other wallet's write (a parallel
debit, a credit-only submit, or record_nonce) releases an in-flight
savepoint mid-debit. The debit's UPDATE then commits inside the foreign
transaction, but the original submit_transaction's `RELEASE SAVEPOINT`
raises "no such savepoint" → the call returns an error AFTER the money
already moved → silent no-refund loss on /wallet/withdraw.

sp910 adds a connection-wide write lock taken by submit_transaction
(always — credit-only included) and record_nonce, held from before the
savepoint through the final commit, so no foreign commit can interleave.

This is a concurrency-correctness test: pre-sp910 the cross-wallet
interleave produces "no such savepoint" errors / lost debits; post-sp910
every concurrent write succeeds cleanly and FTNS is conserved.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from prsm.node.dag_ledger import DAGLedger, TransactionType


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
async def ledger(tmp_db):
    led = DAGLedger(db_path=tmp_db, verify_signatures=False)
    await led.initialize()
    yield led
    if led._db is not None:
        await led._db.close()


@pytest.mark.asyncio
async def test_concurrent_cross_wallet_debits_and_nonces_no_savepoint_loss(ledger):
    N = 12
    for i in range(N):
        await ledger.credit(
            wallet_id=f"w{i}", amount=100.0,
            tx_type=TransactionType.WELCOME_GRANT, description="seed",
        )

    async def _debit(i):
        return await ledger.debit(
            wallet_id=f"w{i}", amount=30.0,
            tx_type=TransactionType.COMPUTE_PAYMENT, description=f"d{i}",
        )

    async def _nonce(i):
        return await ledger.record_nonce(f"nonce-{i}", "origin")

    # Interleave debits (each opens the connection-global savepoint) with
    # record_nonce commits + a credit-only submit — the exact foreign-commit
    # pattern that destroyed the savepoint pre-sp910.
    tasks = []
    for i in range(N):
        tasks.append(_debit(i))
        tasks.append(_nonce(i))
    tasks.append(ledger.credit(
        wallet_id="bonus", amount=5.0,
        tx_type=TransactionType.REWARD, description="credit-only",
    ))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"concurrent writes raised (savepoint loss): {errors[:3]}"

    # Each wallet debited EXACTLY once — no silent loss, no double-debit.
    for i in range(N):
        assert await ledger.get_balance(f"w{i}") == pytest.approx(70.0), f"w{i}"

    # Conservation: 12*100 seeded + 5 bonus = 1205; debits moved 12*30=360
    # to "system". Nothing minted/lost.
    total = sum([await ledger.get_balance(f"w{i}") for i in range(N)])
    total += await ledger.get_balance("system")
    total += await ledger.get_balance("bonus")
    assert total == pytest.approx(N * 100.0 + 5.0)
    assert await ledger.get_balance("system") == pytest.approx(N * 30.0)


@pytest.mark.asyncio
async def test_serial_debit_still_works(ledger):
    """Regression: the connection-wide lock must not break the normal path."""
    await ledger.credit(
        wallet_id="alice", amount=50.0,
        tx_type=TransactionType.WELCOME_GRANT, description="seed",
    )
    await ledger.debit(
        wallet_id="alice", amount=20.0,
        tx_type=TransactionType.COMPUTE_PAYMENT, description="pay",
    )
    assert await ledger.get_balance("alice") == pytest.approx(30.0)
    # record_nonce still claims-once under the shared lock.
    assert await ledger.record_nonce("n1", "o") is True
    assert await ledger.record_nonce("n1", "o") is False
