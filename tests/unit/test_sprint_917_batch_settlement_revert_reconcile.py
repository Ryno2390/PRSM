"""Sprint 917 — batch-settlement in-flight revert reconciliation.

The sibling of the sp916 withdraw gap. BatchSettlementManager.flush broadcasts
net on-chain transfers (operator escrow → payee) that mirror local-ledger
credits already made. sp914 made it re-queue NEVER-broadcast transfers but
leave PENDING (in-flight) ones "for reconciliation" without actually
reconciling — so if a pending settlement tx later REVERTS on-chain, the queue
was already cleared and the owed on-chain payout is silently DROPPED (an
operator-accounting divergence: the local ledger credited the payee, but the
on-chain settlement never landed and is never retried).

sp917 closes it: each pending transfer is tracked in-flight; reconcile_in_flight
polls the receipt and, on a confirmed REVERT, re-queues the transfer for retry
(safe — an ERC-20 transfer revert is atomic, no tokens moved, so a re-queue
cannot double-pay). A confirmed tx is dropped (no re-queue). A still-pending tx
is kept for the next tick; an ambiguous never-confirming tx is dropped after a
bounded number of attempts WITHOUT auto-requeue (re-queuing a tx that might
still land would double-pay — operator reconciles manually).

Scope note: the content_economy / sp911 royalty claim-release-on-revert bug
(the claim is consumed before the tx and not released on revert → a reverted
royalty is permanently un-retryable) was found during this investigation but
is gated off by default + money-sensitive to fix, so it is deferred to its own
adversarially-reviewed sprint (sp918).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prsm.economy.batch_settlement import (
    BatchSettlementManager,
    PendingTransfer,
    SettlementMode,
)

_FROM = "0x" + "a" * 40
_TO = "0x" + "b" * 40


class _FakeEth:
    def __init__(self, receipts):
        self._receipts = receipts   # tx_hash → dict | None | Exception

    def get_transaction_receipt(self, h):
        r = self._receipts.get(h)
        if isinstance(r, Exception):
            raise r
        return r


def _mgr(receipts=None, *, transfer=None):
    led = MagicMock()
    led.w3 = SimpleNamespace(eth=_FakeEth(receipts or {}))
    if transfer is not None:
        led.transfer = transfer
    mgr = BatchSettlementManager(
        ftns_ledger=led, node_id="n1", connected_address=_FROM,
        mode=SettlementMode.MANUAL,
    )
    return mgr


# ── reconcile_in_flight ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reverted_in_flight_is_requeued():
    mgr = _mgr({"0xrev": {"status": 0}})
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xrev")
    out = await mgr.reconcile_in_flight()
    assert out["reverted"] == 1
    # owed payout preserved — re-queued for the next flush
    assert any(p.to_wallet == _TO and p.amount == 5.0 for p in mgr._queue)
    assert mgr._in_flight == []


@pytest.mark.asyncio
async def test_confirmed_in_flight_is_dropped_not_requeued():
    mgr = _mgr({"0xok": {"status": 1, "blockNumber": 9}})
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xok")
    out = await mgr.reconcile_in_flight()
    assert out["confirmed"] == 1
    assert mgr._queue == []          # the settlement landed — no re-queue
    assert mgr._in_flight == []


@pytest.mark.asyncio
async def test_still_pending_in_flight_is_kept():
    mgr = _mgr({"0xpend": None})     # no receipt yet
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xpend")
    out = await mgr.reconcile_in_flight()
    assert out["still_pending"] == 1
    assert len(mgr._in_flight) == 1  # retried next tick
    assert mgr._queue == []


@pytest.mark.asyncio
async def test_ambiguous_dropped_after_max_attempts_without_requeue():
    # A tx that never gets a receipt is AMBIGUOUS (may still land). After a
    # bounded number of attempts, stop tracking it but do NOT auto-requeue —
    # re-queuing a tx that might still confirm would double-pay.
    mgr = _mgr({"0xstuck": None})
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xstuck")
    for _ in range(mgr._max_reconcile_attempts + 2):
        await mgr.reconcile_in_flight()
    assert mgr._in_flight == []
    assert mgr._queue == []           # NOT auto-requeued (double-pay safe)


@pytest.mark.asyncio
async def test_receipt_rpc_error_keeps_in_flight():
    mgr = _mgr({"0xerr": RuntimeError("rpc down")})
    mgr._track_in_flight(_FROM, _TO, 5.0, "0xerr")
    out = await mgr.reconcile_in_flight()
    assert out["still_pending"] == 1  # transient — retry next tick
    assert len(mgr._in_flight) == 1


# ── flush integration: a pending transfer is tracked in-flight ───────────


@pytest.mark.asyncio
async def test_flush_tracks_pending_transfer_in_flight():
    pending_rec = SimpleNamespace(status="pending", tx_hash="0x" + "f" * 64,
                                  block_number=None)
    mgr = _mgr(transfer=AsyncMock(return_value=pending_rec))
    mgr._queue.append(PendingTransfer(
        tx_id="t1", from_wallet=_FROM, to_wallet=_TO, amount=5.0, job_id="j1"))
    await mgr.flush()
    assert len(mgr._in_flight) == 1
    assert mgr._in_flight[0]["tx_hash"] == "0x" + "f" * 64
    assert mgr._in_flight[0]["amount"] == 5.0


@pytest.mark.asyncio
async def test_flush_reconciles_in_flight_then_processes_requeue():
    # A reverted in-flight tx is reconciled at flush start → re-queued → and
    # then broadcast again in the SAME flush.
    confirmed = SimpleNamespace(status="confirmed", tx_hash="0x" + "c" * 64,
                                block_number=1)
    transfer = AsyncMock(return_value=confirmed)
    mgr = _mgr({"0xrev": {"status": 0}}, transfer=transfer)
    mgr._track_in_flight(_FROM, _TO, 7.0, "0xrev")
    await mgr.flush()
    # the reverted payout was re-queued and re-broadcast (confirmed this time)
    transfer.assert_awaited()
    assert mgr._in_flight == []
