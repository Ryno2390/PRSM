"""Sprint 549 — CompensationDistributorWatcher persistent event dedup.

Sibling bug to sprint 544. The watcher's existing `state_store`
persists ``last_processed_block`` AFTER the for-loop dispatches every
event in the tick's window. If the daemon crashes between callback
dispatch and the post-loop persist, restart re-scans the same window
and re-dispatches every event the previous run already handled.

Pre-sprint impact (CompensationDistributorWatcher specifically):

  - ``distribution_log.append(...)`` writes a DUPLICATE row for each
    re-emitted Distributed event — the financial audit trail diverges
    from the on-chain truth (one chain event → two log rows).
  - ``webhook_deliverer.deliver(...)`` fires the
    ``distribution.distributed`` webhook a second time — external
    consumers without their own dedup re-process the distribution.

Fix (mirrors sprint 544's ``InboundCheckpointStore.credited_deposits``
pattern): a persistent ``(watcher_key, tx_hash, log_index)`` dedup
store. Watcher consults it BEFORE invoking the callback; marks after
successful dispatch. In-memory state is process-local; restart reads
the persistent store + skips already-processed events.

Honest-scope: KeyDistributionWatcher + StorageSlashingWatcher share
the same shape and same gap — closed as sprint-550 + 551 follow-ons.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── store API ──────────────────────────────────────────────


def test_event_dedup_store_unknown_returns_false(tmp_path):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    assert store.has_processed_event(
        "compensation_distributor", "0xaa" * 16, 0,
    ) is False


def test_event_dedup_store_mark_then_query(tmp_path):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    store.mark_processed_event(
        "compensation_distributor", "0xaa" * 16, 3,
    )
    assert store.has_processed_event(
        "compensation_distributor", "0xaa" * 16, 3,
    ) is True


def test_event_dedup_store_persists_across_instances(tmp_path):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    db = str(tmp_path / "dedup.db")
    EventDedupStore(db).mark_processed_event(
        "compensation_distributor", "0xbb" * 16, 0,
    )

    fresh = EventDedupStore(db)
    assert fresh.has_processed_event(
        "compensation_distributor", "0xbb" * 16, 0,
    ) is True


def test_event_dedup_store_idempotent_mark(tmp_path):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    store.mark_processed_event(
        "compensation_distributor", "0xcc" * 16, 1,
    )
    store.mark_processed_event(  # must not raise
        "compensation_distributor", "0xcc" * 16, 1,
    )
    assert store.has_processed_event(
        "compensation_distributor", "0xcc" * 16, 1,
    ) is True


def test_event_dedup_store_per_watcher_isolation(tmp_path):
    """Same (tx_hash, log_index) for different watcher_keys is
    independent — different watchers can see the same on-chain log
    (different contract event types in the same tx) without
    interfering."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    tx, idx = "0xdd" * 16, 0
    store.mark_processed_event("compensation_distributor", tx, idx)
    assert store.has_processed_event(
        "compensation_distributor", tx, idx,
    ) is True
    assert store.has_processed_event(
        "key_distribution", tx, idx,
    ) is False


def test_event_dedup_store_per_log_index_isolation(tmp_path):
    """Same tx, different log_index → independent dedup. One tx can
    emit multiple Distributed events (contract call dispatching
    several distributions in one block)."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    tx = "0xee" * 16
    store.mark_processed_event("compensation_distributor", tx, 0)
    assert store.has_processed_event(
        "compensation_distributor", tx, 0,
    ) is True
    assert store.has_processed_event(
        "compensation_distributor", tx, 1,
    ) is False


# ── event schema extension ─────────────────────────────────


def test_distributed_event_accepts_tx_hash_and_log_index():
    """DistributedEvent gains Optional[str] tx_hash + Optional[int]
    log_index so the watcher can identify each event for dedup.
    Defaults preserve back-compat (callers building DistributedEvent
    without these stay valid)."""
    from prsm.economy.web3.compensation_distributor import (
        DistributedEvent,
    )

    event = DistributedEvent(
        to_creator=100, to_operator=50, to_grant=25,
    )
    # Default state: identifiers absent.
    assert getattr(event, "tx_hash", None) is None
    assert getattr(event, "log_index", None) is None

    event_with_id = DistributedEvent(
        to_creator=100, to_operator=50, to_grant=25,
        tx_hash="0xab" * 32, log_index=3,
    )
    assert event_with_id.tx_hash == "0xab" * 32
    assert event_with_id.log_index == 3


def test_get_distributed_events_populates_tx_identifiers():
    """The client decoder threads tx_hash + log_index from the raw
    log into the DistributedEvent — the watcher needs them for
    dedup."""
    from prsm.economy.web3.compensation_distributor import (
        CompensationDistributorClient,
    )

    # Build a stub log with the canonical web3.py shape.
    class _Args:
        def __getitem__(self, k):
            return {
                "toCreator": 100, "toOperator": 50, "toGrant": 25,
            }[k]

    class _Log:
        pass

    raw = _Log()
    raw.args = _Args()
    raw.transactionHash = bytes.fromhex("aa" * 32)
    raw.logIndex = 5
    # web3.py logs are dict-like; events decoder reads ["args"].
    raw.__getitem__ = lambda self_, k: {
        "args": _Args(),
        "transactionHash": raw.transactionHash,
        "logIndex": 5,
    }[k]

    # Stub contract.events.Distributed().get_logs.
    distributed = MagicMock()
    distributed.return_value.get_logs.return_value = [{
        "args": {
            "toCreator": 100, "toOperator": 50, "toGrant": 25,
        },
        "transactionHash": bytes.fromhex("aa" * 32),
        "logIndex": 5,
    }]

    client = CompensationDistributorClient.__new__(
        CompensationDistributorClient,
    )
    client.contract = MagicMock()
    client.contract.events.Distributed = distributed

    events = client.get_distributed_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].to_creator == 100
    assert events[0].tx_hash == "0x" + "aa" * 32
    assert events[0].log_index == 5


# ── watcher integration ───────────────────────────────────


class _StubClient:
    """CompensationDistributorClient surface the watcher uses."""

    def __init__(self, latest_block: int, events):
        self._latest = latest_block
        self._events = events
        self.scan_calls = []

    def latest_block(self) -> int:
        return self._latest

    def get_distributed_events(self, from_block, to_block):
        self.scan_calls.append((from_block, to_block))
        return list(self._events)


@pytest.mark.asyncio
async def test_watcher_does_not_double_dispatch_across_restart(tmp_path):
    """v1 sees event E in blocks 101-200, calls callback, crashes
    before persisting last_processed_block. v2 resumes from
    persisted 100, re-scans 101-200, sees E again. Pre-sprint:
    callback fires AGAIN. Post-sprint: dedup store skips it."""
    from prsm.economy.web3.compensation_distributor import (
        DistributedEvent,
    )
    from prsm.economy.web3.compensation_distributor_watcher import (
        CompensationDistributorWatcher,
    )
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("compensation_distributor", 100)
    dedup_db = str(tmp_path / "dedup.db")

    event = DistributedEvent(
        to_creator=100, to_operator=50, to_grant=25,
        tx_hash="0x" + "ab" * 32, log_index=2,
    )
    client = _StubClient(latest_block=200, events=[event])

    # v1: callback dispatches, mark_processed fires.
    v1_calls: list = []

    async def _cb1(e):
        v1_calls.append(e)

    watcher_v1 = CompensationDistributorWatcher(
        client=client,
        on_distributed=_cb1,
        state_store=state,
        dedup_store=EventDedupStore(dedup_db),
    )
    await watcher_v1.tick()
    assert len(v1_calls) == 1, (
        f"v1 should have dispatched once; got {len(v1_calls)}"
    )

    # Now simulate the crash: rewind state to 100 (the crash-before-
    # persist scenario).
    state.save("compensation_distributor", 100)

    # v2: same client, same dedup db, fresh watcher instance.
    v2_calls: list = []

    async def _cb2(e):
        v2_calls.append(e)

    watcher_v2 = CompensationDistributorWatcher(
        client=client,
        on_distributed=_cb2,
        state_store=state,
        dedup_store=EventDedupStore(dedup_db),
    )
    await watcher_v2.tick()
    assert len(v2_calls) == 0, (
        f"v2 double-dispatched! v2_calls={v2_calls!r}"
    )


@pytest.mark.asyncio
async def test_watcher_dispatches_new_events_normally(tmp_path):
    """Sanity: dedup doesn't accidentally drop NEW events. Sequential
    distinct (tx_hash, log_index) events fire callback exactly once."""
    from prsm.economy.web3.compensation_distributor import (
        DistributedEvent,
    )
    from prsm.economy.web3.compensation_distributor_watcher import (
        CompensationDistributorWatcher,
    )
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("compensation_distributor", 100)
    dedup = EventDedupStore(str(tmp_path / "dedup.db"))

    events = [
        DistributedEvent(
            to_creator=100, to_operator=50, to_grant=25,
            tx_hash="0x" + "11" * 32, log_index=0,
        ),
        DistributedEvent(
            to_creator=200, to_operator=100, to_grant=50,
            tx_hash="0x" + "22" * 32, log_index=0,
        ),
        DistributedEvent(
            to_creator=300, to_operator=150, to_grant=75,
            tx_hash="0x" + "22" * 32, log_index=1,
        ),
    ]
    client = _StubClient(latest_block=200, events=events)

    calls: list = []

    async def _cb(e):
        calls.append(e)

    watcher = CompensationDistributorWatcher(
        client=client,
        on_distributed=_cb,
        state_store=state,
        dedup_store=dedup,
    )
    await watcher.tick()
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_watcher_dedup_kwarg_optional(tmp_path):
    """When dedup_store is None (back-compat default), watcher works
    exactly as pre-sprint (in-memory only, double-dispatch on
    restart is the documented honest-scope behavior the operator
    opts into by not configuring persistence)."""
    from prsm.economy.web3.compensation_distributor import (
        DistributedEvent,
    )
    from prsm.economy.web3.compensation_distributor_watcher import (
        CompensationDistributorWatcher,
    )
    from prsm.economy.web3.last_processed_block_store import (
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("compensation_distributor", 100)
    event = DistributedEvent(
        to_creator=100, to_operator=50, to_grant=25,
        tx_hash="0x" + "ee" * 32, log_index=0,
    )
    client = _StubClient(latest_block=200, events=[event])

    calls: list = []

    async def _cb(e):
        calls.append(e)

    # No dedup_store kwarg.
    watcher = CompensationDistributorWatcher(
        client=client,
        on_distributed=_cb,
        state_store=state,
    )
    await watcher.tick()
    assert len(calls) == 1
