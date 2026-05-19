"""Sprint 550 — KeyDistributionWatcher event-dedup persistence.

Follow-on to sprint 549. Same shape:
  - Watcher dispatches each event's callback inside a for-loop.
  - Baseline persists AFTER the for-loop, so crash mid-loop →
    restart re-dispatches every event the previous run handled.

Three event types instead of one (KeyReleased / KeyDeposited /
KeyDeauthorized) but the dedup primitive (EventDedupStore) and
wiring pattern were both proven in sprint 549 — this sprint
extends the 3 event dataclasses + their decoders + threads
``dedup_store`` through the watcher's ``_poll_event_type`` helper.

Why the dedup matters here specifically:
  - KeyReleased fires the chat-key-release path: duplicate emits
    re-run the operator's release-handling logic (re-write to
    local key store, re-fire ``slash.*`` webhooks for monitoring).
  - KeyDeposited tracks publisher-side commits; duplicate logs
    pollute the operator's "what keys did this publisher commit"
    audit view.
  - KeyDeauthorized signals revocation; duplicate emits could
    spuriously re-run revocation cleanup (delete-key-from-cache).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── event schema extension ─────────────────────────────────


def test_key_released_event_accepts_identifiers():
    """``KeyReleasedEvent`` gains Optional tx_hash + log_index
    so the watcher can dedup across restart."""
    from prsm.economy.web3.key_distribution import KeyReleasedEvent

    event = KeyReleasedEvent(
        content_hash=b"\xaa" * 32,
        recipient="0x" + "11" * 20,
        encrypted_key=b"\xbb" * 64,
    )
    assert getattr(event, "tx_hash", None) is None
    assert getattr(event, "log_index", None) is None

    event_with_id = KeyReleasedEvent(
        content_hash=b"\xaa" * 32,
        recipient="0x" + "11" * 20,
        encrypted_key=b"\xbb" * 64,
        tx_hash="0x" + "cc" * 32,
        log_index=2,
    )
    assert event_with_id.tx_hash == "0x" + "cc" * 32
    assert event_with_id.log_index == 2


def test_key_deposited_event_accepts_identifiers():
    from prsm.economy.web3.key_distribution_watcher import (
        KeyDepositedEvent,
    )

    event = KeyDepositedEvent(
        content_hash=b"\xaa" * 32,
        publisher="0x" + "11" * 20,
        royalty="0x" + "22" * 20,
        release_fee_ftns_wei=1000,
    )
    assert getattr(event, "tx_hash", None) is None
    assert getattr(event, "log_index", None) is None

    event_with_id = KeyDepositedEvent(
        content_hash=b"\xaa" * 32,
        publisher="0x" + "11" * 20,
        royalty="0x" + "22" * 20,
        release_fee_ftns_wei=1000,
        tx_hash="0x" + "dd" * 32,
        log_index=7,
    )
    assert event_with_id.tx_hash == "0x" + "dd" * 32
    assert event_with_id.log_index == 7


def test_key_deauthorized_event_accepts_identifiers():
    from prsm.economy.web3.key_distribution_watcher import (
        KeyDeauthorizedEvent,
    )

    event = KeyDeauthorizedEvent(
        content_hash=b"\xaa" * 32,
        publisher="0x" + "11" * 20,
    )
    assert getattr(event, "tx_hash", None) is None

    event_with_id = KeyDeauthorizedEvent(
        content_hash=b"\xaa" * 32,
        publisher="0x" + "11" * 20,
        tx_hash="0x" + "ee" * 32,
        log_index=0,
    )
    assert event_with_id.tx_hash == "0x" + "ee" * 32
    assert event_with_id.log_index == 0


# ── decoder threads identifiers ────────────────────────────


def test_get_key_released_events_populates_identifiers():
    from prsm.economy.web3.key_distribution import (
        KeyDistributionClient,
    )

    raw_log = {
        "args": {
            "contentHash": b"\xaa" * 32,
            "recipient": "0x" + "11" * 20,
            "encryptedKey": b"\xbb" * 64,
        },
        "transactionHash": bytes.fromhex("cc" * 32),
        "logIndex": 2,
    }
    released = MagicMock()
    released.return_value.get_logs.return_value = [raw_log]

    client = KeyDistributionClient.__new__(KeyDistributionClient)
    client.contract = MagicMock()
    client.contract.events.KeyReleased = released

    events = client.get_key_released_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].tx_hash == "0x" + "cc" * 32
    assert events[0].log_index == 2


def test_get_key_deposited_events_populates_identifiers():
    from prsm.economy.web3.key_distribution import (
        KeyDistributionClient,
    )

    raw_log = {
        "args": {
            "contentHash": b"\xaa" * 32,
            "publisher": "0x" + "11" * 20,
            "royalty": "0x" + "22" * 20,
            "releaseFeeFtnsWei": 1000,
        },
        "transactionHash": bytes.fromhex("dd" * 32),
        "logIndex": 7,
    }
    deposited = MagicMock()
    deposited.return_value.get_logs.return_value = [raw_log]

    client = KeyDistributionClient.__new__(KeyDistributionClient)
    client.contract = MagicMock()
    client.contract.events.KeyDeposited = deposited

    events = client.get_key_deposited_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].tx_hash == "0x" + "dd" * 32
    assert events[0].log_index == 7


def test_get_key_deauthorized_events_populates_identifiers():
    from prsm.economy.web3.key_distribution import (
        KeyDistributionClient,
    )

    raw_log = {
        "args": {
            "contentHash": b"\xaa" * 32,
            "publisher": "0x" + "11" * 20,
        },
        "transactionHash": bytes.fromhex("ee" * 32),
        "logIndex": 0,
    }
    deauthorized = MagicMock()
    deauthorized.return_value.get_logs.return_value = [raw_log]

    client = KeyDistributionClient.__new__(KeyDistributionClient)
    client.contract = MagicMock()
    client.contract.events.KeyDeauthorized = deauthorized

    events = client.get_key_deauthorized_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].tx_hash == "0x" + "ee" * 32
    assert events[0].log_index == 0


# ── watcher integration ───────────────────────────────────


class _StubClient:
    def __init__(
        self,
        latest_block: int,
        released=None,
        deposited=None,
        deauthorized=None,
    ):
        self._latest = latest_block
        self._released = list(released or [])
        self._deposited = list(deposited or [])
        self._deauthorized = list(deauthorized or [])

    def latest_block(self):
        return self._latest

    def get_key_released_events(
        self, from_block, to_block, argument_filters=None,
    ):
        return list(self._released)

    def get_key_deposited_events(
        self, from_block, to_block, argument_filters=None,
    ):
        return list(self._deposited)

    def get_key_deauthorized_events(
        self, from_block, to_block, argument_filters=None,
    ):
        return list(self._deauthorized)


@pytest.mark.asyncio
async def test_watcher_does_not_double_dispatch_across_restart(
    tmp_path,
):
    """Crash-between-callback-and-baseline-persist + restart →
    no double-dispatch for any of the 3 event types."""
    from prsm.economy.web3.key_distribution import KeyReleasedEvent
    from prsm.economy.web3.key_distribution_watcher import (
        KeyDistributionWatcher,
        KeyDepositedEvent,
        KeyDeauthorizedEvent,
    )
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("key_distribution", 100)
    dedup_db = str(tmp_path / "dedup.db")

    released = [KeyReleasedEvent(
        content_hash=b"\xaa" * 32,
        recipient="0x" + "11" * 20,
        encrypted_key=b"\xbb" * 64,
        tx_hash="0x" + "01" * 32, log_index=0,
    )]
    deposited = [KeyDepositedEvent(
        content_hash=b"\xcc" * 32,
        publisher="0x" + "22" * 20,
        royalty="0x" + "33" * 20,
        release_fee_ftns_wei=42,
        tx_hash="0x" + "02" * 32, log_index=0,
    )]
    deauthorized = [KeyDeauthorizedEvent(
        content_hash=b"\xdd" * 32,
        publisher="0x" + "44" * 20,
        tx_hash="0x" + "03" * 32, log_index=0,
    )]
    client = _StubClient(
        latest_block=200,
        released=released, deposited=deposited,
        deauthorized=deauthorized,
    )

    v1_released, v1_deposited, v1_deauth = [], [], []
    async def _r(e): v1_released.append(e)
    async def _d(e): v1_deposited.append(e)
    async def _x(e): v1_deauth.append(e)

    watcher_v1 = KeyDistributionWatcher(
        client=client,
        on_key_released=_r,
        on_key_deposited=_d,
        on_key_deauthorized=_x,
        state_store=state,
        dedup_store=EventDedupStore(dedup_db),
    )
    await watcher_v1.tick()
    assert (len(v1_released), len(v1_deposited), len(v1_deauth)) == (
        1, 1, 1,
    )

    # Crash before persist — rewind state to 100.
    state.save("key_distribution", 100)

    v2_released, v2_deposited, v2_deauth = [], [], []
    async def _r2(e): v2_released.append(e)
    async def _d2(e): v2_deposited.append(e)
    async def _x2(e): v2_deauth.append(e)

    watcher_v2 = KeyDistributionWatcher(
        client=client,
        on_key_released=_r2,
        on_key_deposited=_d2,
        on_key_deauthorized=_x2,
        state_store=state,
        dedup_store=EventDedupStore(dedup_db),
    )
    await watcher_v2.tick()
    assert v2_released == [], "KeyReleased re-dispatched"
    assert v2_deposited == [], "KeyDeposited re-dispatched"
    assert v2_deauth == [], "KeyDeauthorized re-dispatched"


@pytest.mark.asyncio
async def test_watcher_dedup_kwarg_optional(tmp_path):
    """dedup_store=None preserves pre-sprint behavior."""
    from prsm.economy.web3.key_distribution import KeyReleasedEvent
    from prsm.economy.web3.key_distribution_watcher import (
        KeyDistributionWatcher,
    )
    from prsm.economy.web3.last_processed_block_store import (
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("key_distribution", 100)
    released = [KeyReleasedEvent(
        content_hash=b"\xaa" * 32,
        recipient="0x" + "11" * 20,
        encrypted_key=b"\xbb" * 64,
        tx_hash="0x" + "0f" * 32, log_index=0,
    )]
    client = _StubClient(latest_block=200, released=released)

    calls = []
    async def _r(e): calls.append(e)

    watcher = KeyDistributionWatcher(
        client=client,
        on_key_released=_r,
        state_store=state,
        # no dedup_store
    )
    await watcher.tick()
    assert len(calls) == 1
