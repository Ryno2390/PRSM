"""Unit tests for prsm.emission.emission_client + prsm.emission.watcher.

Per docs/2026-04-22-phase8-design-plan.md §6 Task 4.

Uses stubs in place of web3 rather than a full eth-tester setup — the
wrapper is thin enough that binding it to web3.py internals in tests just
tests web3.py. What we DO test here is:

  * client method signatures return ints / bools as advertised
  * snapshot() bundles reads consistently
  * get_minted_events parses log shape into MintEvent objects
  * watcher fires callbacks only on epoch change
  * watcher seeds last_event_block on first tick (no historical replay)
  * watcher survives RPC failures mid-loop
  * watcher stop() exits run_forever promptly
  * async callbacks are awaited correctly
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from prsm.emission.emission_client import EmissionClient, MintEvent
from prsm.emission.watcher import EmissionWatcher


# -----------------------------------------------------------------------------
# Fake web3 — just enough for EmissionClient's ABI surface.
# -----------------------------------------------------------------------------


class _Call:
    def __init__(self, value):
        self._v = value

    def call(self):
        return self._v


class _Func:
    """Callable that returns an object with .call() → value."""

    def __init__(self, value):
        self._v = value

    def __call__(self):
        return _Call(self._v)


class _Functions:
    def __init__(self, values: dict):
        self._values = values

    def __getattr__(self, name):
        if name in self._values:
            return _Func(self._values[name])
        raise AttributeError(name)


class _EventProxy:
    def __init__(self, logs: List[dict]):
        self._logs = logs

    def get_logs(self, from_block: int, to_block: int):
        return [
            log
            for log in self._logs
            if from_block <= log["blockNumber"] <= to_block
        ]


class _Events:
    def __init__(self, logs: List[dict]):
        self._logs = logs

    def Minted(self):
        return _EventProxy(self._logs)


class _Contract:
    def __init__(self, values: dict, logs: List[dict]):
        self.functions = _Functions(values)
        self.events = _Events(logs)


class _Eth:
    def __init__(self, contract: _Contract, block_number: int):
        self._contract = contract
        self.block_number = block_number

    def contract(self, address, abi):
        return self._contract


class FakeWeb3:
    """Minimal Web3 substitute. Real Web3.to_checksum_address is a staticmethod
    on Web3, so we import and reuse it for address normalisation."""

    def __init__(self, values: dict, logs: Optional[List[dict]] = None, block_number: int = 100):
        self.eth = _Eth(_Contract(values, logs or []), block_number)

    def set_block_number(self, n: int) -> None:
        self.eth.block_number = n

    def set_value(self, name: str, value) -> None:
        self.eth._contract.functions._values[name] = value

    def set_logs(self, logs: List[dict]) -> None:
        self.eth._contract.events._logs = logs


ADDR = "0x0000000000000000000000000000000000001234"


def _default_values(**overrides):
    v = {
        "currentEpoch": 0,
        "currentEpochRate": 10**18,
        "timeUntilNextHalving": 3600,
        "mintedToDate": 5 * 10**18,
        "mintCap": 1000 * 10**18,
        "paused": False,
        "lastMintTimestamp": 1000,
    }
    v.update(overrides)
    return v


def _build_client(values=None, logs=None, block_number=100):
    w3 = FakeWeb3(values or _default_values(), logs=logs, block_number=block_number)
    return EmissionClient(w3, ADDR), w3


# -----------------------------------------------------------------------------
# EmissionClient scalar reads
# -----------------------------------------------------------------------------


def test_client_current_epoch_returns_int():
    client, _ = _build_client(_default_values(currentEpoch=3))
    assert client.current_epoch() == 3


def test_client_current_epoch_rate_returns_int():
    client, _ = _build_client(_default_values(currentEpochRate=5 * 10**17))
    assert client.current_epoch_rate_per_sec() == 5 * 10**17


def test_client_time_until_next_halving():
    client, _ = _build_client(_default_values(timeUntilNextHalving=7200))
    assert client.time_until_next_halving_sec() == 7200


def test_client_minted_to_date_and_cap():
    client, _ = _build_client(
        _default_values(mintedToDate=123, mintCap=456)
    )
    assert client.minted_to_date_wei() == 123
    assert client.mint_cap_wei() == 456


def test_client_is_paused():
    client, _ = _build_client(_default_values(paused=True))
    assert client.is_paused() is True


def test_client_snapshot_bundles_reads():
    client, _ = _build_client(
        _default_values(
            currentEpoch=2,
            currentEpochRate=10**18,
            timeUntilNextHalving=1234,
            mintedToDate=99,
            mintCap=900,
            paused=False,
            lastMintTimestamp=4242,
        )
    )
    snap = client.snapshot()
    assert snap.current_epoch == 2
    assert snap.current_epoch_rate_per_sec == 10**18
    assert snap.time_until_next_halving_sec == 1234
    assert snap.minted_to_date_wei == 99
    assert snap.mint_cap_wei == 900
    assert snap.is_paused is False
    assert snap.last_mint_timestamp == 4242


# -----------------------------------------------------------------------------
# Event parsing
# -----------------------------------------------------------------------------


def test_get_minted_events_parses_logs():
    logs = [
        {
            "args": {
                "recipient": "0xabc",
                "amount": 10**18,
                "epoch": 0,
                "epochRate": 10**18,
            },
            "blockNumber": 50,
            "transactionHash": bytes.fromhex("aa" * 32),
        },
        {
            "args": {
                "recipient": "0xdef",
                "amount": 5 * 10**17,
                "epoch": 1,
                "epochRate": 5 * 10**17,
            },
            "blockNumber": 75,
            "transactionHash": bytes.fromhex("bb" * 32),
        },
    ]
    client, _ = _build_client(logs=logs)
    events = client.get_minted_events(0, 100)
    assert len(events) == 2
    assert events[0].recipient == "0xabc"
    assert events[0].amount_wei == 10**18
    assert events[0].block_number == 50
    assert events[1].epoch == 1


def test_get_minted_events_empty_range_returns_empty_list():
    client, _ = _build_client()
    assert client.get_minted_events(100, 50) == []


# -----------------------------------------------------------------------------
# EmissionWatcher
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watcher_fires_on_epoch_transition():
    client, w3 = _build_client(_default_values(currentEpoch=0))
    seen = []

    watcher = EmissionWatcher(
        client, on_epoch_transition=lambda o, n, r: seen.append((o, n, r))
    )
    await watcher._tick()  # seed last_epoch=0
    w3.set_value("currentEpoch", 1)
    w3.set_value("currentEpochRate", 5 * 10**17)
    await watcher._tick()

    assert seen == [(0, 1, 5 * 10**17)]


@pytest.mark.asyncio
async def test_watcher_does_not_fire_when_epoch_stable():
    client, _ = _build_client(_default_values(currentEpoch=0))
    seen = []

    watcher = EmissionWatcher(
        client, on_epoch_transition=lambda *a: seen.append(a)
    )
    await watcher._tick()
    await watcher._tick()
    await watcher._tick()

    assert seen == []


@pytest.mark.asyncio
async def test_watcher_stop_exits_run_forever_promptly():
    client, _ = _build_client()
    watcher = EmissionWatcher(client, poll_interval_sec=10.0)
    task = asyncio.create_task(watcher.run_forever())
    await asyncio.sleep(0.01)
    await watcher.stop()
    # stop() should unblock the wait_for timeout immediately.
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.asyncio
async def test_watcher_survives_rpc_failure_in_current_epoch():
    """If current_epoch() raises, the tick exits early but the watcher
    state is unchanged and subsequent ticks recover."""
    client, w3 = _build_client(_default_values(currentEpoch=0))
    watcher = EmissionWatcher(client)

    await watcher._tick()  # seed last_epoch=0

    # Force the next read to raise, then recover.
    original = client.current_epoch
    fail_next = [True]

    def flaky():
        if fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("RPC transient")
        return original()

    client.current_epoch = flaky  # type: ignore[method-assign]
    await watcher._tick()  # swallows the exception
    assert watcher._last_epoch == 0  # state preserved

    await watcher._tick()  # recovers, current=0, no transition
    assert watcher._last_epoch == 0


@pytest.mark.asyncio
async def test_watcher_seeds_last_event_block_on_first_tick():
    """Restart must NOT replay historical Minted events."""
    logs = [
        {
            "args": {
                "recipient": "0xabc",
                "amount": 10**18,
                "epoch": 0,
                "epochRate": 10**18,
            },
            "blockNumber": 50,
            "transactionHash": bytes.fromhex("aa" * 32),
        },
    ]
    client, w3 = _build_client(logs=logs, block_number=100)
    seen = []

    watcher = EmissionWatcher(client, on_mint=lambda e: seen.append(e))
    await watcher._tick()
    # Historical event at block 50 is below the seeded block (100); watcher
    # must not replay it.
    assert seen == []


@pytest.mark.asyncio
async def test_watcher_fires_on_new_mint_event():
    client, w3 = _build_client(block_number=100)
    seen: list[MintEvent] = []
    watcher = EmissionWatcher(client, on_mint=lambda e: seen.append(e))

    await watcher._tick()  # seed last_event_block=100

    new_log = {
        "args": {
            "recipient": "0xabc",
            "amount": 2 * 10**18,
            "epoch": 0,
            "epochRate": 10**18,
        },
        "blockNumber": 105,
        "transactionHash": bytes.fromhex("cc" * 32),
    }
    w3.set_logs([new_log])
    w3.set_block_number(110)

    await watcher._tick()

    assert len(seen) == 1
    assert seen[0].recipient == "0xabc"
    assert seen[0].amount_wei == 2 * 10**18
    assert seen[0].block_number == 105


@pytest.mark.asyncio
async def test_watcher_awaits_async_callback():
    client, w3 = _build_client(_default_values(currentEpoch=0))
    seen = []

    async def async_cb(old, new, rate):
        await asyncio.sleep(0)
        seen.append((old, new, rate))

    watcher = EmissionWatcher(client, on_epoch_transition=async_cb)
    await watcher._tick()  # seed last_epoch=0
    w3.set_value("currentEpoch", 2)
    await watcher._tick()

    assert seen == [(0, 2, 10**18)]


@pytest.mark.asyncio
async def test_watcher_callback_exception_does_not_crash_loop():
    client, w3 = _build_client(_default_values(currentEpoch=0))
    invocations = [0]

    def bad_cb(old, new, rate):
        invocations[0] += 1
        raise RuntimeError("boom")

    watcher = EmissionWatcher(client, on_epoch_transition=bad_cb)
    await watcher._tick()  # seed

    w3.set_value("currentEpoch", 1)
    await watcher._tick()  # fires, raises, swallowed

    w3.set_value("currentEpoch", 2)
    await watcher._tick()  # fires again, raises, swallowed

    assert invocations[0] == 2
