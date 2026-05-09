"""RPC-side address-filter on watcher event subscriptions.

Closes the deferred follow-on from audit-prep §7.22 honest-scope:
"today's watchers fire callbacks for ALL events in [from_block,
to_block]; per-address filtering happens in the callback. RPC-side
`argument_filters` is a future enhancement."

This sprint extends the 2 indexed-arg-bearing watchers
(KeyDistributionWatcher + StorageSlashingWatcher) with an
`event_filters` kwarg that propagates to web3.py's
`event.get_logs(argument_filters=...)` RPC-side filtering.

Operational impact:
  - Storage providers monitoring only own provider address: receive
    only their own slashing events, not fleet-wide
  - Tier C publishers monitoring own publisher address: receive
    only their own KeyDistribution events
  - RPC bandwidth ↓; callback invocations ↓; log noise ↓

CompensationDistributor.Distributed has NO indexed args, so the
watcher does not gain an event_filters kwarg — there's nothing
addressable to filter on.
"""
from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.compensation_distributor import DistributedEvent
from prsm.economy.web3.compensation_distributor_watcher import (
    CompensationDistributorWatcher,
)
from prsm.economy.web3.key_distribution import KeyReleasedEvent
from prsm.economy.web3.key_distribution_watcher import (
    KeyDistributionWatcher,
)
from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent,
    HeartbeatRecordedEvent,
)
from prsm.economy.web3.storage_slashing_watcher import (
    StorageSlashingWatcher,
)


# ──────────────────────────────────────────────────────────────────────
# Stub clients that capture argument_filters per get_*_events call
# ──────────────────────────────────────────────────────────────────────


class _CaptureClient:
    """Generic capture stub: records every get_*_events call shape
    so tests can verify filters propagated correctly."""

    def __init__(self, latest_block: int = 100):
        self._latest_block = latest_block
        self.calls: List[dict] = []

    def latest_block(self):
        return self._latest_block

    def advance_to(self, block):
        self._latest_block = block

    def _record(self, name, from_block, to_block, **kwargs):
        self.calls.append({
            "event": name,
            "from_block": from_block,
            "to_block": to_block,
            **kwargs,
        })


class _KeyDistributionCaptureClient(_CaptureClient):
    def get_key_released_events(self, from_block, to_block,
                                argument_filters=None):
        self._record("released", from_block, to_block,
                     argument_filters=argument_filters)
        return []

    def get_key_deposited_events(self, from_block, to_block,
                                 argument_filters=None):
        self._record("deposited", from_block, to_block,
                     argument_filters=argument_filters)
        return []

    def get_key_deauthorized_events(self, from_block, to_block,
                                    argument_filters=None):
        self._record("deauthorized", from_block, to_block,
                     argument_filters=argument_filters)
        return []


class _StorageSlashingCaptureClient(_CaptureClient):
    def get_heartbeat_recorded_events(self, from_block, to_block,
                                      argument_filters=None):
        self._record("recorded", from_block, to_block,
                     argument_filters=argument_filters)
        return []

    def get_proof_failure_slashed_events(self, from_block, to_block,
                                         argument_filters=None):
        self._record("proof", from_block, to_block,
                     argument_filters=argument_filters)
        return []

    def get_heartbeat_missing_slashed_events(self, from_block, to_block,
                                             argument_filters=None):
        self._record("missing", from_block, to_block,
                     argument_filters=argument_filters)
        return []


class _CompensationDistributorCaptureClient(_CaptureClient):
    def get_distributed_events(self, from_block, to_block):
        self._record("distributed", from_block, to_block)
        return []


# ──────────────────────────────────────────────────────────────────────
# KeyDistributionWatcher — event_filters propagation
# ──────────────────────────────────────────────────────────────────────


class TestKeyDistributionWatcherEventFilters:
    @pytest.mark.asyncio
    async def test_no_filters_kwarg_passes_none(self):
        """Backwards-compat: no event_filters kwarg = argument_filters
        passed as None to client (legacy behavior preserved)."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        watcher = KeyDistributionWatcher(
            client=client, on_key_released=cb,
        )
        await watcher.tick()  # baseline 100
        client.advance_to(110)
        await watcher.tick()
        # The released-events call must have used argument_filters=None.
        released_calls = [c for c in client.calls if c["event"] == "released"]
        assert released_calls, "expected at least one released-events poll"
        assert released_calls[-1]["argument_filters"] is None

    @pytest.mark.asyncio
    async def test_per_event_filter_propagates_to_client(self):
        """event_filters['KeyReleased'] must propagate as
        argument_filters to the get_key_released_events call."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        my_recipient = "0x" + "11" * 20
        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            event_filters={
                "KeyReleased": {"recipient": my_recipient},
            },
        )
        await watcher.tick()  # baseline
        client.advance_to(110)
        await watcher.tick()
        released_calls = [c for c in client.calls if c["event"] == "released"]
        assert released_calls
        assert released_calls[-1]["argument_filters"] == {
            "recipient": my_recipient,
        }

    @pytest.mark.asyncio
    async def test_filter_only_for_subscribed_event_types(self):
        """If only one event-type callback is wired + a filter is
        provided for that event, only that event-type is polled."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            event_filters={
                "KeyReleased": {"recipient": "0x" + "11" * 20},
            },
        )
        await watcher.tick()
        client.advance_to(110)
        await watcher.tick()
        events_polled = {c["event"] for c in client.calls}
        assert events_polled == {"released"}

    @pytest.mark.asyncio
    async def test_distinct_filters_per_event_type(self):
        """An operator can have different filters for different event
        types (e.g., monitor own publisher for deposits AND own
        recipient for releases)."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        my_recipient = "0x" + "11" * 20
        my_publisher = "0x" + "22" * 20
        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            on_key_deposited=cb,
            event_filters={
                "KeyReleased": {"recipient": my_recipient},
                "KeyDeposited": {"publisher": my_publisher},
            },
        )
        await watcher.tick()
        client.advance_to(110)
        await watcher.tick()

        released = [c for c in client.calls if c["event"] == "released"]
        deposited = [c for c in client.calls if c["event"] == "deposited"]
        assert released[-1]["argument_filters"] == {"recipient": my_recipient}
        assert deposited[-1]["argument_filters"] == {"publisher": my_publisher}

    @pytest.mark.asyncio
    async def test_partial_filter_dict_unfiltered_event_passes_none(self):
        """If event_filters specifies KeyReleased but not KeyDeposited,
        deposited events poll with argument_filters=None (no filter)
        rather than crashing or applying KeyReleased's filter."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        watcher = KeyDistributionWatcher(
            client=client,
            on_key_released=cb,
            on_key_deposited=cb,
            event_filters={
                "KeyReleased": {"recipient": "0x" + "11" * 20},
                # KeyDeposited intentionally absent.
            },
        )
        await watcher.tick()
        client.advance_to(110)
        await watcher.tick()
        deposited = [c for c in client.calls if c["event"] == "deposited"]
        assert deposited[-1]["argument_filters"] is None


# ──────────────────────────────────────────────────────────────────────
# StorageSlashingWatcher — event_filters propagation
# ──────────────────────────────────────────────────────────────────────


class TestStorageSlashingWatcherEventFilters:
    @pytest.mark.asyncio
    async def test_no_filters_passes_none(self):
        client = _StorageSlashingCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        watcher = StorageSlashingWatcher(
            client=client, on_heartbeat_missing_slashed=cb,
        )
        await watcher.tick()
        client.advance_to(110)
        await watcher.tick()
        missing = [c for c in client.calls if c["event"] == "missing"]
        assert missing
        assert missing[-1]["argument_filters"] is None

    @pytest.mark.asyncio
    async def test_provider_filter_propagates(self):
        """A storage provider monitoring only their own slashing
        events sets `provider` filter; gets RPC-side narrowing."""
        client = _StorageSlashingCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        my_provider = "0x" + "11" * 20
        watcher = StorageSlashingWatcher(
            client=client,
            on_heartbeat_missing_slashed=cb,
            on_proof_failure_slashed=cb,
            event_filters={
                "HeartbeatMissingSlashed": {"provider": my_provider},
                "ProofFailureSlashed": {"provider": my_provider},
            },
        )
        await watcher.tick()
        client.advance_to(110)
        await watcher.tick()

        missing = [c for c in client.calls if c["event"] == "missing"]
        proof = [c for c in client.calls if c["event"] == "proof"]
        assert missing[-1]["argument_filters"] == {"provider": my_provider}
        assert proof[-1]["argument_filters"] == {"provider": my_provider}

    @pytest.mark.asyncio
    async def test_filter_with_address_list_propagates(self):
        """argument_filters values can be lists for OR-style matching
        (web3.py supports this). Watcher must pass the list through
        unmodified."""
        client = _StorageSlashingCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        my_providers = ["0x" + "11" * 20, "0x" + "22" * 20]
        watcher = StorageSlashingWatcher(
            client=client,
            on_heartbeat_recorded=cb,
            event_filters={
                "HeartbeatRecorded": {"provider": my_providers},
            },
        )
        await watcher.tick()
        client.advance_to(110)
        await watcher.tick()
        recorded = [c for c in client.calls if c["event"] == "recorded"]
        assert recorded[-1]["argument_filters"] == {"provider": my_providers}


# ──────────────────────────────────────────────────────────────────────
# CompensationDistributorWatcher — NO event_filters kwarg
# ──────────────────────────────────────────────────────────────────────


class TestCompensationDistributorWatcherNoFilters:
    """The Distributed event has NO indexed arguments. The watcher
    deliberately does NOT expose an event_filters kwarg — there's
    nothing to filter on at the RPC level."""

    def test_constructor_does_not_accept_event_filters_kwarg(self):
        """Defensive design check: passing event_filters to this
        watcher's constructor must raise TypeError. If a future
        engineer adds the kwarg without a real indexed-arg surface
        to filter on, the addition should be deliberate (TypeError
        surfaces it)."""
        client = _CompensationDistributorCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        with pytest.raises(TypeError):
            CompensationDistributorWatcher(
                client=client,
                on_distributed=cb,
                event_filters={"Distributed": {"unused": "field"}},
            )


# ──────────────────────────────────────────────────────────────────────
# Cross-watcher kwarg validation
# ──────────────────────────────────────────────────────────────────────


class TestEventFiltersValidation:
    def test_rejects_non_dict_event_filters(self):
        """event_filters must be None or a dict."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        # Pass a list instead of dict.
        with pytest.raises((TypeError, ValueError)):
            KeyDistributionWatcher(
                client=client,
                on_key_released=cb,
                event_filters=["recipient"],  # type: ignore[arg-type]
            )

    def test_rejects_unknown_event_name_in_filters(self):
        """If the filter dict contains an event name that doesn't
        exist on this watcher, raise — likely a typo."""
        client = _KeyDistributionCaptureClient(latest_block=100)
        async def cb(ev):
            pass
        with pytest.raises(ValueError, match="unknown event"):
            KeyDistributionWatcher(
                client=client,
                on_key_released=cb,
                event_filters={
                    "NotARealEvent": {"recipient": "0x..."},
                },
            )
