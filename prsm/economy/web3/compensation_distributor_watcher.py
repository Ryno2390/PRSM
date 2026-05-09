"""CompensationDistributor event watcher.

Async daemon polling the on-chain CompensationDistributor contract
for Distributed events. Operationally smaller surface than the
other two watchers — only one event class is operationally
meaningful (admin-triggered WeightsScheduled / WeightsActivated /
PoolAddressesUpdated events are visible on Basescan and don't
drive operator-side automation; if an operator wants to react to
weight changes they should monitor Basescan directly).

Use cases:
  - Pool operators reconciling on-chain Distributed amounts against
    their internal accounting.
  - Foundation governance monitoring distribution cadence (call-gap
    > 7 days alert per CompensationDistributor.sol §3.5).
  - Public dashboards showing real-time distribution flow.

Mirrors KeyDistributionWatcher / StorageSlashingWatcher shape.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Union

from prsm.economy.web3.compensation_distributor import DistributedEvent


logger = logging.getLogger(__name__)


DistributedCallback = Callable[
    [DistributedEvent], Union[None, Awaitable[None]],
]


class CompensationDistributorWatcher:
    """Polls a CompensationDistributorClient and fires callbacks on
    each new Distributed event observed.

    Construction:
        client: CompensationDistributorClient instance (must expose
            latest_block / get_distributed_events).
        on_distributed: optional callback. If None, no polling
            happens (saves RPC).
        poll_interval_sec: cadence between polls. Default 30.0.

    First-tick semantics: marks current chain tip as baseline; does
    NOT replay history.

    Failure-mode contract: RPC failures swallowed; last_processed_block
    does NOT advance on RPC error. Callback exceptions swallowed.
    """

    WATCHER_KEY = "compensation_distributor"

    def __init__(
        self,
        client,
        *,
        on_distributed: Optional[DistributedCallback] = None,
        poll_interval_sec: float = 30.0,
        state_store=None,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError(
                f"poll_interval_sec must be > 0, got {poll_interval_sec}"
            )
        self._client = client
        self._on_distributed = on_distributed
        self._poll_interval = float(poll_interval_sec)
        self._state_store = state_store
        self._stop_event = asyncio.Event()
        self.last_processed_block: Optional[int] = None

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval

    async def run_forever(self) -> None:
        self._stop_event.clear()
        while not self._stop_event.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval,
                )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_event.set()

    async def tick(self) -> None:
        try:
            latest = self._client.latest_block()
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: latest_block() RPC failed"
            )
            return

        if self.last_processed_block is None:
            persisted = None
            if self._state_store is not None:
                try:
                    persisted = self._state_store.load(self.WATCHER_KEY)
                except Exception:
                    logger.exception(
                        "CompensationDistributorWatcher: state_store."
                        "load() raised; falling back to chain-tip baseline",
                    )
            if persisted is not None:
                self.last_processed_block = persisted
            else:
                self.last_processed_block = latest
                self._persist_baseline()
                return

        if latest <= self.last_processed_block:
            return

        if self._on_distributed is None:
            # No subscriber — just advance baseline so we don't waste
            # RPC on subsequent ticks.
            self.last_processed_block = latest
            self._persist_baseline()
            return

        from_block = self.last_processed_block + 1
        to_block = latest

        try:
            events = self._client.get_distributed_events(from_block, to_block)
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: get_distributed_events "
                "RPC failed",
            )
            return  # do NOT advance baseline

        for event in events:
            await self._invoke_cb(event)
        self.last_processed_block = to_block
        self._persist_baseline()

    def _persist_baseline(self) -> None:
        if self._state_store is None or self.last_processed_block is None:
            return
        try:
            self._state_store.save(
                self.WATCHER_KEY, self.last_processed_block,
            )
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: state_store.save() "
                "raised for block=%d; will retry on next baseline "
                "advance",
                self.last_processed_block,
            )

    async def _invoke_cb(self, event) -> None:
        assert self._on_distributed is not None
        try:
            result = self._on_distributed(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "CompensationDistributorWatcher: callback raised; "
                "daemon continues"
            )
