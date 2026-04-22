"""Async watcher for EmissionController epoch transitions + Minted events.

Per docs/2026-04-22-phase8-design-plan.md §4.3 + §6 Task 4.

The watcher polls the EmissionClient at a configurable cadence and fires
user-supplied callbacks on:

  * epoch transitions (currentEpoch changes between ticks — the
    EmissionController does NOT emit EpochTransition on-chain because
    epoch is a pure view function; polling is the canonical detection
    mechanism).
  * Minted events (observed via get_logs on a sliding block window).

Failure handling: RPC errors during a tick are logged and swallowed — the
watcher keeps running rather than crashing. Operators alerting on
"watcher silent for >N intervals" is the recommended external liveness
check (plan §5.3).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Union

from prsm.emission.emission_client import EmissionClient, MintEvent


logger = logging.getLogger(__name__)


EpochTransitionCallback = Callable[[int, int, int], Union[None, Awaitable[None]]]
"""Called as `callback(old_epoch, new_epoch, new_rate_wei_per_sec)`."""

MintCallback = Callable[[MintEvent], Union[None, Awaitable[None]]]
"""Called once per observed Minted event."""


class EmissionWatcher:
    """Polls an EmissionClient and fires callbacks on state changes.

    Callbacks may be sync or async. Async callbacks are awaited; a raised
    exception from any callback is logged but does not tear down the
    watcher loop.
    """

    def __init__(
        self,
        client: EmissionClient,
        *,
        on_epoch_transition: Optional[EpochTransitionCallback] = None,
        on_mint: Optional[MintCallback] = None,
        poll_interval_sec: float = 60.0,
    ) -> None:
        self._client = client
        self._on_epoch_transition = on_epoch_transition
        self._on_mint = on_mint
        self._poll_interval = poll_interval_sec
        self._last_epoch: Optional[int] = None
        self._last_event_block: Optional[int] = None
        self._stop_event = asyncio.Event()

    async def run_forever(self) -> None:
        """Run the poll loop until `stop()` is called."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:  # pragma: no cover — defensive
                logger.exception("EmissionWatcher tick failed; continuing")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._poll_interval
                )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stop_event.set()

    async def _tick(self) -> None:
        # --- Epoch transition detection -----------------------------------
        try:
            current_epoch = self._client.current_epoch()
        except Exception:
            logger.exception("current_epoch() RPC failed")
            return

        if self._last_epoch is not None and current_epoch > self._last_epoch:
            try:
                new_rate = self._client.current_epoch_rate_per_sec()
            except Exception:
                logger.exception("current_epoch_rate() RPC failed after transition")
                new_rate = 0
            if self._on_epoch_transition is not None:
                await self._invoke_epoch_cb(
                    self._last_epoch, current_epoch, new_rate
                )
        self._last_epoch = current_epoch

        # --- Minted event stream ------------------------------------------
        if self._on_mint is not None:
            try:
                await self._poll_mint_events()
            except Exception:
                logger.exception("mint-event poll failed")

    async def _poll_mint_events(self) -> None:
        latest = self._client.latest_block()
        if self._last_event_block is None:
            # First tick — start from current tip so we do not replay
            # history on restart. Operators wanting historical backfill
            # should call client.get_minted_events() directly.
            self._last_event_block = latest
            return

        from_block = self._last_event_block + 1
        if from_block > latest:
            return

        events = self._client.get_minted_events(from_block, latest)
        self._last_event_block = latest
        assert self._on_mint is not None
        for evt in events:
            await self._invoke_mint_cb(evt)

    async def _invoke_epoch_cb(
        self, old_epoch: int, new_epoch: int, new_rate: int
    ) -> None:
        assert self._on_epoch_transition is not None
        try:
            result = self._on_epoch_transition(old_epoch, new_epoch, new_rate)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("on_epoch_transition callback raised; continuing")

    async def _invoke_mint_cb(self, evt: MintEvent) -> None:
        assert self._on_mint is not None
        try:
            result = self._on_mint(evt)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("on_mint callback raised; continuing")
