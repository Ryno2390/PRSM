"""KeyDistribution event watcher.

Async daemon that polls the on-chain KeyDistribution contract for
KeyReleased / KeyDeposited / KeyDeauthorized events and fires
user-supplied callbacks. Closes the operationally-meaningful half
of `EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §5.4 detection
scenario: without a watcher, the only way to detect KeyReleased
events is manual Basescan polling — too slow for a P0 surface
where Tier C trust depends on payment-verification correctness.

Mirrors `prsm/emission/watcher.py` (EmissionWatcher) shape:
async asyncio.Event-driven loop, exception-swallowing, optional
callbacks, graceful stop. Only events the operator subscribes to
are polled — saves RPC bandwidth when only one of the three is
of interest.

Activation pattern::

    from prsm.economy.web3.key_distribution import (
        KeyDistributionClient,
    )
    from prsm.economy.web3.key_distribution_watcher import (
        KeyDistributionWatcher,
    )

    client = KeyDistributionClient(rpc_url=..., contract_address=...)

    async def on_release(event):
        # Cross-check against payment-escrow records.
        ...

    watcher = KeyDistributionWatcher(
        client=client,
        on_key_released=on_release,
        poll_interval_sec=30.0,
    )
    asyncio.create_task(watcher.run_forever())
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from prsm.economy.web3.key_distribution import KeyReleasedEvent


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Event dataclasses missing from the client module
# ──────────────────────────────────────────────────────────────────────

# Note: KeyReleasedEvent already lives in
# `prsm/economy/web3/key_distribution.py`. The two below augment it
# so the watcher can decode the full event surface.


def _validate_content_hash(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError(
            f"content_hash must be bytes, got {type(value).__name__}"
        )
    if len(value) != 32:
        raise ValueError(f"content_hash must be 32 bytes, got {len(value)}")
    return bytes(value)


@dataclass(frozen=True)
class KeyDepositedEvent:
    """Decoded ``KeyDeposited(bytes32 indexed contentHash,
    address indexed publisher, address indexed royalty,
    uint256 releaseFeeFtnsWei)``."""
    content_hash: bytes
    publisher: str
    royalty: str
    release_fee_ftns_wei: int

    def __post_init__(self) -> None:
        _validate_content_hash(self.content_hash)
        if not isinstance(self.release_fee_ftns_wei, int) or self.release_fee_ftns_wei < 0:
            raise ValueError(
                f"release_fee_ftns_wei must be non-negative int, "
                f"got {self.release_fee_ftns_wei!r}"
            )

    @classmethod
    def from_decoded_args(cls, args: Dict[str, Any]) -> "KeyDepositedEvent":
        return cls(
            content_hash=bytes(args["contentHash"]),
            publisher=str(args["publisher"]),
            royalty=str(args["royalty"]),
            release_fee_ftns_wei=int(args["releaseFeeFtnsWei"]),
        )


@dataclass(frozen=True)
class KeyDeauthorizedEvent:
    """Decoded ``KeyDeauthorized(bytes32 indexed contentHash,
    address indexed publisher)``."""
    content_hash: bytes
    publisher: str

    def __post_init__(self) -> None:
        _validate_content_hash(self.content_hash)

    @classmethod
    def from_decoded_args(cls, args: Dict[str, Any]) -> "KeyDeauthorizedEvent":
        return cls(
            content_hash=bytes(args["contentHash"]),
            publisher=str(args["publisher"]),
        )


# ──────────────────────────────────────────────────────────────────────
# Callback typing
# ──────────────────────────────────────────────────────────────────────


KeyReleasedCallback = Callable[[KeyReleasedEvent], Union[None, Awaitable[None]]]
KeyDepositedCallback = Callable[[KeyDepositedEvent], Union[None, Awaitable[None]]]
KeyDeauthorizedCallback = Callable[
    [KeyDeauthorizedEvent], Union[None, Awaitable[None]],
]


# ──────────────────────────────────────────────────────────────────────
# Watcher
# ──────────────────────────────────────────────────────────────────────


class KeyDistributionWatcher:
    """Polls a KeyDistributionClient and fires callbacks on each new
    event observed.

    Construction:
        client: KeyDistributionClient instance (must expose
            latest_block / get_key_released_events /
            get_key_deposited_events / get_key_deauthorized_events).
        on_key_released / on_key_deposited / on_key_deauthorized:
            optional async or sync callbacks. If None for a given
            event type, that event is NOT polled (saves RPC).
        poll_interval_sec: cadence between polls. Default 30.0.

    First-tick semantics: marks the current chain tip as the
    baseline; does NOT replay history. Operators wanting historical
    backfill should call `client.get_*_events` directly.

    Failure-mode contract:
      - Per-event-type RPC failures are swallowed; ``last_processed_block``
        does NOT advance on RPC error (so events aren't lost on
        transient failure — next tick retries the same range).
      - Callback exceptions are swallowed; daemon stays alive across
        user-callback bugs.
    """

    def __init__(
        self,
        client,
        *,
        on_key_released: Optional[KeyReleasedCallback] = None,
        on_key_deposited: Optional[KeyDepositedCallback] = None,
        on_key_deauthorized: Optional[KeyDeauthorizedCallback] = None,
        poll_interval_sec: float = 30.0,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError(
                f"poll_interval_sec must be > 0, got {poll_interval_sec}"
            )
        self._client = client
        self._on_released = on_key_released
        self._on_deposited = on_key_deposited
        self._on_deauthorized = on_key_deauthorized
        self._poll_interval = float(poll_interval_sec)
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
        """One poll iteration. Always returns; never raises."""
        try:
            latest = self._client.latest_block()
        except Exception:
            logger.exception(
                "KeyDistributionWatcher: latest_block() RPC failed"
            )
            return

        if self.last_processed_block is None:
            # First tick — set baseline at chain tip; do NOT replay
            # history. Operators wanting backfill should call
            # client.get_*_events directly.
            self.last_processed_block = latest
            return

        if latest <= self.last_processed_block:
            return  # no new blocks

        from_block = self.last_processed_block + 1
        to_block = latest

        # Track per-event-type RPC success — we only advance the
        # baseline if ALL subscribed event types succeed for this
        # range. A partial failure rolls back the advance so the next
        # tick retries the full range. Trade-off: re-emits already-
        # seen events for the successful types on retry; documented
        # as honest-scope. Callback idempotency is the operator's
        # contract.
        all_succeeded = True

        if self._on_released is not None:
            if not await self._poll_event_type(
                "released", from_block, to_block,
                self._client.get_key_released_events, self._on_released,
            ):
                all_succeeded = False

        if self._on_deposited is not None:
            if not await self._poll_event_type(
                "deposited", from_block, to_block,
                self._client.get_key_deposited_events, self._on_deposited,
            ):
                all_succeeded = False

        if self._on_deauthorized is not None:
            if not await self._poll_event_type(
                "deauthorized", from_block, to_block,
                self._client.get_key_deauthorized_events, self._on_deauthorized,
            ):
                all_succeeded = False

        if all_succeeded:
            self.last_processed_block = to_block

    async def _poll_event_type(
        self, name: str, from_block: int, to_block: int, getter, callback,
    ) -> bool:
        try:
            events = getter(from_block, to_block)
        except Exception:
            logger.exception(
                "KeyDistributionWatcher: get_%s_events RPC failed", name,
            )
            return False
        for event in events:
            await self._invoke_cb(callback, event)
        return True

    async def _invoke_cb(self, callback, event) -> None:
        try:
            result = callback(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "KeyDistributionWatcher: callback raised; daemon continues"
            )
