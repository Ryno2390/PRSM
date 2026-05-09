"""StorageSlashing event watcher.

Async daemon polling the on-chain StorageSlashing contract for
HeartbeatRecorded / ProofFailureSlashed / HeartbeatMissingSlashed
events. Closes the operator-monitoring half of
`EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md` §5.3 (anomalous
slashing detection): without a watcher, the only way to detect
slashes is manual Basescan polling.

Operational use cases:
  - Storage providers monitoring for slashing of their own address
    (real-time alert).
  - Foundation council monitoring fleet-wide slashing rate (anomaly
    detection).
  - Dashboards / forensics tooling.

Mirrors KeyDistributionWatcher shape.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Union

from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent,
    HeartbeatRecordedEvent,
    ProofFailureSlashedEvent,
)


logger = logging.getLogger(__name__)


HeartbeatRecordedCallback = Callable[
    [HeartbeatRecordedEvent], Union[None, Awaitable[None]],
]
ProofFailureSlashedCallback = Callable[
    [ProofFailureSlashedEvent], Union[None, Awaitable[None]],
]
HeartbeatMissingSlashedCallback = Callable[
    [HeartbeatMissingSlashedEvent], Union[None, Awaitable[None]],
]


class StorageSlashingWatcher:
    """Polls a StorageSlashingClient and fires callbacks on each new
    event observed.

    Construction:
        client: StorageSlashingClient instance (must expose
            latest_block / get_heartbeat_recorded_events /
            get_proof_failure_slashed_events /
            get_heartbeat_missing_slashed_events).
        on_heartbeat_recorded / on_proof_failure_slashed /
        on_heartbeat_missing_slashed: optional callbacks. If None
        for a given event type, that event is NOT polled.
        poll_interval_sec: cadence between polls. Default 30.0.

    First-tick semantics: marks current chain tip as baseline; does
    NOT replay history.

    Failure-mode contract: per-event-type RPC failures swallowed;
    last_processed_block does NOT advance on RPC error. Callback
    exceptions swallowed.
    """

    WATCHER_KEY = "storage_slashing"

    def __init__(
        self,
        client,
        *,
        on_heartbeat_recorded: Optional[HeartbeatRecordedCallback] = None,
        on_proof_failure_slashed: Optional[ProofFailureSlashedCallback] = None,
        on_heartbeat_missing_slashed: Optional[
            HeartbeatMissingSlashedCallback
        ] = None,
        poll_interval_sec: float = 30.0,
        state_store=None,
    ) -> None:
        if poll_interval_sec <= 0:
            raise ValueError(
                f"poll_interval_sec must be > 0, got {poll_interval_sec}"
            )
        self._client = client
        self._on_recorded = on_heartbeat_recorded
        self._on_proof = on_proof_failure_slashed
        self._on_missing = on_heartbeat_missing_slashed
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
                "StorageSlashingWatcher: latest_block() RPC failed"
            )
            return

        if self.last_processed_block is None:
            persisted = None
            if self._state_store is not None:
                try:
                    persisted = self._state_store.load(self.WATCHER_KEY)
                except Exception:
                    logger.exception(
                        "StorageSlashingWatcher: state_store.load() "
                        "raised; falling back to chain-tip baseline",
                    )
            if persisted is not None:
                self.last_processed_block = persisted
            else:
                self.last_processed_block = latest
                self._persist_baseline()
                return

        if latest <= self.last_processed_block:
            return

        from_block = self.last_processed_block + 1
        to_block = latest

        all_succeeded = True

        if self._on_recorded is not None:
            if not await self._poll_event_type(
                "heartbeat_recorded", from_block, to_block,
                self._client.get_heartbeat_recorded_events, self._on_recorded,
            ):
                all_succeeded = False

        if self._on_proof is not None:
            if not await self._poll_event_type(
                "proof_failure_slashed", from_block, to_block,
                self._client.get_proof_failure_slashed_events, self._on_proof,
            ):
                all_succeeded = False

        if self._on_missing is not None:
            if not await self._poll_event_type(
                "heartbeat_missing_slashed", from_block, to_block,
                self._client.get_heartbeat_missing_slashed_events,
                self._on_missing,
            ):
                all_succeeded = False

        if all_succeeded:
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
                "StorageSlashingWatcher: state_store.save() raised "
                "for block=%d; will retry on next baseline advance",
                self.last_processed_block,
            )

    async def _poll_event_type(
        self, name: str, from_block: int, to_block: int, getter, callback,
    ) -> bool:
        try:
            events = getter(from_block, to_block)
        except Exception:
            logger.exception(
                "StorageSlashingWatcher: get_%s_events RPC failed", name,
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
                "StorageSlashingWatcher: callback raised; daemon continues"
            )
