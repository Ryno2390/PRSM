"""Heartbeat scheduler — async daemon for StorageSlashingClient.

Closes the deferred-follow-on item from
``docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`` §6.2:
storage providers running v1.7.0 had to invoke ``record_heartbeat``
externally (cron, manual, custom service) until a daemon shipped.
Without periodic heartbeats they become vulnerable to permissionless
``slash_for_missing_heartbeat()`` once their grace window elapses.

Mirrors ``prsm/emission/watcher.py`` (EmissionWatcher) — async
asyncio.Event-driven loop, exception-swallowing, optional callback,
graceful stop.

Activation pattern:

    from prsm.economy.web3.storage_slashing import StorageSlashingClient
    from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler

    client = StorageSlashingClient(rpc_url=..., contract_address=...,
                                   private_key=...)
    scheduler = HeartbeatScheduler(client=client,
                                   interval_seconds=900)  # 15 min
    asyncio.create_task(scheduler.run_forever())
    # ... later ...
    await scheduler.stop()

Cadence guidance: choose ``interval_seconds`` to be substantially
shorter than ``client.heartbeat_grace_seconds()`` so a single missed
tick (RPC outage, container restart) does not push the provider
past the grace window. Default ``interval_seconds`` is 900 (15 min) —
appropriate for the contract's MIN_HEARTBEAT_GRACE = 1 hour. Operators
running with a longer grace can lengthen the interval; the daemon
does not auto-tune.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Union

from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
)


logger = logging.getLogger(__name__)


SuccessCallback = Callable[[str], Union[None, Awaitable[None]]]
"""Called as ``callback(tx_hash_hex)`` after each successful heartbeat."""


class HeartbeatScheduler:
    """Periodically calls ``client.record_heartbeat()``.

    Construction:
        client: StorageSlashingClient (must have private_key set).
        interval_seconds: poll cadence. Default 900s = 15 min.
        on_success: optional callback fired with tx_hash on success.

    The scheduler swallows all exceptions from the client — a failure
    on one tick does not crash the loop. Failure modes:

      - BroadcastFailedError: log + counter; next tick retries.
      - OnChainPendingError: log at WARNING (receipt unknown); next
        tick retries (heartbeat is idempotent — another call resets
        the timestamp).
      - OnChainRevertedError: log + counter; next tick retries
        (recordHeartbeat has no real revert path on-chain, so this
        would indicate a deeper problem, but the daemon stays alive).
      - Unexpected exceptions: log + counter; next tick retries.

    success_count + failure_count are exposed for operator telemetry.
    """

    # Default fallback when auto-tune is unavailable. Matches the
    # contract's MIN_HEARTBEAT_GRACE = 1 hour: 3600 / AUTO_TUNE_DIVISOR
    # = 900s.
    DEFAULT_INTERVAL_SECONDS = 900.0

    # Auto-tune ratio: heartbeats per grace window. 4 = "miss up to
    # 3 ticks before grace expires" — defense against transient RPC
    # failures, container bounces, etc.
    AUTO_TUNE_DIVISOR = 4

    # Auto-tune floor: defends against runaway tight cadence if the
    # operator misconfigured grace_seconds to a very small value.
    # 60s is short enough to be useful + long enough to avoid
    # hammering the chain.
    AUTO_TUNE_MIN_INTERVAL_SECONDS = 60.0

    def __init__(
        self,
        client,
        *,
        interval_seconds: Optional[float] = None,
        on_success: Optional[SuccessCallback] = None,
    ) -> None:
        # interval_seconds resolution:
        #   None (default) → auto-tune from client.heartbeat_grace_seconds()
        #     with floor at AUTO_TUNE_MIN_INTERVAL_SECONDS
        #   numeric        → operator-supplied; use directly (must be > 0)
        #
        # Auto-tune fallback to DEFAULT_INTERVAL_SECONDS (900s) when:
        #   - client lacks heartbeat_grace_seconds() method
        #   - method raises (e.g., RPC down at construction)
        #   - method returns non-positive (operator misconfig)
        if interval_seconds is None:
            interval_seconds = self._auto_tune_from_client(client)
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {interval_seconds}"
            )
        self._client = client
        self._interval = float(interval_seconds)
        self._on_success = on_success
        self._stop_event = asyncio.Event()
        self.success_count = 0
        self.failure_count = 0

    @classmethod
    def _auto_tune_from_client(cls, client) -> float:
        """Read client.heartbeat_grace_seconds() and compute the
        proportional interval. Falls back to DEFAULT_INTERVAL_SECONDS
        on any error path."""
        if not hasattr(client, "heartbeat_grace_seconds"):
            logger.warning(
                "HeartbeatScheduler: client lacks heartbeat_grace_seconds() "
                "method; auto-tune unavailable, using "
                "fallback interval=%ss",
                cls.DEFAULT_INTERVAL_SECONDS,
            )
            return cls.DEFAULT_INTERVAL_SECONDS
        try:
            grace = client.heartbeat_grace_seconds()
        except Exception as exc:
            logger.warning(
                "HeartbeatScheduler: heartbeat_grace_seconds() raised "
                "%s: %s; auto-tune fallback to interval=%ss",
                type(exc).__name__, exc, cls.DEFAULT_INTERVAL_SECONDS,
            )
            return cls.DEFAULT_INTERVAL_SECONDS
        if not isinstance(grace, (int, float)) or grace <= 0:
            logger.warning(
                "HeartbeatScheduler: heartbeat_grace_seconds() returned "
                "non-positive value %r; treating as misconfig, "
                "auto-tune fallback to interval=%ss",
                grace, cls.DEFAULT_INTERVAL_SECONDS,
            )
            return cls.DEFAULT_INTERVAL_SECONDS
        tuned = max(
            grace / cls.AUTO_TUNE_DIVISOR,
            cls.AUTO_TUNE_MIN_INTERVAL_SECONDS,
        )
        logger.info(
            "HeartbeatScheduler: auto-tuned interval=%ss from grace=%ss "
            "(ratio 1/%d, floor %ss)",
            tuned, grace, cls.AUTO_TUNE_DIVISOR,
            cls.AUTO_TUNE_MIN_INTERVAL_SECONDS,
        )
        return tuned

    @property
    def interval_seconds(self) -> float:
        return self._interval

    async def run_forever(self) -> None:
        """Run the heartbeat loop until ``stop()`` is called."""
        self._stop_event.clear()
        while not self._stop_event.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._interval,
                )
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        """Signal the loop to exit at the next iteration boundary."""
        self._stop_event.set()

    async def tick(self) -> None:
        """Run one heartbeat attempt. Always returns; never raises.

        Public for unit testing; production code should call
        ``run_forever()`` instead.
        """
        try:
            tx_hash, _status = self._client.record_heartbeat()
        except OnChainPendingError as exc:
            # Concerning but not fatal — receipt unknown means the tx
            # may or may not have landed. Next tick will resubmit.
            self.failure_count += 1
            logger.warning(
                "heartbeat tx pending (receipt unknown): %s; will retry "
                "next tick (heartbeat is idempotent)",
                exc,
            )
            return
        except BroadcastFailedError as exc:
            self.failure_count += 1
            logger.info(
                "heartbeat broadcast failed: %s; will retry next tick", exc,
            )
            return
        except OnChainRevertedError as exc:
            self.failure_count += 1
            logger.warning(
                "heartbeat reverted unexpectedly (recordHeartbeat has no "
                "real revert path): %s",
                exc,
            )
            return
        except Exception:  # pragma: no cover — defensive
            self.failure_count += 1
            logger.exception("heartbeat tick failed with unexpected exception")
            return

        self.success_count += 1
        logger.info("heartbeat ok: %s", tx_hash)

        if self._on_success is not None:
            await self._invoke_success_cb(tx_hash)

    async def _invoke_success_cb(self, tx_hash: str) -> None:
        assert self._on_success is not None
        try:
            result = self._on_success(tx_hash)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception(
                "on_success callback raised; daemon continues"
            )
