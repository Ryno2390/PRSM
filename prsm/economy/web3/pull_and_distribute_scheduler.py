"""Pull-and-distribute scheduler — async daemon for
CompensationDistributorClient.

Closes the final deferred-follow-on item from
``docs/security/EXPLOIT_RESPONSE_PLAYBOOK_ANNEX_2026_05.md`` §6.2.

Per ``contracts/contracts/CompensationDistributor.sol`` §3.5
contract source comments::

    Operator economics depend on this being called frequently enough
    that accrued allowance is drained; monitoring alerts on call-gap
    > 7 days per Phase 8 plan §8.2 + EmissionController design.

This daemon ships the periodic-invocation half of that surface;
alerting is handled implicitly by setting a default cadence
(86400s = 24h) well below the 7-day monitoring threshold. The
constructor REJECTS intervals > 7 days so an operator
misconfiguration cannot silently drift the daemon into a state
where it would itself trigger the call-gap > 7 days alert it is
supposed to prevent.

Structurally near-twin of ``HeartbeatScheduler``. Differences:

  - Default cadence 86400s (24h) vs heartbeat's 900s (15 min).
  - Constructor enforces ``interval_seconds <= 7 days``.
  - Calls ``client.pull_and_distribute()`` not
    ``client.record_heartbeat()``.

Same exception-swallowing contract, same ``success_count`` /
``failure_count`` telemetry, same optional ``on_success(tx_hash)``
callback, same graceful ``stop()``.

Activation pattern::

    from prsm.economy.web3.compensation_distributor import (
        CompensationDistributorClient,
    )
    from prsm.economy.web3.pull_and_distribute_scheduler import (
        PullAndDistributeScheduler,
    )

    client = CompensationDistributorClient(
        rpc_url=..., contract_address=..., private_key=...,
    )
    scheduler = PullAndDistributeScheduler(
        client=client, interval_seconds=86400,  # 24h
    )
    asyncio.create_task(scheduler.run_forever())
    # ... later ...
    await scheduler.stop()
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
"""Called as ``callback(tx_hash_hex)`` after each successful
pull_and_distribute."""


SEVEN_DAYS_SECONDS = 7 * 24 * 60 * 60  # 604_800


class PullAndDistributeScheduler:
    """Periodically calls ``client.pull_and_distribute()``.

    Construction:
        client: CompensationDistributorClient (must have private_key
            set — pull_and_distribute is permissionless on-chain but
            still requires an operator-funded signer for gas).
        interval_seconds: poll cadence. Default 86400s = 24h.
            Constructor rejects values > 7 days.
        on_success: optional callback fired with tx_hash on success.

    The scheduler swallows all exceptions from the client — a failure
    on one tick does not crash the loop. Failure-mode contract
    matches HeartbeatScheduler:

      - BroadcastFailedError: log + counter; next tick retries.
      - OnChainPendingError: log at WARNING (receipt unknown); next
        tick retries (state on-chain stays consistent regardless of
        whether the prior tx landed — duplicate mint+distribute with
        zero balance is a no-op).
      - OnChainRevertedError: log + counter; main on-chain revert
        path is TransferFailed when an FTNS transfer to a pool fails.
      - Unexpected exceptions: log + counter; next tick retries.

    success_count + failure_count are exposed for operator telemetry.
    """

    def __init__(
        self,
        client,
        *,
        interval_seconds: float = 86400.0,
        on_success: Optional[SuccessCallback] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be > 0, got {interval_seconds}"
            )
        if interval_seconds > SEVEN_DAYS_SECONDS:
            raise ValueError(
                f"interval_seconds must be <= 7 days "
                f"({SEVEN_DAYS_SECONDS}s) per CompensationDistributor.sol "
                f"§3.5 monitoring threshold; got {interval_seconds}"
            )
        self._client = client
        self._interval = float(interval_seconds)
        self._on_success = on_success
        self._stop_event = asyncio.Event()
        self.success_count = 0
        self.failure_count = 0

    @property
    def interval_seconds(self) -> float:
        return self._interval

    async def run_forever(self) -> None:
        """Run the pull-and-distribute loop until ``stop()`` is called."""
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
        self._stop_event.set()

    async def tick(self) -> None:
        """Run one pull-and-distribute attempt. Always returns; never
        raises.

        Public for unit testing; production code should call
        ``run_forever()`` instead.
        """
        try:
            tx_hash, _status = self._client.pull_and_distribute()
        except OnChainPendingError as exc:
            self.failure_count += 1
            logger.warning(
                "pull_and_distribute tx pending (receipt unknown): %s; "
                "will retry next tick (state on-chain stays consistent "
                "either way)",
                exc,
            )
            return
        except BroadcastFailedError as exc:
            self.failure_count += 1
            logger.info(
                "pull_and_distribute broadcast failed: %s; will retry "
                "next tick",
                exc,
            )
            return
        except OnChainRevertedError as exc:
            self.failure_count += 1
            logger.warning(
                "pull_and_distribute reverted: %s (main on-chain revert "
                "path is TransferFailed when an FTNS transfer to a pool "
                "fails)",
                exc,
            )
            return
        except Exception:  # pragma: no cover — defensive
            self.failure_count += 1
            logger.exception(
                "pull_and_distribute tick failed with unexpected exception"
            )
            return

        self.success_count += 1
        logger.info("pull_and_distribute ok: %s", tx_hash)

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
