"""Sprint 765 — operator-controlled auto-claim of accumulated FTNS rewards.

Pre-765 operators had to manually call `prsm node claim-rewards`
(or hit the API) periodically. For long-running consumer-device
operators (laptop in pool overnight every night) this is friction.

This module ships an `AutoClaimWorker` that:
- Reads env config at startup (threshold + interval)
- Periodically checks accumulated rewards via
  `StakingManager.calculate_rewards`
- When total >= threshold, calls `claim_rewards` + logs

Backward-compat: env unset → worker disabled → no background
activity. Operators must explicitly opt in.

The on-chain side of claim_rewards already handles gas estimation
+ tx signing via the existing staking infrastructure — this
module just adds the scheduling.

Sprint 766 wires this into node.start() as a background task.
Sprint 767 adds CLI inspection. Sprint 768 docs.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoClaimConfig:
    """Frozen config for the auto-claim worker. Hashable so two
    configs from same env produce same hash."""

    threshold_ftns: Decimal
    interval_seconds: float

    @property
    def enabled(self) -> bool:
        """Worker only runs when threshold > 0."""
        return self.threshold_ftns > 0


def resolve_auto_claim_config_from_env() -> AutoClaimConfig:
    """Read `PRSM_AUTO_CLAIM_THRESHOLD_FTNS` (Decimal, default 0
    = disabled) + `PRSM_AUTO_CLAIM_INTERVAL_S` (float seconds,
    default 3600 = 1 hour).

    Returns the parsed config. Non-numeric values fall back to
    safe defaults (disabled / 1hr) with a warning log — daemon
    must not crash on a typo in an opt-in env.
    """
    threshold_raw = os.environ.get(
        "PRSM_AUTO_CLAIM_THRESHOLD_FTNS", "",
    ).strip()
    interval_raw = os.environ.get(
        "PRSM_AUTO_CLAIM_INTERVAL_S", "",
    ).strip()
    # Threshold: 0 = disabled
    threshold = Decimal("0")
    if threshold_raw:
        try:
            threshold = Decimal(threshold_raw)
            if threshold < 0:
                logger.warning(
                    "PRSM_AUTO_CLAIM_THRESHOLD_FTNS=%s is negative; "
                    "disabling auto-claim (set to 0).",
                    threshold_raw,
                )
                threshold = Decimal("0")
        except Exception:
            logger.warning(
                "PRSM_AUTO_CLAIM_THRESHOLD_FTNS=%r is not a valid "
                "Decimal; disabling auto-claim.",
                threshold_raw,
            )
    # Interval: 3600s default
    interval = 3600.0
    if interval_raw:
        try:
            interval = float(interval_raw)
            if interval < 60.0:
                logger.warning(
                    "PRSM_AUTO_CLAIM_INTERVAL_S=%s below 60s "
                    "(claim attempts every minute is wasteful); "
                    "clamping to 60.",
                    interval_raw,
                )
                interval = 60.0
        except ValueError:
            logger.warning(
                "PRSM_AUTO_CLAIM_INTERVAL_S=%r is not a valid "
                "float; defaulting to 3600s.",
                interval_raw,
            )
    return AutoClaimConfig(
        threshold_ftns=threshold, interval_seconds=interval,
    )


class AutoClaimWorker:
    """Background worker that periodically claims accumulated FTNS
    rewards above an operator-configured threshold.

    Construct with a staking_manager + user_id. Call .start() to
    schedule the loop; .stop() cancels it cleanly.

    The worker is safe to construct + start even when disabled —
    the run-loop short-circuits when config.enabled is False.
    """

    def __init__(
        self,
        staking_manager: Any,
        user_id: str,
        config: Optional[AutoClaimConfig] = None,
    ):
        self.staking_manager = staking_manager
        self.user_id = user_id
        self.config = config or resolve_auto_claim_config_from_env()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Track cumulative outcomes for ops visibility.
        self.total_claimed_ftns: Decimal = Decimal("0")
        self.claim_attempts: int = 0
        self.claim_failures: int = 0

    async def start(self) -> None:
        """Schedule the background loop on the current event loop."""
        if self._running or not self.config.enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Cancel + await the background task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run_loop(self) -> None:
        """Periodically check + claim. Failures don't crash the
        worker — they log + increment the failure counter."""
        while self._running:
            try:
                await asyncio.sleep(self.config.interval_seconds)
                await self._maybe_claim()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self.claim_failures += 1
                logger.warning(
                    "AutoClaimWorker iteration failed: %s", exc,
                )

    async def _maybe_claim(self) -> Optional[Decimal]:
        """One iteration: calculate, claim-if-above, log. Returns
        the amount claimed (Decimal) or None if below threshold."""
        if not self.config.enabled:
            return None
        # Calculate accumulated rewards.
        calculations = await self.staking_manager.calculate_rewards(
            self.user_id,
        )
        total_accumulated = sum(
            (calc.reward_amount for calc in calculations),
            Decimal("0"),
        )
        if total_accumulated < self.config.threshold_ftns:
            logger.debug(
                "Auto-claim: accumulated=%s FTNS < threshold=%s; "
                "skipping.",
                total_accumulated, self.config.threshold_ftns,
            )
            return None
        # Above threshold — claim.
        self.claim_attempts += 1
        try:
            claimed = await self.staking_manager.claim_rewards(
                self.user_id,
            )
        except Exception as exc:
            self.claim_failures += 1
            logger.error(
                "Auto-claim attempt failed: %s", exc,
            )
            return None
        self.total_claimed_ftns += claimed
        logger.info(
            "Auto-claim: claimed %s FTNS (cumulative=%s, "
            "attempts=%d, failures=%d)",
            claimed, self.total_claimed_ftns,
            self.claim_attempts, self.claim_failures,
        )
        return claimed
