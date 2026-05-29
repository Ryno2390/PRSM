"""Sprint 878 — operator-controlled auto-sweep of the onramp funnel.

Pre-878, sp857's conversion funnel only advanced when an operator
manually hit POST /wallet/onramp/sweep (or ran
`prsm node onramp-funnel --sweep`). For a production deployment
serving real users, intents would sit at PENDING_SETTLEMENT
indefinitely until someone remembered to sweep — and the sp871
onramp→swap envelope + sp874 completion webhook would never fire.

This module ships a `FunnelAutoSweepWorker` that:
- Reads env config at startup (interval; 0/unset = disabled)
- Periodically calls OnrampFunnel.sweep with the same
  on_confirmed callback the manual endpoint uses (sp871 envelope
  build + sp874 outbound notify)
- Tracks cumulative outcomes (sweeps run, confirmed, expired,
  errors) for ops visibility via the worker's counters

Backward-compat: env unset → disabled → no background activity.
Operators opt in explicitly with PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S.

Mirrors sp765's AutoClaimWorker shape (start/stop/_run_loop,
fail-soft per-iteration, opt-in via env) so node.start() wiring
+ operator mental model are consistent across background workers.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Minimum sweep interval. Each sweep does N×3 Base RPC calls
# (sp862 balance reader per open intent). Sub-60s would hammer
# the RPC for no benefit — settlement takes minutes, not seconds.
_MIN_INTERVAL_S = 60.0
_DEFAULT_INTERVAL_S = 300.0  # 5 min — matches typical settlement lag


@dataclass(frozen=True)
class FunnelAutoSweepConfig:
    interval_seconds: float

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0


def resolve_auto_sweep_config_from_env() -> FunnelAutoSweepConfig:
    """Read `PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S` (float seconds;
    0 or unset = disabled).

    Non-numeric values fall back to disabled with a warning —
    daemon must not crash on a typo in an opt-in env.
    """
    raw = os.environ.get(
        "PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S", "",
    ).strip()
    if not raw:
        return FunnelAutoSweepConfig(interval_seconds=0.0)
    try:
        interval = float(raw)
    except ValueError:
        logger.warning(
            "PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S=%r is not a valid "
            "float; disabling auto-sweep.",
            raw,
        )
        return FunnelAutoSweepConfig(interval_seconds=0.0)
    if interval <= 0:
        return FunnelAutoSweepConfig(interval_seconds=0.0)
    if interval < _MIN_INTERVAL_S:
        logger.warning(
            "PRSM_FUNNEL_AUTO_SWEEP_INTERVAL_S=%s below %ss "
            "(sweeping faster than settlement lag wastes RPC); "
            "clamping to %s.",
            raw, _MIN_INTERVAL_S, _MIN_INTERVAL_S,
        )
        interval = _MIN_INTERVAL_S
    return FunnelAutoSweepConfig(interval_seconds=interval)


class FunnelAutoSweepWorker:
    """Background worker that periodically sweeps the onramp funnel.

    Construct with a callable `sweep_fn` that performs one sweep +
    returns the {checked, confirmed_new, expired_new} summary. The
    node wires sweep_fn to a closure that builds the balance reader
    + on_confirmed callback exactly like the manual sweep endpoint.

    Safe to construct + start even when disabled — the run-loop
    short-circuits when config.enabled is False.
    """

    def __init__(
        self,
        sweep_fn: Any,
        config: Optional[FunnelAutoSweepConfig] = None,
    ):
        self._sweep_fn = sweep_fn
        self.config = config or resolve_auto_sweep_config_from_env()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Ops counters.
        self.sweeps_run: int = 0
        self.total_confirmed: int = 0
        self.total_expired: int = 0
        self.sweep_failures: int = 0
        self.last_sweep_at: float = 0.0

    async def start(self) -> None:
        if self._running or not self.config.enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "FunnelAutoSweepWorker started (interval=%ss)",
            self.config.interval_seconds,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.interval_seconds)
                await self._run_one_sweep()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                self.sweep_failures += 1
                logger.warning(
                    "FunnelAutoSweepWorker iteration failed: %s",
                    exc,
                )

    async def _run_one_sweep(self) -> Optional[dict]:
        """One sweep iteration. sweep_fn may be sync or async;
        we handle both so the node can pass either.

        Sp894 — a SYNCHRONOUS sweep_fn (the node's `_do_sweep`
        closure) runs OnrampFunnel.sweep → sp862 balance reader →
        blocking Base RPC (httpx, 15s timeout) for EACH open intent.
        Running it inline would block the daemon's event loop for up
        to N×3×15s, stalling every concurrent request — a daemon-wide
        liveness DoS if an RPC hangs near its timeout. So a sync
        sweep_fn is offloaded to the default thread-pool executor;
        the loop stays responsive while the sweep blocks in a thread.
        An ASYNC sweep_fn yields on its own I/O, so it's awaited
        cooperatively on the loop (offloading it would defeat that).
        """
        if asyncio.iscoroutinefunction(self._sweep_fn):
            result = await self._sweep_fn()
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._sweep_fn)
            # Defensive: a plain-def fn that returns a coroutine.
            if asyncio.iscoroutine(result):
                result = await result
        self.sweeps_run += 1
        self.last_sweep_at = time.time()
        if isinstance(result, dict):
            self.total_confirmed += int(
                result.get("confirmed_new", 0) or 0,
            )
            self.total_expired += int(
                result.get("expired_new", 0) or 0,
            )
            if result.get("confirmed_new"):
                logger.info(
                    "FunnelAutoSweepWorker: %s intent(s) confirmed "
                    "this sweep", result["confirmed_new"],
                )
        return result if isinstance(result, dict) else None

    def stats(self) -> dict:
        """Ops snapshot for the worker's lifetime."""
        return {
            "enabled": self.config.enabled,
            "interval_seconds": self.config.interval_seconds,
            "running": self._running,
            "sweeps_run": self.sweeps_run,
            "total_confirmed": self.total_confirmed,
            "total_expired": self.total_expired,
            "sweep_failures": self.sweep_failures,
            "last_sweep_at": self.last_sweep_at,
        }
