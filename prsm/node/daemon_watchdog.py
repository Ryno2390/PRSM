"""DaemonWatchdog — wires daemon-task crash detection to webhooks.

The 6 long-running daemon tasks (escrow cleanup, heartbeat,
compensation scheduler, 3 watchers, job reaper) already have
liveness probes via /health/detailed and Prometheus gauges.
This watchdog adds an active-push half: detect a transition
from running → done (silent crash) and dispatch a webhook
event so operators get paged without polling.

Lifecycle parallels the other daemons:
- watch() → infinite poll loop
- stop() → exits the loop
- Same task_running probe extends cleanly to monitor the
  watchdog itself (recursive observability — the watcher can
  be watched)

Stateless across restarts: state-tracking is per-process. A
node restart resets last-seen state so the watchdog won't
fire on the inherited "this daemon was running before restart"
condition.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


_DEFAULT_INTERVAL_SECONDS = 30.0


# Daemon registry: (subsystem_name, task_attr) tuples.
# Mirrors the /health/detailed _daemon_subsystem helper coverage.
_DAEMON_REGISTRY: Tuple[Tuple[str, str], ...] = (
    ("escrow_cleanup", "_escrow_cleanup_task"),
    ("heartbeat_scheduler", "_heartbeat_scheduler_task"),
    ("compensation_scheduler", "_compensation_scheduler_task"),
    ("key_distribution_watcher", "_key_distribution_watcher_task"),
    ("storage_slashing_watcher", "_storage_slashing_watcher_task"),
    ("compensation_distributor_watcher",
     "_compensation_distributor_watcher_task"),
    ("job_reaper", "_job_reaper_task"),
)


class DaemonWatchdog:
    """Active-push companion to the daemon liveness probes.

    Construct with a Node-like object exposing the daemon task
    attributes + a WebhookDeliverer configured with the operator's
    webhook URL.
    """

    def __init__(
        self,
        *,
        node: Any,
        webhook_deliverer: Any,
        webhook_url: str,
        webhook_secret: Optional[str] = None,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        check_canonical_pins: bool = False,
        canonical_check_fn=None,
    ) -> None:
        if not webhook_url:
            raise ValueError("webhook_url must be non-empty")
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, "
                f"got {interval_seconds}"
            )
        self._node = node
        self._deliverer = webhook_deliverer
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret
        self._interval_seconds = interval_seconds
        self._running = False
        # asyncio.Event for interruptible sleep — set by stop()
        # to wake the watch loop immediately. Constructed lazily
        # in watch() so it binds to the right event loop.
        self._stop_event: Optional[asyncio.Event] = None
        # Track last-seen "alive" state per daemon. None means
        # we haven't observed a wired daemon yet (first poll
        # establishes baseline).
        self._last_alive: Dict[str, Optional[bool]] = {
            name: None for name, _ in _DAEMON_REGISTRY
        }
        # Canonical-pin drift detection (optional).
        # canonical_check_fn returns dict[subsystem_name →
        # (wired_addr, canonical_addr)]. Watchdog detects
        # match-flips and fires canonical.drifted events.
        self._check_canonical_pins = check_canonical_pins
        self._canonical_check_fn = canonical_check_fn
        # Per-subsystem last-seen match state.
        self._last_canonical_match: Dict[str, Optional[bool]] = {}

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    def _is_task_alive(self, task_attr: str) -> Optional[bool]:
        """Return True/False/None: True if task wired and running,
        False if wired but .done(), None if not wired."""
        task = getattr(self._node, task_attr, None)
        if task is None:
            return None
        try:
            return not task.done()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "DaemonWatchdog probe raised for %s: %s",
                task_attr, exc,
            )
            return None

    async def check_once(self) -> List[str]:
        """One sweep. Returns list of daemon names that
        transitioned (either alive→crashed OR crashed→alive)
        since last sweep — both transitions emit events.

        First sweep (prior is None) doesn't fire either way —
        could be inherited from before watchdog started.
        """
        emitted: List[str] = []
        for name, task_attr in _DAEMON_REGISTRY:
            alive = self._is_task_alive(task_attr)
            prior = self._last_alive[name]
            self._last_alive[name] = alive
            if prior is True and alive is False:
                # Crash transition.
                logger.warning(
                    "DaemonWatchdog: %s transitioned to .done() — "
                    "dispatching daemon.crashed", name,
                )
                await self._dispatch(name, "daemon.crashed")
                emitted.append(name)
            elif prior is False and alive is True:
                # Recovery transition — was crashed, now alive
                # again. Tells operators "you can stop paging."
                logger.info(
                    "DaemonWatchdog: %s recovered (alive again) — "
                    "dispatching daemon.recovered", name,
                )
                await self._dispatch(name, "daemon.recovered")
                emitted.append(name)
        # Canonical-pin drift sweep (optional).
        if self._check_canonical_pins and self._canonical_check_fn:
            try:
                pins = self._canonical_check_fn() or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DaemonWatchdog canonical_check_fn raised: %s",
                    exc,
                )
                pins = {}
            for subsys, (wired, canonical) in pins.items():
                match = (
                    wired is not None
                    and canonical is not None
                    and wired.lower() == canonical.lower()
                )
                prior_match = self._last_canonical_match.get(subsys)
                self._last_canonical_match[subsys] = match
                # Drift onset: True → False transition fires
                # canonical.drifted. Recovery: False → True
                # transition fires canonical.recovered (operator
                # close-out signal, parallel to daemon.recovered).
                if prior_match is True and match is False:
                    logger.warning(
                        "DaemonWatchdog: %s canonical pin drifted "
                        "(wired=%s, canonical=%s) — dispatching "
                        "canonical.drifted",
                        subsys, wired, canonical,
                    )
                    await self._dispatch_canonical(
                        subsys, wired, canonical, "canonical.drifted",
                    )
                    emitted.append(f"canonical:{subsys}")
                elif prior_match is False and match is True:
                    logger.info(
                        "DaemonWatchdog: %s canonical pin restored "
                        "(wired=%s) — dispatching canonical.recovered",
                        subsys, wired,
                    )
                    await self._dispatch_canonical(
                        subsys, wired, canonical, "canonical.recovered",
                    )
                    emitted.append(f"canonical:{subsys}")
        return emitted

    async def _dispatch_canonical(
        self,
        subsystem: str,
        wired: Optional[str],
        canonical: Optional[str],
        event: str = "canonical.drifted",
    ) -> None:
        payload = {
            "event": event,
            "node_id": getattr(
                getattr(self._node, "identity", None),
                "node_id",
                "unknown",
            ),
            "subsystem": subsystem,
            "wired": wired,
            "canonical": canonical,
            "timestamp": time.time(),
        }
        try:
            result = await self._deliverer.deliver(
                url=self._webhook_url,
                event=event,
                payload=payload,
                secret=self._webhook_secret,
            )
            if not result.success:
                logger.warning(
                    "DaemonWatchdog: %s delivery failed for "
                    "%s after %d attempts: %s",
                    event, subsystem, result.attempts, result.error,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DaemonWatchdog: %s dispatch raised for %s: %s",
                event, subsystem, exc,
            )

    async def _dispatch(self, daemon_name: str, event: str) -> None:
        """POST the event to the webhook URL. Used for both
        daemon.crashed and daemon.recovered."""
        payload = {
            "event": event,
            "node_id": getattr(
                getattr(self._node, "identity", None),
                "node_id",
                "unknown",
            ),
            "daemon": daemon_name,
            "timestamp": time.time(),
        }
        try:
            result = await self._deliverer.deliver(
                url=self._webhook_url,
                event=event,
                payload=payload,
                secret=self._webhook_secret,
            )
            if not result.success:
                logger.warning(
                    "DaemonWatchdog: webhook delivery failed for "
                    "%s (%s) after %d attempts: %s",
                    daemon_name, event, result.attempts, result.error,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DaemonWatchdog: webhook dispatch raised for "
                "%s (%s): %s", daemon_name, event, exc,
            )

    async def watch(self) -> None:
        """Long-running poll loop. Same lifecycle as the other
        daemons (so it can itself be monitored via task_running
        probe).

        Uses an asyncio.Event for interruptible sleep so stop()
        actually wakes the loop within milliseconds rather than
        waiting up to interval_seconds (which previously caused
        the 5s shutdown timeout in node.stop()).
        """
        self._running = True
        if self._stop_event is None:
            self._stop_event = asyncio.Event()
        self._stop_event.clear()
        while self._running:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_seconds,
                )
                # stop_event fired — exit cleanly without one
                # last check_once() (caller wants out NOW).
                break
            except asyncio.TimeoutError:
                pass  # interval elapsed, run a sweep
            try:
                await self.check_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("DaemonWatchdog loop error: %s", exc)

    async def stop(self) -> None:
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
