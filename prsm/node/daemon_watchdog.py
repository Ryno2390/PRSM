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
        # Track last-seen "alive" state per daemon. None means
        # we haven't observed a wired daemon yet (first poll
        # establishes baseline).
        self._last_alive: Dict[str, Optional[bool]] = {
            name: None for name, _ in _DAEMON_REGISTRY
        }

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
        transitioned from alive to crashed since last sweep
        (i.e., events emitted)."""
        emitted: List[str] = []
        for name, task_attr in _DAEMON_REGISTRY:
            alive = self._is_task_alive(task_attr)
            prior = self._last_alive[name]
            self._last_alive[name] = alive
            # Detect transition: was alive last sweep, now done.
            # First sweep (prior is None) doesn't fire even if
            # task is .done() — could be inherited from before
            # watchdog started.
            if prior is True and alive is False:
                logger.warning(
                    "DaemonWatchdog: %s transitioned to .done() — "
                    "dispatching webhook event",
                    name,
                )
                await self._dispatch(name)
                emitted.append(name)
        return emitted

    async def _dispatch(self, daemon_name: str) -> None:
        """POST a daemon.crashed event to the webhook URL."""
        payload = {
            "event": "daemon.crashed",
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
                event="daemon.crashed",
                payload=payload,
                secret=self._webhook_secret,
            )
            if not result.success:
                logger.warning(
                    "DaemonWatchdog: webhook delivery failed for "
                    "%s after %d attempts: %s",
                    daemon_name, result.attempts, result.error,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "DaemonWatchdog: webhook dispatch raised for %s: %s",
                daemon_name, exc,
            )

    async def watch(self) -> None:
        """Long-running poll loop. Same lifecycle as the other
        daemons (so it can itself be monitored via task_running
        probe)."""
        self._running = True
        while self._running:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self.check_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("DaemonWatchdog loop error: %s", exc)

    async def stop(self) -> None:
        self._running = False
