"""
Health Monitor
==============

Background loop that checks node health and logs alerts.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors node health and raises alerts."""

    def __init__(self, node=None, check_interval: float = 60.0):
        self._node = node
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check: Dict[str, Any] = {}
        self._alert_callbacks = []

    def on_alert(self, callback):
        self._alert_callbacks.append(callback)

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Health monitor started (interval={self.check_interval}s)")

    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def check_health(self) -> Dict[str, Any]:
        """Run a single health check."""
        checks = {
            "timestamp": time.time(),
            "rings_ok": True,
            "settlement_ok": True,
            "peers_ok": True,
            "alerts": [],
        }

        if self._node is None:
            checks["alerts"].append("No node reference — monitor not connected")
            self._last_check = checks
            self._fire_alerts(checks)
            return checks

        # Check ring initialization
        try:
            from prsm.observability.dashboard_metrics import DashboardMetrics
            metrics = DashboardMetrics(node=self._node)
            summary = metrics.get_summary()
            initialized = summary.get("rings_initialized", 0)
            if initialized < 10:
                checks["alerts"].append(f"Only {initialized}/10 rings initialized")
                checks["rings_ok"] = initialized >= 5  # At least core rings
        except Exception as e:
            checks["alerts"].append(f"Ring check failed: {e}")
            checks["rings_ok"] = False

        # Check peer connectivity
        if hasattr(self._node, 'transport') and self._node.transport:
            peer_count = getattr(self._node.transport, 'peer_count', 0)
            if peer_count == 0:
                checks["alerts"].append("No peers connected (isolated node)")
                checks["peers_ok"] = False
            checks["peer_count"] = peer_count

        # Check privacy budget
        if hasattr(self._node, 'privacy_budget') and self._node.privacy_budget:
            remaining = self._node.privacy_budget.remaining
            if remaining < 10:
                checks["alerts"].append(f"Privacy budget low: {remaining:.1f}\u03b5 remaining")

        self._last_check = checks
        self._fire_alerts(checks)
        return checks

    def _fire_alerts(self, checks: Dict[str, Any]) -> None:
        """Invoke registered alert callbacks if there are alerts."""
        if checks["alerts"]:
            for callback in self._alert_callbacks:
                try:
                    callback(checks)
                except Exception:
                    pass

    @property
    def last_check(self) -> Dict[str, Any]:
        return self._last_check

    async def _monitor_loop(self):
        while self._running:
            try:
                checks = await self.check_health()
                if checks["alerts"]:
                    for alert in checks["alerts"]:
                        logger.warning(f"Health alert: {alert}")
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(self.check_interval)
