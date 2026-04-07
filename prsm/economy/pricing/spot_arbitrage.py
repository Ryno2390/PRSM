"""
Spot Market Arbitrage
=====================

Monitors network utilization and automatically adjusts compute prices
to balance supply and demand. When job queue is high and acceptance
rate low, prices decrease to attract idle hardware.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketMetrics:
    """Current market state for spot pricing decisions."""
    pending_jobs: int = 0
    active_providers: int = 0
    acceptance_rate: float = 1.0  # ratio of accepted/total jobs
    avg_wait_time_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def utilization(self) -> float:
        """Estimated network utilization from market metrics."""
        if self.active_providers == 0:
            return 0.0
        # Simple model: utilization = pending jobs / available capacity
        capacity = max(self.active_providers * 3, 1)  # ~3 jobs per provider
        return min(self.pending_jobs / capacity, 1.0)


class SpotArbitrage:
    """Monitors market and adjusts spot pricing automatically."""

    def __init__(
        self,
        pricing_engine=None,
        check_interval: float = 60.0,
        low_acceptance_threshold: float = 0.5,
        high_wait_threshold: float = 30.0,
    ):
        self._pricing_engine = pricing_engine
        self.check_interval = check_interval
        self.low_acceptance_threshold = low_acceptance_threshold
        self.high_wait_threshold = high_wait_threshold
        self._metrics_history: list = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def update_metrics(self, metrics: MarketMetrics) -> None:
        """Record current market metrics."""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-50:]

        # Update pricing engine with current utilization
        if self._pricing_engine:
            self._pricing_engine.update_utilization(metrics.utilization)

    def should_lower_prices(self) -> bool:
        """Check if prices should be lowered to attract providers."""
        if not self._metrics_history:
            return False
        latest = self._metrics_history[-1]
        return (
            latest.acceptance_rate < self.low_acceptance_threshold
            or latest.avg_wait_time_seconds > self.high_wait_threshold
        )

    def should_raise_prices(self) -> bool:
        """Check if prices should be raised (high demand, fast acceptance)."""
        if not self._metrics_history:
            return False
        latest = self._metrics_history[-1]
        return latest.utilization > 0.8 and latest.acceptance_rate > 0.9

    def get_recommended_adjustment(self) -> Dict[str, Any]:
        """Get pricing adjustment recommendation."""
        if not self._metrics_history:
            return {"action": "hold", "reason": "No metrics available"}

        latest = self._metrics_history[-1]

        if self.should_lower_prices():
            return {
                "action": "lower",
                "reason": f"Low acceptance rate ({latest.acceptance_rate:.0%}) or high wait time ({latest.avg_wait_time_seconds:.0f}s)",
                "current_utilization": latest.utilization,
                "suggested_multiplier": max(0.5, latest.utilization),
            }
        elif self.should_raise_prices():
            return {
                "action": "raise",
                "reason": f"High demand (utilization {latest.utilization:.0%})",
                "current_utilization": latest.utilization,
                "suggested_multiplier": min(1.25, 1.0 + (latest.utilization - 0.8)),
            }
        else:
            return {
                "action": "hold",
                "reason": "Market in balance",
                "current_utilization": latest.utilization,
            }

    def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _monitor_loop(self) -> None:
        """Periodically check market state and adjust prices."""
        while self._running:
            await asyncio.sleep(self.check_interval)
            try:
                adjustment = self.get_recommended_adjustment()
                if adjustment["action"] != "hold" and self._pricing_engine:
                    suggested = adjustment.get("suggested_multiplier", 1.0)
                    self._pricing_engine.update_utilization(
                        self._metrics_history[-1].utilization if self._metrics_history else 0.5
                    )
                    logger.info(f"Spot arbitrage: {adjustment['action']} — {adjustment['reason']}")
            except Exception as e:
                logger.warning(f"Spot arbitrage error: {e}")
