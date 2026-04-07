"""
Dashboard Metrics Aggregator
=============================

Collects and formats Ring 1-10 metrics for Streamlit dashboard display.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RingStatus:
    """Status of a single capability ring."""
    ring_number: int
    name: str
    initialized: bool = False
    healthy: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class DashboardMetrics:
    """Aggregates metrics across all 10 rings for dashboard display."""

    def __init__(self, node=None):
        self._node = node

    def collect_ring_status(self) -> List[RingStatus]:
        """Collect initialization and health status for all rings."""
        if self._node is None:
            return []

        rings = [
            RingStatus(1, "The Sandbox",
                      initialized=True,  # WASM always available
                      healthy=True,
                      metrics={"wasmtime_available": True}),
            RingStatus(2, "The Courier",
                      initialized=getattr(self._node, 'agent_dispatcher', None) is not None,
                      healthy=getattr(self._node, 'agent_executor', None) is not None),
            RingStatus(3, "The Swarm",
                      initialized=getattr(self._node, 'swarm_coordinator', None) is not None,
                      healthy=getattr(self._node, 'swarm_coordinator', None) is not None),
            RingStatus(4, "The Economy",
                      initialized=getattr(self._node, 'pricing_engine', None) is not None,
                      healthy=getattr(self._node, 'prosumer_manager', None) is not None),
            RingStatus(5, "The Brain",
                      initialized=getattr(self._node, 'agent_forge', None) is not None,
                      healthy=getattr(self._node, 'agent_forge', None) is not None,
                      metrics={"traces": len(getattr(self._node, 'agent_forge', None).traces) if getattr(self._node, 'agent_forge', None) else 0}),
            RingStatus(6, "The Polish", initialized=True, healthy=True),
            RingStatus(7, "The Vault",
                      initialized=getattr(self._node, 'confidential_executor', None) is not None),
            RingStatus(8, "The Shield",
                      initialized=getattr(self._node, 'tensor_executor', None) is not None),
            RingStatus(9, "The Mind",
                      initialized=getattr(self._node, 'nwtn_model_service', None) is not None),
            RingStatus(10, "The Fortress",
                      initialized=getattr(self._node, 'integrity_verifier', None) is not None,
                      healthy=getattr(self._node, 'privacy_budget', None) is not None),
        ]

        return rings

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary suitable for API response or dashboard."""
        rings = self.collect_ring_status()
        initialized = sum(1 for r in rings if r.initialized)
        healthy = sum(1 for r in rings if r.healthy)

        # Collect pricing metrics
        pricing_metrics = {}
        if self._node and getattr(self._node, 'pricing_engine', None):
            pricing_metrics = {
                "spot_multiplier": str(self._node.pricing_engine.spot.multiplier),
                "utilization": self._node.pricing_engine.spot.network_utilization,
            }

        # Collect privacy metrics
        privacy_metrics = {}
        if self._node and getattr(self._node, 'privacy_budget', None):
            privacy_metrics = self._node.privacy_budget.get_audit_report()

        # Collect forge metrics
        forge_metrics = {}
        if self._node and getattr(self._node, 'agent_forge', None):
            forge_metrics = {
                "traces_collected": len(self._node.agent_forge.traces),
            }

        return {
            "rings_initialized": initialized,
            "rings_healthy": healthy,
            "rings_total": 10,
            "rings": [
                {"ring": r.ring_number, "name": r.name, "initialized": r.initialized, "healthy": r.healthy}
                for r in rings
            ],
            "pricing": pricing_metrics,
            "privacy": privacy_metrics,
            "forge": forge_metrics,
            "timestamp": time.time(),
        }
