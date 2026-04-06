"""
Pricing Engine
==============

Deterministic compute costing, spot-price multipliers, swarm-job quoting,
and provider yield estimation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from prsm.economy.pricing.models import (
    NETWORK_FEE_RATE,
    CostQuote,
    DataAccessFee,
    PCURate,
    ProsumerTier,
    SpotPriceState,
)

# Default TFLOPS by hardware tier — used to estimate PCU from manifests.
DEFAULT_TFLOPS_BY_TIER: Dict[str, float] = {
    "t1": 2.0,
    "t2": 15.0,
    "t3": 50.0,
    "t4": 100.0,
}


class PricingEngine:
    """Hybrid pricing engine combining deterministic compute pricing with
    network-utilisation-based spot adjustments.
    """

    def __init__(
        self,
        rates: Optional[PCURate] = None,
        network_utilization: float = 0.5,
    ) -> None:
        self.rates = rates or PCURate()
        self.spot = SpotPriceState(network_utilization=network_utilization)

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    def update_utilization(self, utilization: float) -> None:
        """Update the network utilisation (clamped to 0-1)."""
        self.spot.network_utilization = max(0.0, min(1.0, utilization))

    # ------------------------------------------------------------------
    # PCU estimation
    # ------------------------------------------------------------------

    def estimate_pcu(self, manifest: Any) -> float:
        """Estimate PCU from an AgentManifest.

        PCU = tflops_for_tier * execution_seconds / 3600
        """
        tier = getattr(manifest, "min_hardware_tier", "t1")
        seconds = getattr(manifest, "max_execution_seconds", 60)
        tflops = DEFAULT_TFLOPS_BY_TIER.get(tier, DEFAULT_TFLOPS_BY_TIER["t1"])
        return tflops * seconds / 3600

    # ------------------------------------------------------------------
    # Compute cost
    # ------------------------------------------------------------------

    def compute_cost(self, pcu: float, tier: str = "t1") -> Decimal:
        """Compute cost in FTNS for the given PCU and hardware tier."""
        rate = self.rates.for_tier(tier)
        return Decimal(str(pcu)) * rate * self.spot.multiplier

    # ------------------------------------------------------------------
    # Provider earning
    # ------------------------------------------------------------------

    def provider_earning(
        self,
        pcu: float,
        tier: str = "t1",
        prosumer_tier: ProsumerTier = ProsumerTier.CASUAL,
    ) -> Decimal:
        """What a provider earns, inclusive of yield boost."""
        base = self.compute_cost(pcu, tier)
        return base * prosumer_tier.yield_boost

    # ------------------------------------------------------------------
    # Swarm job quoting
    # ------------------------------------------------------------------

    def quote_swarm_job(
        self,
        shard_count: int,
        hardware_tier: str = "t1",
        estimated_pcu_per_shard: float = 1.0,
        data_fee: Optional[DataAccessFee] = None,
    ) -> CostQuote:
        """Build a full CostQuote for a swarm job."""
        shard_breakdown: List[Dict[str, Any]] = []
        total_compute = Decimal("0")

        for i in range(shard_count):
            cost = self.compute_cost(estimated_pcu_per_shard, hardware_tier)
            total_compute += cost
            shard_breakdown.append(
                {
                    "shard_index": i,
                    "pcu": estimated_pcu_per_shard,
                    "cost": str(cost),
                }
            )

        data_cost = (
            data_fee.total_for_shards(shard_count)
            if data_fee is not None
            else Decimal("0")
        )

        subtotal = total_compute + data_cost
        network_fee = subtotal * NETWORK_FEE_RATE

        return CostQuote(
            compute_cost=total_compute,
            data_cost=data_cost,
            network_fee=network_fee,
            shard_breakdown=shard_breakdown,
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # Yield estimation
    # ------------------------------------------------------------------

    def yield_estimate(
        self,
        hardware_tier: str = "t1",
        tflops: float = 2.0,
        hours_per_day: float = 8.0,
        prosumer_tier: ProsumerTier = ProsumerTier.CASUAL,
    ) -> Dict[str, Any]:
        """Estimate daily and monthly FTNS earnings for a provider."""
        pcu_per_hour = tflops  # 1 PCU ≈ 1 TFLOP-hour
        daily_pcu = pcu_per_hour * hours_per_day
        daily_ftns = self.provider_earning(daily_pcu, hardware_tier, prosumer_tier)
        monthly_ftns = daily_ftns * Decimal("30")

        return {
            "hardware_tier": hardware_tier,
            "tflops": tflops,
            "hours_per_day": hours_per_day,
            "prosumer_tier": prosumer_tier.name,
            "daily_pcu": daily_pcu,
            "daily_ftns": str(daily_ftns),
            "monthly_ftns": str(monthly_ftns),
        }
