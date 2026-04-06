"""
Prosumer Manager
================

Staking-tier lifecycle, job tracking, slashing, and yield estimation
for compute providers in the PRSM network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from prsm.economy.pricing.models import ProsumerTier
from prsm.economy.pricing.engine import PricingEngine


# ---------------------------------------------------------------------------
# ProsumerProfile
# ---------------------------------------------------------------------------


@dataclass
class ProsumerProfile:
    """Mutable profile for a single prosumer (compute provider)."""

    node_id: str
    stake_amount: float = 0.0
    uptime_7d: float = 1.0
    jobs_completed: int = 0
    jobs_failed: int = 0
    total_pcu_provided: float = 0.0
    total_ftns_earned: float = 0.0
    registered_at: float = field(default_factory=time.time)

    # -- derived properties --------------------------------------------------

    @property
    def tier(self) -> ProsumerTier:
        return ProsumerTier.from_stake(self.stake_amount)

    @property
    def reliability(self) -> float:
        total = self.jobs_completed + self.jobs_failed
        if total == 0:
            return 1.0
        return self.jobs_completed / total

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "stake_amount": self.stake_amount,
            "uptime_7d": self.uptime_7d,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "total_pcu_provided": self.total_pcu_provided,
            "total_ftns_earned": self.total_ftns_earned,
            "registered_at": self.registered_at,
            "tier": self.tier.name,
            "reliability": self.reliability,
        }


# ---------------------------------------------------------------------------
# ProsumerManager
# ---------------------------------------------------------------------------


class ProsumerManager:
    """Manages the lifecycle of a single prosumer node."""

    def __init__(self, node_id: str, ledger: Any = None) -> None:
        self.node_id = node_id
        self.ledger = ledger
        self._pricing = PricingEngine()
        self._profile: Optional[ProsumerProfile] = None

    # -- registration --------------------------------------------------------

    async def register(self, stake_amount: float = 0.0) -> ProsumerProfile:
        """Register this node as a prosumer.

        If *stake_amount* > 0 and a ledger is provided, verifies sufficient
        balance before staking.
        """
        if stake_amount > 0 and self.ledger is not None:
            balance = await self.ledger.get_balance(self.node_id)
            if balance < stake_amount:
                raise ValueError(
                    f"Insufficient balance: have {balance}, need {stake_amount}"
                )

        self._profile = ProsumerProfile(
            node_id=self.node_id,
            stake_amount=stake_amount,
        )
        return self._profile

    # -- query ---------------------------------------------------------------

    def get_profile(self) -> Optional[ProsumerProfile]:
        return self._profile

    # -- job tracking --------------------------------------------------------

    def record_job_completed(self, pcu: float = 0.0, ftns_earned: float = 0.0) -> None:
        if self._profile is None:
            raise RuntimeError("Node not registered")
        self._profile.jobs_completed += 1
        self._profile.total_pcu_provided += pcu
        self._profile.total_ftns_earned += ftns_earned

    def record_job_failed(self) -> None:
        if self._profile is None:
            raise RuntimeError("Node not registered")
        self._profile.jobs_failed += 1

    # -- slashing ------------------------------------------------------------

    async def slash(self, reason: str, rate: float = 0.10) -> float:
        """Slash *rate* fraction of the provider's stake.  Returns amount slashed."""
        if self._profile is None:
            raise RuntimeError("Node not registered")
        amount = self._profile.stake_amount * rate
        self._profile.stake_amount -= amount
        return amount

    # -- yield estimation ----------------------------------------------------

    def yield_estimate(
        self,
        hardware_tier: str = "t1",
        tflops: float = 2.0,
        hours_per_day: float = 8.0,
    ) -> Dict[str, Any]:
        """Delegate to PricingEngine.yield_estimate with the profile's tier."""
        tier = self._profile.tier if self._profile else ProsumerTier.CASUAL
        return self._pricing.yield_estimate(
            hardware_tier=hardware_tier,
            tflops=tflops,
            hours_per_day=hours_per_day,
            prosumer_tier=tier,
        )
