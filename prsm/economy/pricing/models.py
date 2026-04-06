"""
Pricing Data Models
===================

PCURate, CostQuote, SpotPriceState, ProsumerTier, DataAccessFee — the core
value types for the hybrid pricing engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NETWORK_FEE_RATE = Decimal("0.05")
BULK_DISCOUNT_THRESHOLD = 10

# ---------------------------------------------------------------------------
# PCURate — per-tier compute rates
# ---------------------------------------------------------------------------


@dataclass
class PCURate:
    """PRSM Compute Unit rates for each hardware tier."""

    t1_rate: Decimal = Decimal("0.001")
    t2_rate: Decimal = Decimal("0.005")
    t3_rate: Decimal = Decimal("0.02")
    t4_rate: Decimal = Decimal("0.10")

    _TIER_MAP = {"t1": "t1_rate", "t2": "t2_rate", "t3": "t3_rate", "t4": "t4_rate"}

    def for_tier(self, tier_str: str) -> Decimal:
        """Return rate for the given tier string, defaulting to t1 for unknown."""
        attr = self._TIER_MAP.get(tier_str.lower(), "t1_rate")
        return getattr(self, attr)


# ---------------------------------------------------------------------------
# SpotPriceState — network-utilization-based multiplier
# ---------------------------------------------------------------------------


@dataclass
class SpotPriceState:
    """Tracks network utilisation and derives a price multiplier.

    - Below 0.4: linear discount down to 0.5x at utilisation == 0
    - 0.4 – 0.8: flat 1.0x
    - Above 0.8: linear premium up to 1.25x at utilisation == 1.0
    """

    network_utilization: float = 0.5

    @property
    def multiplier(self) -> Decimal:
        u = self.network_utilization
        if u < 0.4:
            # Linear from 0.5 (at u=0) to 1.0 (at u=0.4)
            return Decimal(str(0.5 + (u / 0.4) * 0.5))
        elif u <= 0.8:
            return Decimal("1.0")
        else:
            # Linear from 1.0 (at u=0.8) to 1.25 (at u=1.0)
            return Decimal(str(1.0 + ((u - 0.8) / 0.2) * 0.25))


# ---------------------------------------------------------------------------
# ProsumerTier — staking tiers with yield boosts
# ---------------------------------------------------------------------------


class ProsumerTier(Enum):
    """Provider staking tiers.

    Each member is a (stake_required, yield_boost) tuple.
    """

    CASUAL = (0, Decimal("1.0"))
    PLEDGED = (100, Decimal("1.25"))
    DEDICATED = (1000, Decimal("1.5"))
    SENTINEL = (10000, Decimal("2.0"))

    @property
    def stake_required(self) -> int:
        return self.value[0]

    @property
    def yield_boost(self) -> Decimal:
        return self.value[1]

    @classmethod
    def from_stake(cls, amount: int) -> "ProsumerTier":
        """Return the highest tier the given stake qualifies for."""
        best = cls.CASUAL
        for member in cls:
            if amount >= member.stake_required:
                best = member
        return best


# ---------------------------------------------------------------------------
# DataAccessFee — per-dataset access pricing
# ---------------------------------------------------------------------------


@dataclass
class DataAccessFee:
    """Pricing for accessing a sharded dataset."""

    dataset_id: str = ""
    base_access_fee: Decimal = Decimal("0.01")
    per_shard_fee: Decimal = Decimal("0.002")
    bulk_discount: Decimal = Decimal("0.1")  # 0-1

    def total_for_shards(self, n: int) -> Decimal:
        """Calculate total access fee for *n* shards.

        Bulk discount applies only when *n* >= BULK_DISCOUNT_THRESHOLD.
        """
        discount_factor = (
            (Decimal("1") - self.bulk_discount)
            if n >= BULK_DISCOUNT_THRESHOLD
            else Decimal("1")
        )
        return self.base_access_fee + self.per_shard_fee * Decimal(str(n)) * discount_factor


# ---------------------------------------------------------------------------
# CostQuote — full cost breakdown for a job
# ---------------------------------------------------------------------------


@dataclass
class CostQuote:
    """Itemised cost quote returned by the pricing engine."""

    compute_cost: Decimal = Decimal("0")
    data_cost: Decimal = Decimal("0")
    network_fee: Decimal = Decimal("0")
    shard_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.95
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return self.compute_cost + self.data_cost + self.network_fee

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compute_cost": str(self.compute_cost),
            "data_cost": str(self.data_cost),
            "network_fee": str(self.network_fee),
            "total": str(self.total),
            "shard_breakdown": self.shard_breakdown,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
        }
