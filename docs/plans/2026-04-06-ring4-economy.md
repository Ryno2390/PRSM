# Ring 4 — "The Economy" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Self-sustaining network economics with deterministic compute pricing (the "menu"), value-based data pricing (the "market"), prosumer staking tiers with yield estimation, and NWTN as pricing broker that quotes jobs before execution.

**Architecture:** A `PricingEngine` calculates PCU-based compute costs using Ring 1's hardware tiers. A `ProsumerManager` handles staking tiers and yield estimation, wrapping the existing `StakingManager`. A `CostQuote` dataclass represents the total cost breakdown a researcher sees before committing. These integrate with Ring 3's `SwarmCoordinator` for pre-execution quoting.

**Tech Stack:** Existing PRSM infrastructure (staking manager, batch settlement, compute tiers). No new external dependencies.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `prsm/economy/pricing/__init__.py` | Package exports |
| Create | `prsm/economy/pricing/models.py` | `CostQuote`, `PCURate`, `SpotPriceState` dataclasses |
| Create | `prsm/economy/pricing/engine.py` | `PricingEngine` — PCU rates, spot pricing, job quoting |
| Create | `prsm/economy/prosumer.py` | `ProsumerManager` — staking tiers, yield estimation, slashing |
| Modify | `prsm/node/node.py` | Wire PricingEngine + ProsumerManager into PRSMNode |
| Create | `tests/unit/test_pricing_engine.py` | Pricing + quoting tests |
| Create | `tests/unit/test_prosumer.py` | Prosumer staking + yield tests |
| Create | `tests/integration/test_ring4_economy.py` | End-to-end economy smoke test |

---

### Task 1: Pricing Data Models

**Files:**
- Create: `prsm/economy/pricing/__init__.py`
- Create: `prsm/economy/pricing/models.py`
- Test: `tests/unit/test_pricing_engine.py`

- [ ] **Step 1: Create directory**

```bash
mkdir -p prsm/economy/pricing
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/test_pricing_engine.py`:

```python
"""Tests for pricing engine data models and PCU calculation."""

import pytest
from decimal import Decimal

from prsm.economy.pricing.models import (
    PCURate,
    CostQuote,
    SpotPriceState,
    ProsumerTier,
    DataAccessFee,
)


class TestPCURate:
    def test_tier_rates(self):
        rate = PCURate()
        assert rate.t1_rate == Decimal("0.001")
        assert rate.t2_rate == Decimal("0.005")
        assert rate.t3_rate == Decimal("0.02")
        assert rate.t4_rate == Decimal("0.10")

    def test_rate_for_tier(self):
        rate = PCURate()
        assert rate.for_tier("t1") == Decimal("0.001")
        assert rate.for_tier("t2") == Decimal("0.005")
        assert rate.for_tier("t3") == Decimal("0.02")
        assert rate.for_tier("t4") == Decimal("0.10")

    def test_rate_for_unknown_tier_defaults_to_t1(self):
        rate = PCURate()
        assert rate.for_tier("unknown") == Decimal("0.001")


class TestCostQuote:
    def test_quote_creation(self):
        quote = CostQuote(
            compute_cost=Decimal("2.50"),
            data_cost=Decimal("5.00"),
            network_fee=Decimal("0.375"),
        )
        assert quote.total == Decimal("7.875")

    def test_quote_network_fee_percentage(self):
        quote = CostQuote(
            compute_cost=Decimal("10.0"),
            data_cost=Decimal("0.0"),
            network_fee=Decimal("0.50"),
        )
        assert quote.total == Decimal("10.50")

    def test_quote_breakdown(self):
        quote = CostQuote(
            compute_cost=Decimal("1.00"),
            data_cost=Decimal("3.00"),
            network_fee=Decimal("0.20"),
            shard_breakdown=[
                {"cid": "QmA", "pcu_estimate": 10.0, "cost": Decimal("0.50")},
                {"cid": "QmB", "pcu_estimate": 10.0, "cost": Decimal("0.50")},
            ],
        )
        breakdown = quote.to_dict()
        assert breakdown["total"] == "4.20"
        assert len(breakdown["shard_breakdown"]) == 2

    def test_quote_zero_cost(self):
        quote = CostQuote(
            compute_cost=Decimal("0"),
            data_cost=Decimal("0"),
            network_fee=Decimal("0"),
        )
        assert quote.total == Decimal("0")


class TestSpotPriceState:
    def test_default_multiplier(self):
        state = SpotPriceState()
        assert state.multiplier == Decimal("1.0")

    def test_low_utilization_discount(self):
        state = SpotPriceState(network_utilization=0.3)
        # Below 40% → discount up to 50%
        assert state.multiplier < Decimal("1.0")

    def test_high_utilization_premium(self):
        state = SpotPriceState(network_utilization=0.9)
        # Above 80% → premium up to 25%
        assert state.multiplier > Decimal("1.0")

    def test_normal_utilization_no_change(self):
        state = SpotPriceState(network_utilization=0.6)
        assert state.multiplier == Decimal("1.0")


class TestProsumerTier:
    def test_casual_tier(self):
        tier = ProsumerTier.CASUAL
        assert tier.stake_required == 0
        assert tier.yield_boost == Decimal("1.0")

    def test_dedicated_tier(self):
        tier = ProsumerTier.DEDICATED
        assert tier.stake_required == 1000
        assert tier.yield_boost == Decimal("1.5")

    def test_sentinel_tier(self):
        tier = ProsumerTier.SENTINEL
        assert tier.stake_required == 10000
        assert tier.yield_boost == Decimal("2.0")


class TestDataAccessFee:
    def test_fee_creation(self):
        fee = DataAccessFee(
            dataset_id="nada-nc-2025",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("0.1"),
            bulk_discount=0.1,
        )
        assert fee.total_for_shards(1) == Decimal("5.1")

    def test_bulk_discount(self):
        fee = DataAccessFee(
            dataset_id="ds-1",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("1.0"),
            bulk_discount=0.2,  # 20% discount for 10+ shards
        )
        # 12 shards: base(5) + 12*1.0*0.8 = 5 + 9.6 = 14.6
        total = fee.total_for_shards(12)
        assert total == Decimal("14.6")

    def test_no_bulk_discount_under_threshold(self):
        fee = DataAccessFee(
            dataset_id="ds-1",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("1.0"),
            bulk_discount=0.2,
        )
        # 5 shards (under 10): base(5) + 5*1.0 = 10.0 (no discount)
        total = fee.total_for_shards(5)
        assert total == Decimal("10.0")
```

- [ ] **Step 3: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_pricing_engine.py::TestPCURate -v`
Expected: FAIL

- [ ] **Step 4: Implement models**

Create `prsm/economy/pricing/__init__.py`:

```python
"""
Hybrid Pricing Engine
=====================

Deterministic compute pricing (the "menu") + value-based data pricing (the "market").
Ring 4 of the Sovereign-Edge AI architecture.
"""

from prsm.economy.pricing.models import (
    PCURate,
    CostQuote,
    SpotPriceState,
    ProsumerTier,
    DataAccessFee,
)

__all__ = [
    "PCURate",
    "CostQuote",
    "SpotPriceState",
    "ProsumerTier",
    "DataAccessFee",
]
```

Create `prsm/economy/pricing/models.py`:

```python
"""
Pricing Data Models
===================

PCU rates, cost quotes, spot pricing, prosumer tiers, data access fees.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class PCURate:
    """PRSM Compute Unit rates per hardware tier (FTNS per PCU)."""
    t1_rate: Decimal = Decimal("0.001")   # Mobile, IoT
    t2_rate: Decimal = Decimal("0.005")   # Consoles, mid-range
    t3_rate: Decimal = Decimal("0.02")    # High-end desktops
    t4_rate: Decimal = Decimal("0.10")    # Datacenter GPUs

    def for_tier(self, tier: str) -> Decimal:
        rates = {"t1": self.t1_rate, "t2": self.t2_rate, "t3": self.t3_rate, "t4": self.t4_rate}
        return rates.get(tier, self.t1_rate)


@dataclass
class SpotPriceState:
    """Dynamic spot pricing based on network utilization."""
    network_utilization: float = 0.5  # 0.0 to 1.0

    @property
    def multiplier(self) -> Decimal:
        """Compute spot price multiplier.
        
        Below 40% utilization: discount up to 50% (linearly)
        40-80%: no adjustment (1.0x)
        Above 80%: premium up to 25% (linearly)
        """
        if self.network_utilization < 0.4:
            # Linear discount: 0% util → 0.5x, 40% util → 1.0x
            discount = Decimal(str(1.0 - (0.4 - self.network_utilization) / 0.4 * 0.5))
            return discount
        elif self.network_utilization > 0.8:
            # Linear premium: 80% util → 1.0x, 100% util → 1.25x
            premium = Decimal(str(1.0 + (self.network_utilization - 0.8) / 0.2 * 0.25))
            return premium
        else:
            return Decimal("1.0")


class ProsumerTier(Enum):
    """Prosumer staking tiers with yield boosts."""
    CASUAL = ("casual", 0, Decimal("1.0"), 0)
    PLEDGED = ("pledged", 100, Decimal("1.25"), 8)
    DEDICATED = ("dedicated", 1000, Decimal("1.5"), 20)
    SENTINEL = ("sentinel", 10000, Decimal("2.0"), 24)

    def __init__(self, label: str, stake_required: int, yield_boost: Decimal, min_hours: int):
        self.label = label
        self._stake_required = stake_required
        self._yield_boost = yield_boost
        self.min_hours_per_day = min_hours

    @property
    def stake_required(self) -> int:
        return self._stake_required

    @property
    def yield_boost(self) -> Decimal:
        return self._yield_boost

    @classmethod
    def from_stake(cls, amount: float) -> "ProsumerTier":
        if amount >= 10000:
            return cls.SENTINEL
        elif amount >= 1000:
            return cls.DEDICATED
        elif amount >= 100:
            return cls.PLEDGED
        else:
            return cls.CASUAL


NETWORK_FEE_RATE = Decimal("0.05")  # 5% network fee
BULK_DISCOUNT_THRESHOLD = 10  # Shards before bulk discount applies


@dataclass
class DataAccessFee:
    """Pricing for accessing a proprietary dataset."""
    dataset_id: str
    base_access_fee: Decimal = Decimal("0")
    per_shard_fee: Decimal = Decimal("0")
    bulk_discount: float = 0.0  # % discount for 10+ shards
    requires_stake: Decimal = Decimal("0")

    def total_for_shards(self, shard_count: int) -> Decimal:
        if shard_count >= BULK_DISCOUNT_THRESHOLD and self.bulk_discount > 0:
            shard_cost = self.per_shard_fee * shard_count * Decimal(str(1.0 - self.bulk_discount))
        else:
            shard_cost = self.per_shard_fee * shard_count
        return self.base_access_fee + shard_cost


@dataclass
class CostQuote:
    """Complete cost breakdown for a job before execution."""
    compute_cost: Decimal = Decimal("0")
    data_cost: Decimal = Decimal("0")
    network_fee: Decimal = Decimal("0")
    shard_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.9
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
```

- [ ] **Step 5: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_pricing_engine.py -v`
Expected: All 16 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prsm/economy/pricing/__init__.py prsm/economy/pricing/models.py tests/unit/test_pricing_engine.py
git commit -m "feat(ring4): PCURate, CostQuote, ProsumerTier, DataAccessFee pricing models"
```

---

### Task 2: Pricing Engine

**Files:**
- Create: `prsm/economy/pricing/engine.py`
- Modify: `prsm/economy/pricing/__init__.py`
- Test: `tests/unit/test_pricing_engine.py` (append)

- [ ] **Step 1: Append tests**

Append to `tests/unit/test_pricing_engine.py`:

```python
from prsm.economy.pricing.engine import PricingEngine
from prsm.compute.agents.models import AgentManifest


class TestPricingEngine:
    def test_estimate_pcu_for_manifest(self):
        engine = PricingEngine()
        manifest = AgentManifest(
            required_cids=["QmA"],
            min_hardware_tier="t2",
            max_memory_bytes=256 * 1024 * 1024,
            max_execution_seconds=30,
        )
        pcu = engine.estimate_pcu(manifest)
        assert pcu > 0

    def test_compute_cost_for_tier(self):
        engine = PricingEngine()
        cost = engine.compute_cost(pcu=100.0, tier="t2")
        # 100 PCU * 0.005 FTNS/PCU = 0.50 FTNS
        assert cost == Decimal("0.50")

    def test_compute_cost_with_spot_pricing(self):
        engine = PricingEngine(network_utilization=0.3)
        cost_spot = engine.compute_cost(pcu=100.0, tier="t2")
        engine_normal = PricingEngine(network_utilization=0.6)
        cost_normal = engine_normal.compute_cost(pcu=100.0, tier="t2")
        # Spot should be cheaper (low utilization discount)
        assert cost_spot < cost_normal

    def test_quote_swarm_job(self):
        engine = PricingEngine()
        quote = engine.quote_swarm_job(
            shard_count=5,
            hardware_tier="t2",
            estimated_pcu_per_shard=20.0,
            data_fee=DataAccessFee(
                dataset_id="ds-1",
                base_access_fee=Decimal("5.0"),
                per_shard_fee=Decimal("0.1"),
            ),
        )
        assert quote.compute_cost > 0
        assert quote.data_cost > 0
        assert quote.network_fee > 0
        assert quote.total > 0
        assert len(quote.shard_breakdown) == 5

    def test_quote_with_no_data_fee(self):
        engine = PricingEngine()
        quote = engine.quote_swarm_job(
            shard_count=3,
            hardware_tier="t1",
            estimated_pcu_per_shard=10.0,
        )
        assert quote.data_cost == Decimal("0")
        assert quote.compute_cost > 0

    def test_quote_with_yield_boost(self):
        engine = PricingEngine()
        # Provider with DEDICATED tier (1.5x boost) earns more
        base_earning = engine.provider_earning(pcu=100.0, tier="t2", prosumer_tier=ProsumerTier.CASUAL)
        boosted_earning = engine.provider_earning(pcu=100.0, tier="t2", prosumer_tier=ProsumerTier.DEDICATED)
        assert boosted_earning > base_earning
        assert boosted_earning == base_earning * Decimal("1.5")

    def test_yield_estimate(self):
        engine = PricingEngine(network_utilization=0.6)
        estimate = engine.yield_estimate(
            hardware_tier="t3",
            tflops=50.0,
            hours_per_day=20,
            prosumer_tier=ProsumerTier.DEDICATED,
        )
        assert estimate["daily_ftns"] > 0
        assert estimate["monthly_ftns"] > 0
        assert estimate["prosumer_tier"] == "dedicated"
```

- [ ] **Step 2: Run tests — verify new tests fail**

Run: `python -m pytest tests/unit/test_pricing_engine.py::TestPricingEngine -v`
Expected: FAIL

- [ ] **Step 3: Implement engine**

Create `prsm/economy/pricing/engine.py`:

```python
"""
Pricing Engine
==============

Calculates PCU-based compute costs, produces cost quotes for swarm jobs,
estimates provider yields, and applies spot pricing adjustments.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from prsm.economy.pricing.models import (
    PCURate,
    CostQuote,
    SpotPriceState,
    ProsumerTier,
    DataAccessFee,
    NETWORK_FEE_RATE,
)

logger = logging.getLogger(__name__)

# Rough PCU estimation: TFLOPS * seconds + memory_gb * seconds
# For a manifest, we estimate from max_execution_seconds and max_memory_bytes
DEFAULT_TFLOPS_BY_TIER = {"t1": 2.0, "t2": 15.0, "t3": 50.0, "t4": 100.0}


class PricingEngine:
    """Calculates costs, quotes, and yield estimates."""

    def __init__(
        self,
        rates: Optional[PCURate] = None,
        network_utilization: float = 0.5,
    ):
        self.rates = rates or PCURate()
        self._spot = SpotPriceState(network_utilization=network_utilization)

    def update_utilization(self, utilization: float) -> None:
        self._spot = SpotPriceState(network_utilization=utilization)

    def estimate_pcu(self, manifest) -> float:
        """Estimate PCU consumption from an AgentManifest."""
        tier = getattr(manifest, "min_hardware_tier", "t1")
        tflops = DEFAULT_TFLOPS_BY_TIER.get(tier, 2.0)
        seconds = getattr(manifest, "max_execution_seconds", 30)
        memory_gb = getattr(manifest, "max_memory_bytes", 256 * 1024 * 1024) / (1024 ** 3)
        return tflops * seconds + memory_gb * seconds

    def compute_cost(self, pcu: float, tier: str) -> Decimal:
        """Calculate FTNS cost for a given PCU amount and tier."""
        base_rate = self.rates.for_tier(tier)
        return Decimal(str(pcu)) * base_rate * self._spot.multiplier

    def provider_earning(
        self,
        pcu: float,
        tier: str,
        prosumer_tier: ProsumerTier = ProsumerTier.CASUAL,
    ) -> Decimal:
        """What a provider earns for executing PCU (before network fee)."""
        base = self.compute_cost(pcu, tier)
        return base * prosumer_tier.yield_boost

    def quote_swarm_job(
        self,
        shard_count: int,
        hardware_tier: str,
        estimated_pcu_per_shard: float,
        data_fee: Optional[DataAccessFee] = None,
    ) -> CostQuote:
        """Produce a complete cost quote for a swarm job."""
        total_pcu = estimated_pcu_per_shard * shard_count
        compute = self.compute_cost(total_pcu, hardware_tier)

        data = Decimal("0")
        if data_fee:
            data = data_fee.total_for_shards(shard_count)

        subtotal = compute + data
        network = subtotal * NETWORK_FEE_RATE

        shard_breakdown = [
            {
                "shard_index": i,
                "pcu_estimate": estimated_pcu_per_shard,
                "cost": str(self.compute_cost(estimated_pcu_per_shard, hardware_tier)),
            }
            for i in range(shard_count)
        ]

        return CostQuote(
            compute_cost=compute,
            data_cost=data,
            network_fee=network,
            shard_breakdown=shard_breakdown,
        )

    def yield_estimate(
        self,
        hardware_tier: str,
        tflops: float,
        hours_per_day: int = 8,
        prosumer_tier: ProsumerTier = ProsumerTier.CASUAL,
    ) -> Dict[str, Any]:
        """Estimate daily/monthly FTNS earnings for a provider."""
        # Estimate PCU per hour: tflops * 3600 seconds + memory_gb * 3600
        # Assume moderate memory usage (8 GB average)
        pcu_per_hour = tflops * 3600 + 8.0 * 3600

        # Assume ~50% job fill rate (not always computing)
        effective_pcu_per_hour = pcu_per_hour * 0.5

        hourly_ftns = self.provider_earning(effective_pcu_per_hour, hardware_tier, prosumer_tier)
        daily_ftns = hourly_ftns * hours_per_day
        monthly_ftns = daily_ftns * 30

        return {
            "hardware_tier": hardware_tier,
            "tflops": tflops,
            "hours_per_day": hours_per_day,
            "prosumer_tier": prosumer_tier.label,
            "yield_boost": str(prosumer_tier.yield_boost),
            "daily_ftns": float(daily_ftns),
            "monthly_ftns": float(monthly_ftns),
            "spot_multiplier": str(self._spot.multiplier),
        }
```

- [ ] **Step 4: Update `__init__.py`**

Add to `prsm/economy/pricing/__init__.py`:

```python
from prsm.economy.pricing.engine import PricingEngine
```
And add `"PricingEngine"` to `__all__`.

- [ ] **Step 5: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_pricing_engine.py -v`
Expected: All 23 tests PASS (16 model + 7 engine)

- [ ] **Step 6: Commit**

```bash
git add prsm/economy/pricing/engine.py prsm/economy/pricing/__init__.py tests/unit/test_pricing_engine.py
git commit -m "feat(ring4): PricingEngine — PCU costing, spot pricing, swarm quoting, yield estimation"
```

---

### Task 3: Prosumer Manager

**Files:**
- Create: `prsm/economy/prosumer.py`
- Test: `tests/unit/test_prosumer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_prosumer.py`:

```python
"""Tests for ProsumerManager — staking tiers, yield estimation, slashing."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from prsm.economy.prosumer import ProsumerManager, ProsumerProfile
from prsm.economy.pricing.models import ProsumerTier


class TestProsumerProfile:
    def test_profile_creation(self):
        profile = ProsumerProfile(
            node_id="node-abc",
            stake_amount=1000.0,
            tier=ProsumerTier.DEDICATED,
            uptime_7d=0.96,
            jobs_completed=150,
            jobs_failed=3,
        )
        assert profile.tier == ProsumerTier.DEDICATED
        assert profile.reliability == pytest.approx(0.98, abs=0.01)

    def test_tier_from_stake(self):
        assert ProsumerProfile(node_id="n", stake_amount=0).tier == ProsumerTier.CASUAL
        assert ProsumerProfile(node_id="n", stake_amount=100).tier == ProsumerTier.PLEDGED
        assert ProsumerProfile(node_id="n", stake_amount=1000).tier == ProsumerTier.DEDICATED
        assert ProsumerProfile(node_id="n", stake_amount=10000).tier == ProsumerTier.SENTINEL

    def test_profile_to_dict(self):
        profile = ProsumerProfile(node_id="n1", stake_amount=500)
        d = profile.to_dict()
        assert d["node_id"] == "n1"
        assert d["tier"] == "pledged"
        assert "reliability" in d


class TestProsumerManager:
    @pytest.fixture
    def manager(self):
        ledger = AsyncMock()
        ledger.get_balance = AsyncMock(return_value=50000.0)
        return ProsumerManager(node_id="test-node", ledger=ledger)

    @pytest.mark.asyncio
    async def test_register_prosumer(self, manager):
        profile = await manager.register(stake_amount=1000.0)
        assert profile.tier == ProsumerTier.DEDICATED
        assert profile.node_id == "test-node"

    @pytest.mark.asyncio
    async def test_register_insufficient_balance(self, manager):
        manager._ledger.get_balance = AsyncMock(return_value=50.0)
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            await manager.register(stake_amount=1000.0)

    @pytest.mark.asyncio
    async def test_get_profile(self, manager):
        await manager.register(stake_amount=500.0)
        profile = manager.get_profile()
        assert profile is not None
        assert profile.tier == ProsumerTier.PLEDGED

    @pytest.mark.asyncio
    async def test_record_job_completion(self, manager):
        await manager.register(stake_amount=100.0)
        manager.record_job_completed()
        profile = manager.get_profile()
        assert profile.jobs_completed == 1

    @pytest.mark.asyncio
    async def test_record_job_failure(self, manager):
        await manager.register(stake_amount=100.0)
        manager.record_job_failed()
        profile = manager.get_profile()
        assert profile.jobs_failed == 1

    @pytest.mark.asyncio
    async def test_slash_for_abandonment(self, manager):
        await manager.register(stake_amount=1000.0)
        slashed = await manager.slash("Job abandoned mid-execution", rate=0.10)
        assert slashed == pytest.approx(100.0)
        profile = manager.get_profile()
        assert profile.stake_amount == pytest.approx(900.0)

    @pytest.mark.asyncio
    async def test_yield_estimate(self, manager):
        await manager.register(stake_amount=1000.0)
        estimate = manager.yield_estimate(
            hardware_tier="t3",
            tflops=50.0,
            hours_per_day=20,
        )
        assert estimate["daily_ftns"] > 0
        assert estimate["prosumer_tier"] == "dedicated"
```

- [ ] **Step 2: Run tests — verify fail**

Run: `python -m pytest tests/unit/test_prosumer.py::TestProsumerProfile -v`
Expected: FAIL

- [ ] **Step 3: Implement ProsumerManager**

Create `prsm/economy/prosumer.py`:

```python
"""
Prosumer Manager
================

Manages prosumer staking tiers, yield estimation, and slashing.
Answers "why would a gamer let PRSM use their PS5?" with concrete numbers.
"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from prsm.economy.pricing.models import ProsumerTier
from prsm.economy.pricing.engine import PricingEngine

logger = logging.getLogger(__name__)


@dataclass
class ProsumerProfile:
    """A node operator's prosumer profile."""
    node_id: str
    stake_amount: float = 0.0
    uptime_7d: float = 1.0  # Rolling 7-day uptime ratio
    jobs_completed: int = 0
    jobs_failed: int = 0
    total_pcu_provided: float = 0.0
    total_ftns_earned: float = 0.0
    registered_at: float = field(default_factory=time.time)

    @property
    def tier(self) -> ProsumerTier:
        return ProsumerTier.from_stake(self.stake_amount)

    @property
    def reliability(self) -> float:
        total = self.jobs_completed + self.jobs_failed
        if total == 0:
            return 1.0
        return self.jobs_completed / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "stake_amount": self.stake_amount,
            "tier": self.tier.label,
            "yield_boost": str(self.tier.yield_boost),
            "uptime_7d": self.uptime_7d,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "reliability": round(self.reliability, 4),
            "total_pcu_provided": self.total_pcu_provided,
            "total_ftns_earned": self.total_ftns_earned,
        }


class ProsumerManager:
    """Manages prosumer staking, profiles, and yield estimation."""

    def __init__(self, node_id: str, ledger=None):
        self._node_id = node_id
        self._ledger = ledger
        self._profile: Optional[ProsumerProfile] = None
        self._pricing = PricingEngine()

    async def register(self, stake_amount: float = 0.0) -> ProsumerProfile:
        """Register as a prosumer with optional stake."""
        if stake_amount > 0 and self._ledger:
            balance = await self._ledger.get_balance(self._node_id)
            if balance < stake_amount:
                raise ValueError(
                    f"Insufficient balance: {balance:.2f} < {stake_amount:.2f} FTNS"
                )

        self._profile = ProsumerProfile(
            node_id=self._node_id,
            stake_amount=stake_amount,
        )

        logger.info(
            f"Prosumer registered: {self._node_id[:8]}, "
            f"stake={stake_amount} FTNS, tier={self._profile.tier.label}"
        )
        return self._profile

    def get_profile(self) -> Optional[ProsumerProfile]:
        return self._profile

    def record_job_completed(self, pcu: float = 0.0, ftns_earned: float = 0.0) -> None:
        if self._profile:
            self._profile.jobs_completed += 1
            self._profile.total_pcu_provided += pcu
            self._profile.total_ftns_earned += ftns_earned

    def record_job_failed(self) -> None:
        if self._profile:
            self._profile.jobs_failed += 1

    async def slash(self, reason: str, rate: float = 0.10) -> float:
        """Slash a percentage of the prosumer's stake.

        Returns the amount slashed.
        """
        if not self._profile or self._profile.stake_amount <= 0:
            return 0.0

        slashed = self._profile.stake_amount * rate
        self._profile.stake_amount -= slashed

        logger.warning(
            f"Prosumer {self._node_id[:8]} slashed {slashed:.2f} FTNS: {reason}"
        )
        return slashed

    def yield_estimate(
        self,
        hardware_tier: str = "t2",
        tflops: float = 15.0,
        hours_per_day: int = 8,
    ) -> Dict[str, Any]:
        """Estimate daily/monthly earnings."""
        tier = self._profile.tier if self._profile else ProsumerTier.CASUAL
        return self._pricing.yield_estimate(
            hardware_tier=hardware_tier,
            tflops=tflops,
            hours_per_day=hours_per_day,
            prosumer_tier=tier,
        )
```

- [ ] **Step 4: Run tests — verify pass**

Run: `python -m pytest tests/unit/test_prosumer.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/economy/prosumer.py tests/unit/test_prosumer.py
git commit -m "feat(ring4): ProsumerManager — staking tiers, yield estimation, slashing"
```

---

### Task 4: Node Integration + Smoke Test

**Files:**
- Modify: `prsm/node/node.py`
- Create: `tests/integration/test_ring4_economy.py`

- [ ] **Step 1: Wire into node.py**

Find the Ring 3 initialization block in `prsm/node/node.py` and add after it:

```python
        # ── Economy Engine (Ring 4) ───────────────────────────────────
        try:
            from prsm.economy.pricing.engine import PricingEngine
            from prsm.economy.prosumer import ProsumerManager

            self.pricing_engine = PricingEngine()
            self.prosumer_manager = ProsumerManager(
                node_id=self.identity.node_id,
                ledger=self.ledger,
            )
            logger.info("Economy engine (Ring 4) initialized")
        except ImportError:
            self.pricing_engine = None
            self.prosumer_manager = None
            logger.debug("Economy engine not available")
```

- [ ] **Step 2: Create integration smoke test**

Create `tests/integration/test_ring4_economy.py`:

```python
"""
Ring 4 Smoke Test
=================

End-to-end: quote a swarm job, check prosumer yield, verify pricing.
"""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

from prsm.economy.pricing import PricingEngine, PCURate, CostQuote, DataAccessFee, ProsumerTier
from prsm.economy.prosumer import ProsumerManager
from prsm.compute.agents.models import AgentManifest


class TestRing4Smoke:
    def test_full_pricing_pipeline(self):
        """Quote a swarm job: compute + data + network fee."""
        engine = PricingEngine(network_utilization=0.6)

        quote = engine.quote_swarm_job(
            shard_count=12,
            hardware_tier="t2",
            estimated_pcu_per_shard=50.0,
            data_fee=DataAccessFee(
                dataset_id="nada-nc-2025",
                base_access_fee=Decimal("5.0"),
                per_shard_fee=Decimal("0.5"),
                bulk_discount=0.2,
            ),
        )

        assert quote.compute_cost > 0
        assert quote.data_cost > 0
        assert quote.network_fee > 0
        assert quote.total == quote.compute_cost + quote.data_cost + quote.network_fee
        assert len(quote.shard_breakdown) == 12

    def test_spot_pricing_affects_cost(self):
        """Low utilization → cheaper. High utilization → more expensive."""
        low = PricingEngine(network_utilization=0.2)
        normal = PricingEngine(network_utilization=0.6)
        high = PricingEngine(network_utilization=0.95)

        cost_low = low.compute_cost(100.0, "t2")
        cost_normal = normal.compute_cost(100.0, "t2")
        cost_high = high.compute_cost(100.0, "t2")

        assert cost_low < cost_normal
        assert cost_high > cost_normal

    @pytest.mark.asyncio
    async def test_prosumer_lifecycle(self):
        """Register → earn → check yield → slash."""
        ledger = AsyncMock()
        ledger.get_balance = AsyncMock(return_value=50000.0)

        manager = ProsumerManager(node_id="ps5-owner", ledger=ledger)

        # Register with stake
        profile = await manager.register(stake_amount=1000.0)
        assert profile.tier == ProsumerTier.DEDICATED

        # Complete some jobs
        manager.record_job_completed(pcu=100.0, ftns_earned=0.5)
        manager.record_job_completed(pcu=200.0, ftns_earned=1.0)
        assert profile.jobs_completed == 2
        assert profile.total_ftns_earned == 1.5

        # Check yield estimate
        estimate = manager.yield_estimate(
            hardware_tier="t3",
            tflops=50.0,
            hours_per_day=20,
        )
        assert estimate["daily_ftns"] > 0
        assert estimate["prosumer_tier"] == "dedicated"

        # Simulate slash for abandoning a job
        slashed = await manager.slash("Abandoned job", rate=0.05)
        assert slashed == pytest.approx(50.0)
        assert profile.stake_amount == pytest.approx(950.0)

    def test_provider_yield_boost_by_tier(self):
        """Higher tier → more earnings for same work."""
        engine = PricingEngine()

        casual = engine.provider_earning(100.0, "t2", ProsumerTier.CASUAL)
        pledged = engine.provider_earning(100.0, "t2", ProsumerTier.PLEDGED)
        dedicated = engine.provider_earning(100.0, "t2", ProsumerTier.DEDICATED)
        sentinel = engine.provider_earning(100.0, "t2", ProsumerTier.SENTINEL)

        assert casual < pledged < dedicated < sentinel
        assert sentinel == casual * Decimal("2.0")
```

- [ ] **Step 3: Run all Ring 4 tests**

Run: `python -m pytest tests/unit/test_pricing_engine.py tests/unit/test_prosumer.py tests/integration/test_ring4_economy.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Run full regression**

Run: `python -m pytest tests/unit/test_wasm_runtime.py tests/unit/test_hardware_profiler.py tests/unit/test_wasm_compute_provider.py tests/unit/test_mobile_agent_models.py tests/unit/test_agent_executor.py tests/unit/test_agent_dispatcher.py tests/unit/test_semantic_shard.py tests/unit/test_swarm_models.py tests/unit/test_swarm_coordinator.py tests/unit/test_pricing_engine.py tests/unit/test_prosumer.py tests/integration/test_ring1_smoke.py tests/integration/test_ring2_dispatch.py tests/integration/test_ring3_swarm.py tests/integration/test_ring4_economy.py -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add prsm/node/node.py tests/integration/test_ring4_economy.py
git commit -m "feat(ring4): wire PricingEngine + ProsumerManager into PRSMNode + integration smoke test"
```

---

### Task 5: Version Bump + Push + PyPI

- [ ] **Step 1:** Bump `__version__` in `prsm/__init__.py` to `"0.29.0"` and `version` in `pyproject.toml` to `"0.29.0"`

- [ ] **Step 2:** Final test run (Ring 4 tests)

- [ ] **Step 3:** Commit, push, build, publish

```bash
git add prsm/__init__.py pyproject.toml
git commit -m "chore: bump version to 0.29.0 for Ring 4 — The Economy"
git push origin main
rm -rf build/ dist/ prsm_network.egg-info/
python3 -m build
python3 -m twine upload dist/prsm_network-0.29.0*
```
