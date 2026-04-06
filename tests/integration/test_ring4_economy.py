"""Ring 4 Smoke Test — pricing + prosumer lifecycle."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

from prsm.economy.pricing import PricingEngine, DataAccessFee, ProsumerTier
from prsm.economy.prosumer import ProsumerManager


class TestRing4Smoke:
    def test_full_pricing_pipeline(self):
        engine = PricingEngine(network_utilization=0.6)
        quote = engine.quote_swarm_job(
            shard_count=12,
            hardware_tier="t2",
            estimated_pcu_per_shard=50.0,
            data_fee=DataAccessFee(
                dataset_id="nada-nc-2025",
                base_access_fee=Decimal("5.0"),
                per_shard_fee=Decimal("0.5"),
                bulk_discount=Decimal("0.2"),
            ),
        )
        assert quote.compute_cost > 0
        assert quote.data_cost > 0
        assert quote.network_fee > 0
        assert quote.total == quote.compute_cost + quote.data_cost + quote.network_fee

    def test_spot_pricing_affects_cost(self):
        low = PricingEngine(network_utilization=0.2)
        normal = PricingEngine(network_utilization=0.6)
        high = PricingEngine(network_utilization=0.95)
        assert low.compute_cost(100.0, "t2") < normal.compute_cost(100.0, "t2")
        assert high.compute_cost(100.0, "t2") > normal.compute_cost(100.0, "t2")

    @pytest.mark.asyncio
    async def test_prosumer_lifecycle(self):
        ledger = AsyncMock()
        ledger.get_balance = AsyncMock(return_value=50000.0)
        manager = ProsumerManager(node_id="ps5-owner", ledger=ledger)
        profile = await manager.register(stake_amount=1000.0)
        assert profile.tier == ProsumerTier.DEDICATED
        manager.record_job_completed(pcu=100.0, ftns_earned=0.5)
        manager.record_job_completed(pcu=200.0, ftns_earned=1.0)
        assert profile.jobs_completed == 2
        estimate = manager.yield_estimate(hardware_tier="t3", tflops=50.0, hours_per_day=20)
        assert float(estimate["daily_ftns"]) > 0
        slashed = await manager.slash("Abandoned job", rate=0.05)
        assert slashed == pytest.approx(50.0)
        assert profile.stake_amount == pytest.approx(950.0)

    def test_provider_yield_boost_by_tier(self):
        engine = PricingEngine()
        casual = engine.provider_earning(100.0, "t2", ProsumerTier.CASUAL)
        sentinel = engine.provider_earning(100.0, "t2", ProsumerTier.SENTINEL)
        assert sentinel == casual * Decimal("2.0")
