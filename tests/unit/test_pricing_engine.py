"""
Tests for the Hybrid Pricing Engine (Ring 4).
"""

from decimal import Decimal

import pytest

from prsm.economy.pricing.models import (
    BULK_DISCOUNT_THRESHOLD,
    NETWORK_FEE_RATE,
    CostQuote,
    DataAccessFee,
    PCURate,
    ProsumerTier,
    SpotPriceState,
)
from prsm.economy.pricing.engine import PricingEngine


# ── PCURate ─────────────────────────────────────────────────────────────────


class TestPCURate:
    def test_default_rates(self):
        r = PCURate()
        assert r.t1_rate == Decimal("0.001")
        assert r.t2_rate == Decimal("0.005")
        assert r.t3_rate == Decimal("0.02")
        assert r.t4_rate == Decimal("0.10")

    def test_for_tier_known(self):
        r = PCURate()
        assert r.for_tier("t3") == Decimal("0.02")

    def test_for_tier_unknown_defaults_to_t1(self):
        r = PCURate()
        assert r.for_tier("unknown") == Decimal("0.001")


# ── CostQuote ───────────────────────────────────────────────────────────────


class TestCostQuote:
    def test_total_sums_three_costs(self):
        q = CostQuote(
            compute_cost=Decimal("1.0"),
            data_cost=Decimal("0.5"),
            network_fee=Decimal("0.075"),
        )
        assert q.total == Decimal("1.575")

    def test_to_dict_includes_total(self):
        q = CostQuote(
            compute_cost=Decimal("2"),
            data_cost=Decimal("1"),
            network_fee=Decimal("0.15"),
        )
        d = q.to_dict()
        assert d["total"] == str(Decimal("3.15"))

    def test_default_confidence(self):
        q = CostQuote()
        assert q.confidence == 0.95

    def test_shard_breakdown_default_empty(self):
        q = CostQuote()
        assert q.shard_breakdown == []


# ── SpotPriceState ──────────────────────────────────────────────────────────


class TestSpotPriceState:
    def test_low_utilization_discount(self):
        """At 0% utilisation the multiplier should be 0.5."""
        s = SpotPriceState(network_utilization=0.0)
        assert s.multiplier == Decimal("0.5")

    def test_mid_utilization_flat(self):
        """Between 0.4 and 0.8 the multiplier should be exactly 1.0."""
        s = SpotPriceState(network_utilization=0.6)
        assert s.multiplier == Decimal("1.0")

    def test_high_utilization_premium(self):
        """At 100% utilisation the multiplier should be 1.25."""
        s = SpotPriceState(network_utilization=1.0)
        assert s.multiplier == Decimal("1.25")

    def test_boundary_04(self):
        """At exactly 0.4 the multiplier should be 1.0."""
        s = SpotPriceState(network_utilization=0.4)
        assert s.multiplier == Decimal("1.0")


# ── ProsumerTier ────────────────────────────────────────────────────────────


class TestProsumerTier:
    def test_from_stake_casual(self):
        assert ProsumerTier.from_stake(0) == ProsumerTier.CASUAL

    def test_from_stake_sentinel(self):
        assert ProsumerTier.from_stake(10000) == ProsumerTier.SENTINEL

    def test_yield_boost_dedicated(self):
        assert ProsumerTier.DEDICATED.yield_boost == Decimal("1.5")


# ── DataAccessFee ───────────────────────────────────────────────────────────


class TestDataAccessFee:
    def test_total_below_bulk_threshold(self):
        fee = DataAccessFee(
            dataset_id="ds1",
            base_access_fee=Decimal("0.01"),
            per_shard_fee=Decimal("0.002"),
            bulk_discount=Decimal("0.1"),
        )
        # 5 shards, no discount
        expected = Decimal("0.01") + Decimal("0.002") * Decimal("5")
        assert fee.total_for_shards(5) == expected

    def test_total_at_bulk_threshold(self):
        fee = DataAccessFee(
            dataset_id="ds1",
            base_access_fee=Decimal("0.01"),
            per_shard_fee=Decimal("0.002"),
            bulk_discount=Decimal("0.1"),
        )
        # 10 shards, discount applied
        expected = Decimal("0.01") + Decimal("0.002") * Decimal("10") * Decimal("0.9")
        assert fee.total_for_shards(10) == expected

    def test_total_above_bulk_threshold(self):
        fee = DataAccessFee(
            dataset_id="ds2",
            base_access_fee=Decimal("0.05"),
            per_shard_fee=Decimal("0.01"),
            bulk_discount=Decimal("0.2"),
        )
        # 20 shards, discount applied
        expected = Decimal("0.05") + Decimal("0.01") * Decimal("20") * Decimal("0.8")
        assert fee.total_for_shards(20) == expected


# ── PricingEngine ───────────────────────────────────────────────────────────


class TestPricingEngine:
    def test_compute_cost_t1_default_utilization(self):
        engine = PricingEngine(network_utilization=0.5)
        cost = engine.compute_cost(10.0, "t1")
        # 10 * 0.001 * 1.0 = 0.01
        assert cost == Decimal("10.0") * Decimal("0.001") * Decimal("1.0")

    def test_compute_cost_with_high_utilization(self):
        engine = PricingEngine(network_utilization=1.0)
        cost = engine.compute_cost(10.0, "t1")
        # 10 * 0.001 * 1.25 = 0.0125
        assert cost == Decimal("10.0") * Decimal("0.001") * Decimal("1.25")

    def test_update_utilization(self):
        engine = PricingEngine(network_utilization=0.5)
        engine.update_utilization(0.0)
        assert engine.spot.network_utilization == 0.0
        assert engine.spot.multiplier == Decimal("0.5")

    def test_provider_earning_with_yield_boost(self):
        engine = PricingEngine(network_utilization=0.5)
        base = engine.compute_cost(100.0, "t2")
        boosted = engine.provider_earning(100.0, "t2", ProsumerTier.DEDICATED)
        assert boosted == base * Decimal("1.5")

    def test_quote_swarm_job_no_data_fee(self):
        engine = PricingEngine(network_utilization=0.5)
        quote = engine.quote_swarm_job(shard_count=5, hardware_tier="t1", estimated_pcu_per_shard=1.0)
        assert len(quote.shard_breakdown) == 5
        assert quote.data_cost == Decimal("0")
        assert quote.network_fee == quote.compute_cost * NETWORK_FEE_RATE

    def test_quote_swarm_job_with_data_fee(self):
        engine = PricingEngine(network_utilization=0.5)
        data_fee = DataAccessFee(
            dataset_id="ds1",
            base_access_fee=Decimal("0.01"),
            per_shard_fee=Decimal("0.002"),
        )
        quote = engine.quote_swarm_job(
            shard_count=12, hardware_tier="t2", estimated_pcu_per_shard=2.0, data_fee=data_fee
        )
        assert quote.data_cost > Decimal("0")
        assert quote.total == quote.compute_cost + quote.data_cost + quote.network_fee

    def test_yield_estimate_returns_expected_keys(self):
        engine = PricingEngine(network_utilization=0.5)
        result = engine.yield_estimate(
            hardware_tier="t2", tflops=15.0, hours_per_day=8.0, prosumer_tier=ProsumerTier.PLEDGED
        )
        assert "daily_ftns" in result
        assert "monthly_ftns" in result
        assert result["prosumer_tier"] == "PLEDGED"
        # monthly should be 30x daily
        assert Decimal(result["monthly_ftns"]) == Decimal(result["daily_ftns"]) * Decimal("30")
