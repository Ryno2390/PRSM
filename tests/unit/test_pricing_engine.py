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
