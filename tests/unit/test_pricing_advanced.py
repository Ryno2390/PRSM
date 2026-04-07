"""Tests for advanced pricing: revenue split, data listings, spot arbitrage."""

import pytest
from decimal import Decimal

from prsm.economy.pricing.revenue_split import RevenueSplitEngine, RevenueSplit
from prsm.economy.pricing.data_listing import DataListingManager, DataListing
from prsm.economy.pricing.spot_arbitrage import SpotArbitrage, MarketMetrics
from prsm.economy.pricing import PricingEngine


class TestRevenueSplit:
    def test_split_with_data_owner(self):
        engine = RevenueSplitEngine()
        split = engine.calculate_split(
            total_payment=Decimal("100.0"),
            data_owner_id="nada-org",
            compute_providers={"ps5-node": 50.0, "pc-node": 50.0},
        )
        assert split.data_owner_amount == Decimal("80.0")
        assert split.treasury_amount == Decimal("5.0")
        total_compute = sum(split.compute_amounts.values())
        assert total_compute == Decimal("15.0")
        assert split.compute_amounts["ps5-node"] == Decimal("7.5")

    def test_split_without_data_owner(self):
        engine = RevenueSplitEngine()
        split = engine.calculate_split(
            total_payment=Decimal("100.0"),
            compute_providers={"node-a": 100.0},
        )
        assert split.data_owner_amount == Decimal("0")
        assert split.treasury_amount == Decimal("5.0")
        assert split.compute_amounts["node-a"] == Decimal("95.0")

    def test_split_proportional_by_pcu(self):
        engine = RevenueSplitEngine()
        split = engine.calculate_split(
            total_payment=Decimal("100.0"),
            data_owner_id="owner",
            compute_providers={"fast": 75.0, "slow": 25.0},
        )
        assert split.compute_amounts["fast"] > split.compute_amounts["slow"]
        # fast did 75% of work -> gets 75% of compute pool
        assert split.compute_amounts["fast"] == Decimal("15.0") * Decimal("0.75")

    def test_split_to_dict(self):
        engine = RevenueSplitEngine()
        split = engine.calculate_split(Decimal("50.0"), "owner", {"p1": 10.0})
        d = split.to_dict()
        assert "data_owner_amount" in d
        assert "treasury_amount" in d

    def test_80_15_5_adds_up(self):
        engine = RevenueSplitEngine()
        split = engine.calculate_split(
            Decimal("200.0"), "owner", {"p1": 50.0, "p2": 50.0}
        )
        total = split.data_owner_amount + sum(split.compute_amounts.values()) + split.treasury_amount
        assert total == Decimal("200.0")


class TestDataListing:
    def test_publish_listing(self):
        mgr = DataListingManager()
        listing = DataListing(
            dataset_id="nada-nc-2025",
            owner_id="nada-org",
            title="NADA NC Vehicle Registrations 2025",
            base_access_fee=Decimal("5.0"),
            per_shard_fee=Decimal("0.5"),
            shard_count=12,
        )
        lid = mgr.publish(listing)
        assert lid == listing.listing_id
        assert mgr.get_listing(lid) is not None

    def test_search_by_keyword(self):
        mgr = DataListingManager()
        mgr.publish(DataListing(dataset_id="d1", title="NADA NC 2025", base_access_fee=Decimal("5")))
        mgr.publish(DataListing(dataset_id="d2", title="Weather Data CA", base_access_fee=Decimal("2")))
        results = mgr.search("NADA")
        assert len(results) == 1
        assert results[0].dataset_id == "d1"

    def test_search_by_max_price(self):
        mgr = DataListingManager()
        mgr.publish(DataListing(dataset_id="cheap", title="Cheap", base_access_fee=Decimal("1")))
        mgr.publish(DataListing(dataset_id="expensive", title="Expensive", base_access_fee=Decimal("100")))
        results = mgr.search(max_price=Decimal("10"))
        assert len(results) == 1
        assert results[0].dataset_id == "cheap"

    def test_check_access_insufficient_stake(self):
        mgr = DataListingManager()
        listing = DataListing(
            dataset_id="premium",
            title="Premium Data",
            requires_stake=Decimal("1000"),
        )
        mgr.publish(listing)
        allowed, reason = mgr.check_access(listing.listing_id, accessor_stake=Decimal("500"))
        assert not allowed
        assert "Insufficient stake" in reason

    def test_check_access_sufficient_stake(self):
        mgr = DataListingManager()
        listing = DataListing(
            dataset_id="premium",
            title="Premium Data",
            requires_stake=Decimal("1000"),
        )
        mgr.publish(listing)
        allowed, reason = mgr.check_access(listing.listing_id, accessor_stake=Decimal("1500"))
        assert allowed

    def test_record_query(self):
        mgr = DataListingManager()
        listing = DataListing(dataset_id="d1", title="Test")
        mgr.publish(listing)
        mgr.record_query(listing.listing_id, Decimal("5.0"))
        mgr.record_query(listing.listing_id, Decimal("5.0"))
        assert listing.total_queries == 2
        assert listing.total_revenue == Decimal("10.0")

    def test_deactivate(self):
        mgr = DataListingManager()
        listing = DataListing(dataset_id="d1", title="Test")
        mgr.publish(listing)
        assert mgr.deactivate(listing.listing_id)
        assert not listing.active
        assert len(mgr.list_all(active_only=True)) == 0

    def test_to_dict_roundtrip(self):
        listing = DataListing(
            dataset_id="d1", title="Test", base_access_fee=Decimal("5.0"),
            requires_stake=Decimal("100"),
        )
        d = listing.to_dict()
        restored = DataListing.from_dict(d)
        assert restored.dataset_id == "d1"
        assert restored.requires_stake == Decimal("100")


class TestSpotArbitrage:
    def test_metrics_utilization(self):
        m = MarketMetrics(pending_jobs=5, active_providers=10)
        assert 0 < m.utilization < 1

    def test_should_lower_prices(self):
        arb = SpotArbitrage(low_acceptance_threshold=0.5)
        arb.update_metrics(MarketMetrics(acceptance_rate=0.3, active_providers=10))
        assert arb.should_lower_prices()

    def test_should_raise_prices(self):
        arb = SpotArbitrage()
        arb.update_metrics(MarketMetrics(
            pending_jobs=100, active_providers=10, acceptance_rate=0.95,
        ))
        assert arb.should_raise_prices()

    def test_hold_in_balance(self):
        arb = SpotArbitrage()
        arb.update_metrics(MarketMetrics(
            pending_jobs=5, active_providers=10, acceptance_rate=0.8,
        ))
        rec = arb.get_recommended_adjustment()
        assert rec["action"] == "hold"

    def test_updates_pricing_engine(self):
        engine = PricingEngine()
        arb = SpotArbitrage(pricing_engine=engine)
        arb.update_metrics(MarketMetrics(pending_jobs=50, active_providers=5))
        # Engine utilization should have been updated
        assert engine.spot.network_utilization > 0


class TestCLIListDataset:
    def test_command_exists(self):
        from click.testing import CliRunner
        from prsm.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["marketplace", "list-dataset", "--help"])
        assert result.exit_code == 0
        assert "--title" in result.output
        assert "--base-fee" in result.output
        assert "--require-stake" in result.output
