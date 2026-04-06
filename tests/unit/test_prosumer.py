"""
Tests for ProsumerManager (Ring 4).
"""

import time

import pytest
from unittest.mock import AsyncMock

from prsm.economy.pricing.models import ProsumerTier
from prsm.economy.prosumer import ProsumerManager, ProsumerProfile


# ── ProsumerProfile ────────────────────────────────────────────────────────


class TestProsumerProfile:
    def test_profile_creation_defaults(self):
        before = time.time()
        p = ProsumerProfile(node_id="node-1")
        after = time.time()
        assert p.node_id == "node-1"
        assert p.stake_amount == 0.0
        assert p.uptime_7d == 1.0
        assert p.jobs_completed == 0
        assert p.jobs_failed == 0
        assert p.total_pcu_provided == 0.0
        assert p.total_ftns_earned == 0.0
        assert before <= p.registered_at <= after

    def test_tier_from_stake(self):
        p = ProsumerProfile(node_id="n1", stake_amount=500)
        assert p.tier == ProsumerTier.PLEDGED

        p2 = ProsumerProfile(node_id="n2", stake_amount=10000)
        assert p2.tier == ProsumerTier.SENTINEL

    def test_reliability_no_jobs(self):
        p = ProsumerProfile(node_id="n1")
        assert p.reliability == 1.0

    def test_reliability_with_jobs(self):
        p = ProsumerProfile(node_id="n1", jobs_completed=8, jobs_failed=2)
        assert p.reliability == pytest.approx(0.8)

    def test_to_dict(self):
        p = ProsumerProfile(node_id="n1", stake_amount=1000, jobs_completed=5)
        d = p.to_dict()
        assert d["node_id"] == "n1"
        assert d["stake_amount"] == 1000
        assert d["tier"] == "DEDICATED"
        assert d["reliability"] == 1.0
        assert "registered_at" in d


# ── ProsumerManager ───────────────────────────────────────────────────────


class TestProsumerManager:
    @pytest.mark.asyncio
    async def test_register(self):
        ledger = AsyncMock()
        ledger.get_balance = AsyncMock(return_value=5000.0)
        mgr = ProsumerManager(node_id="n1", ledger=ledger)
        profile = await mgr.register(stake_amount=1000.0)
        assert profile.stake_amount == 1000.0
        assert profile.tier == ProsumerTier.DEDICATED

    @pytest.mark.asyncio
    async def test_register_insufficient_balance(self):
        ledger = AsyncMock()
        ledger.get_balance = AsyncMock(return_value=50.0)
        mgr = ProsumerManager(node_id="n1", ledger=ledger)
        with pytest.raises(ValueError, match="Insufficient"):
            await mgr.register(stake_amount=1000.0)

    def test_get_profile_before_register(self):
        mgr = ProsumerManager(node_id="n1")
        assert mgr.get_profile() is None

    @pytest.mark.asyncio
    async def test_get_profile_after_register(self):
        mgr = ProsumerManager(node_id="n1")
        await mgr.register(stake_amount=0.0)
        assert mgr.get_profile() is not None

    @pytest.mark.asyncio
    async def test_record_job_completed(self):
        mgr = ProsumerManager(node_id="n1")
        await mgr.register()
        mgr.record_job_completed(pcu=100.0, ftns_earned=0.5)
        mgr.record_job_completed(pcu=200.0, ftns_earned=1.0)
        p = mgr.get_profile()
        assert p.jobs_completed == 2
        assert p.total_pcu_provided == pytest.approx(300.0)
        assert p.total_ftns_earned == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_record_job_failed(self):
        mgr = ProsumerManager(node_id="n1")
        await mgr.register()
        mgr.record_job_failed()
        mgr.record_job_failed()
        p = mgr.get_profile()
        assert p.jobs_failed == 2

    @pytest.mark.asyncio
    async def test_slash(self):
        ledger = AsyncMock()
        ledger.get_balance = AsyncMock(return_value=5000.0)
        mgr = ProsumerManager(node_id="n1", ledger=ledger)
        await mgr.register(stake_amount=1000.0)
        slashed = await mgr.slash("Abandoned job", rate=0.10)
        assert slashed == pytest.approx(100.0)
        assert mgr.get_profile().stake_amount == pytest.approx(900.0)

    @pytest.mark.asyncio
    async def test_yield_estimate(self):
        mgr = ProsumerManager(node_id="n1")
        await mgr.register(stake_amount=1000.0)
        est = mgr.yield_estimate(hardware_tier="t3", tflops=50.0, hours_per_day=20)
        assert "daily_ftns" in est
        assert float(est["daily_ftns"]) > 0
