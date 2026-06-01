"""Sprint 906 — staking utility benefits: lock-based discounts + priority.

PRSM_Tokenomics.md §5.3: staking confers UTILITY benefits — lock-based
service discounts and priority access — and NO token yield (sp904). Until
now those benefits were doc-only. sp906 implements them:

- A lock-tier benefit schedule (30d → 2%/+10%, 90d → 5%/+25%,
  365d → 10%/+50%) computed by `StakingConfig.benefits_for_lock_days`.
- `StakingBenefits` carries the discount + priority and applies them.
- The discount reduces the **network-fee (treasury) share only** — never
  the operator/creator share — so it can't shortchange providers
  (`RevenueSplitEngine.calculate_split(..., network_fee_discount_fraction)`).
- Priority biases dispatch toward higher-capacity providers
  (`MarketplaceOrchestrator._select_top_k(..., priority_boost)`).

These are the pure-function tests; the DB-backed stake/unstake/lookup
tests live in test_staking_incentives.py (reusing its async-DB harness).
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from prsm.economy.tokenomics.staking_manager import (
    StakingBenefits,
    StakingConfig,
    ALLOWED_LOCK_DAYS,
)


# ── StakingBenefits ──────────────────────────────────────


def test_none_benefits_are_inert():
    b = StakingBenefits.none()
    assert b.discount_fraction == 0.0
    assert b.priority_boost == 0.0
    assert b.is_active is False
    assert b.discounted_fee(Decimal("100")) == Decimal("100")
    assert b.boosted_priority(1.0) == 1.0


def test_discounted_fee_reduces_by_fraction():
    b = StakingBenefits(365, "365d", 0.10, 0.50)
    # 10% off a 5 FTNS network fee -> 4.5
    assert b.discounted_fee(Decimal("5")) == Decimal("4.5")


def test_boosted_priority_multiplies():
    b = StakingBenefits(90, "90d", 0.05, 0.25)
    assert b.boosted_priority(2.0) == pytest.approx(2.5)  # 2.0 * 1.25


def test_is_active_true_when_any_benefit():
    assert StakingBenefits(30, "30d", 0.02, 0.10).is_active is True


# ── StakingConfig.benefits_for_lock_days ─────────────────


@pytest.mark.parametrize(
    "days,discount,priority,label",
    [
        (0, 0.0, 0.0, "none"),
        (None, 0.0, 0.0, "none"),
        (29, 0.0, 0.0, "none"),       # below the lowest tier
        (30, 0.02, 0.10, "30d"),
        (89, 0.02, 0.10, "30d"),      # rounds down to the 30d tier
        (90, 0.05, 0.25, "90d"),
        (364, 0.05, 0.25, "90d"),
        (365, 0.10, 0.50, "365d"),
        (1000, 0.10, 0.50, "365d"),   # above the top tier caps at 365d
    ],
)
def test_benefits_for_lock_days(days, discount, priority, label):
    b = StakingConfig().benefits_for_lock_days(days)
    assert b.discount_fraction == discount
    assert b.priority_boost == priority
    assert b.tier_label == label


def test_allowed_lock_days_are_the_tier_thresholds():
    assert ALLOWED_LOCK_DAYS == (30, 90, 365)


# ── RevenueSplitEngine network-fee discount ──────────────


def _engine():
    from prsm.economy.pricing.revenue_split import RevenueSplitEngine
    return RevenueSplitEngine()


def test_fee_discount_reduces_only_treasury():
    """A staker's discount waives part of the 5% treasury fee; the data
    owner (80%) and compute (15%) shares are byte-identical to no-discount."""
    eng = _engine()
    base = eng.calculate_split(
        Decimal("100"), data_owner_id="creator", compute_providers={"op": 1.0},
    )
    disc = eng.calculate_split(
        Decimal("100"), data_owner_id="creator", compute_providers={"op": 1.0},
        network_fee_discount_fraction=0.10,
    )
    # Operator + creator untouched.
    assert disc.data_owner_amount == base.data_owner_amount
    assert disc.compute_amounts == base.compute_amounts
    # Treasury reduced by 10%: 5 -> 4.5
    assert base.treasury_amount == Decimal("5.00")
    assert disc.treasury_amount == Decimal("4.50")
    # The waived 0.5 is surfaced as a payer rebate.
    assert disc.fee_discount_amount == Decimal("0.50")


def test_fee_discount_lowers_payer_effective_total():
    eng = _engine()
    disc = eng.calculate_split(
        Decimal("100"), data_owner_id="creator", compute_providers={"op": 1.0},
        network_fee_discount_fraction=0.10,
    )
    # Payer funds data + compute + discounted treasury = 80 + 15 + 4.5.
    assert disc.effective_total_paid == Decimal("99.50")
    # Conservation: every FTNS is accounted for.
    paid_out = (
        disc.data_owner_amount
        + sum(disc.compute_amounts.values())
        + disc.treasury_amount
    )
    assert paid_out == disc.effective_total_paid
    assert (
        disc.effective_total_paid + disc.fee_discount_amount
        == disc.total_payment
    )


def test_zero_discount_matches_legacy_split():
    eng = _engine()
    s = eng.calculate_split(Decimal("100"), compute_providers={"op": 1.0})
    assert s.treasury_amount == Decimal("5.00")
    assert s.fee_discount_amount == Decimal("0")
    assert s.effective_total_paid == Decimal("100")


# ── Marketplace priority ─────────────────────────────────


def test_dispatch_policy_has_priority_boost_default_zero():
    from prsm.marketplace.policy import DispatchPolicy
    assert DispatchPolicy().requester_priority_boost == 0.0


def _listing(provider_id, tier, price, capacity):
    from prsm.marketplace.listing import ProviderListing
    return ProviderListing(
        listing_id=f"L-{provider_id}",
        provider_id=provider_id,
        provider_pubkey_b64="",
        capacity_shards_per_sec=capacity,
        max_shard_bytes=10**9,
        supported_dtypes=["float64"],
        price_per_shard_ftns=price,
        tee_capable=False,
        stake_tier=tier,
        advertised_at_unix=1,
        ttl_seconds=10**12,
        signature="",
    )


def test_priority_boost_prefers_higher_capacity_provider():
    """With a priority boost, a faster (higher-capacity) provider is
    selected over a marginally-better-priced slow one."""
    from prsm.marketplace.orchestrator import MarketplaceOrchestrator
    # Same tier + price; A is much faster than B.
    fast = _listing("A-fast", "standard", 1.0, capacity=100.0)
    slow = _listing("B-slow", "standard", 1.0, capacity=1.0)
    listings = [slow, fast]

    # No boost: tie on tier/price -> deterministic lexicographic (A-fast
    # wins the tiebreak here anyway, so use a price edge to make B win
    # without boost).
    slow_cheaper = _listing("B-slow", "standard", 0.9, capacity=1.0)
    no_boost = MarketplaceOrchestrator._select_top_k(
        [slow_cheaper, fast], 1, priority_boost=0.0,
    )
    assert no_boost[0].provider_id == "B-slow"  # cheaper wins without boost

    # With a strong boost, the faster provider is preferred despite price.
    boosted = MarketplaceOrchestrator._select_top_k(
        [slow_cheaper, fast], 1, priority_boost=0.50,
    )
    assert boosted[0].provider_id == "A-fast"
