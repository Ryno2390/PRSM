"""Unit tests for EligibilityFilter + DispatchPolicy.

Phase 3 Task 4. One test per short-circuit case in design §3.3 plus
a happy path and a composite test.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from prsm.marketplace.filter import EligibilityFilter
from prsm.marketplace.listing import sign_listing
from prsm.marketplace.policy import DispatchPolicy
from prsm.node.identity import generate_node_identity


def _listing(**overrides):
    identity = overrides.pop("identity", None) or generate_node_identity(
        display_name="p"
    )
    kwargs = dict(
        capacity_shards_per_sec=10.0,
        max_shard_bytes=10 * 1024 * 1024,
        supported_dtypes=["float64"],
        price_per_shard_ftns=0.05,
        tee_capable=False,
        stake_tier="standard",
        ttl_seconds=300,
    )
    kwargs.update(overrides)
    return sign_listing(identity=identity, **kwargs)


def test_filter_happy_path_passes_all():
    listings = [_listing() for _ in range(3)]
    policy = DispatchPolicy(
        max_price_per_shard_ftns=1.0,
        required_dtype="float64",
    )
    out = EligibilityFilter().filter(listings, policy)
    assert len(out) == 3


def test_filter_excludes_expired():
    now = int(time.time())
    active = _listing(ttl_seconds=300, advertised_at_unix=now)
    expired = _listing(ttl_seconds=10, advertised_at_unix=now - 100)
    out = EligibilityFilter().filter(
        [active, expired], DispatchPolicy(), at_unix=now,
    )
    assert len(out) == 1
    assert out[0].listing_id == active.listing_id


def test_filter_excludes_above_price_ceiling():
    cheap = _listing(price_per_shard_ftns=0.03)
    expensive = _listing(price_per_shard_ftns=0.10)
    policy = DispatchPolicy(max_price_per_shard_ftns=0.05)
    out = EligibilityFilter().filter([cheap, expensive], policy)
    assert len(out) == 1
    assert out[0].listing_id == cheap.listing_id


def test_filter_excludes_below_price_floor():
    """Anti-loss-leader: suspiciously cheap listings are rejected."""
    normal = _listing(price_per_shard_ftns=0.05)
    too_cheap = _listing(price_per_shard_ftns=0.0001)
    policy = DispatchPolicy(min_price_per_shard_ftns=0.01)
    out = EligibilityFilter().filter([normal, too_cheap], policy)
    assert len(out) == 1
    assert out[0].listing_id == normal.listing_id


def test_filter_tee_required_excludes_non_tee():
    tee = _listing(tee_capable=True)
    non_tee = _listing(tee_capable=False)
    policy = DispatchPolicy(require_tee=True)
    out = EligibilityFilter().filter([tee, non_tee], policy)
    assert len(out) == 1
    assert out[0].listing_id == tee.listing_id


def test_filter_tee_not_required_accepts_both():
    tee = _listing(tee_capable=True)
    non_tee = _listing(tee_capable=False)
    policy = DispatchPolicy(require_tee=False)
    out = EligibilityFilter().filter([tee, non_tee], policy)
    assert len(out) == 2


def test_filter_excludes_below_stake_tier():
    open_tier = _listing(stake_tier="open")
    standard_tier = _listing(stake_tier="standard")
    premium_tier = _listing(stake_tier="premium")
    critical_tier = _listing(stake_tier="critical")

    policy = DispatchPolicy(min_stake_tier="premium")
    out = EligibilityFilter().filter(
        [open_tier, standard_tier, premium_tier, critical_tier], policy,
    )
    out_ids = {l.listing_id for l in out}
    assert out_ids == {premium_tier.listing_id, critical_tier.listing_id}


def test_filter_excludes_missing_required_dtype():
    fp64_only = _listing(supported_dtypes=["float64"])
    fp32_only = _listing(supported_dtypes=["float32"])
    both = _listing(supported_dtypes=["float32", "float64"])

    policy = DispatchPolicy(required_dtype="float64")
    out = EligibilityFilter().filter([fp64_only, fp32_only, both], policy)
    out_ids = {l.listing_id for l in out}
    assert out_ids == {fp64_only.listing_id, both.listing_id}


def test_filter_excludes_below_min_capacity():
    """Zero-capacity listings (from an at-max advertiser per Task 3) are
    the canonical case — a policy with min_capacity_shards_per_sec>0
    filters them out."""
    busy = _listing(capacity_shards_per_sec=0.0)
    available = _listing(capacity_shards_per_sec=5.0)
    policy = DispatchPolicy(min_capacity_shards_per_sec=0.1)
    out = EligibilityFilter().filter([busy, available], policy)
    assert len(out) == 1
    assert out[0].listing_id == available.listing_id


def test_filter_excludes_below_min_reputation():
    """When a ReputationTracker is wired, min_reputation_score gates."""
    trusted = _listing()
    shady = _listing()
    reputation = MagicMock()
    reputation.score_for = MagicMock(
        side_effect=lambda pid: 0.9 if pid == trusted.provider_id else 0.2
    )
    policy = DispatchPolicy(min_reputation_score=0.5)
    out = EligibilityFilter(reputation_tracker=reputation).filter(
        [trusted, shady], policy,
    )
    assert len(out) == 1
    assert out[0].listing_id == trusted.listing_id


def test_filter_skips_reputation_check_when_tracker_not_wired():
    """Without a tracker, min_reputation_score is not enforced — the
    default 0.0 is a no-op even if the caller raises it."""
    listings = [_listing() for _ in range(2)]
    policy = DispatchPolicy(min_reputation_score=0.99)
    out = EligibilityFilter(reputation_tracker=None).filter(listings, policy)
    # All listings pass — tracker absent.
    assert len(out) == 2


def test_filter_composite_multi_criteria():
    """Realistic policy with several filters — each exclusion reason
    rejects a distinct listing so we see the union of filters work."""
    identity_list = [
        generate_node_identity(display_name=f"p{i}") for i in range(5)
    ]
    happy = _listing(
        identity=identity_list[0],
        price_per_shard_ftns=0.03,
        tee_capable=True,
        stake_tier="premium",
        supported_dtypes=["float64"],
        capacity_shards_per_sec=5.0,
    )
    above_price = _listing(
        identity=identity_list[1],
        price_per_shard_ftns=0.99,
        tee_capable=True,
        stake_tier="premium",
        supported_dtypes=["float64"],
    )
    no_tee = _listing(
        identity=identity_list[2],
        price_per_shard_ftns=0.03,
        tee_capable=False,
        stake_tier="premium",
        supported_dtypes=["float64"],
    )
    low_tier = _listing(
        identity=identity_list[3],
        price_per_shard_ftns=0.03,
        tee_capable=True,
        stake_tier="open",
        supported_dtypes=["float64"],
    )
    wrong_dtype = _listing(
        identity=identity_list[4],
        price_per_shard_ftns=0.03,
        tee_capable=True,
        stake_tier="premium",
        supported_dtypes=["float32"],
    )

    policy = DispatchPolicy(
        max_price_per_shard_ftns=0.10,
        require_tee=True,
        min_stake_tier="premium",
        required_dtype="float64",
    )
    out = EligibilityFilter().filter(
        [happy, above_price, no_tee, low_tier, wrong_dtype], policy,
    )
    assert len(out) == 1
    assert out[0].listing_id == happy.listing_id


def test_filter_preserves_input_order():
    """Output ordering matches input ordering (up to excluded listings).
    Downstream callers — e.g., TopologyRandomizer — rely on this."""
    a = _listing(price_per_shard_ftns=0.02)
    b = _listing(price_per_shard_ftns=0.03)
    c = _listing(price_per_shard_ftns=0.04)
    out = EligibilityFilter().filter(
        [a, b, c], DispatchPolicy(max_price_per_shard_ftns=1.0),
    )
    assert [l.listing_id for l in out] == [a.listing_id, b.listing_id, c.listing_id]
