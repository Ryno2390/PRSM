"""Phase 6 chaos scenario tests.

Exercises LivenessMonitor + RateLimiter end-to-end through the harness
at tests/chaos/harness.py. Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §6 Task 7.

Pass criteria (from plan §7 + §4.3):
  * Dead peers are evicted within ~3 ping intervals.
  * Adversarial peers are banned without degrading legitimate traffic
    targeted at the same victim.
  * Legitimate request allow-rate stays high under adversarial load.

The scale here is CI-friendly (20 nodes, 10 ticks). Plan-scale
scenarios (100 nodes, ~120 ticks for 1-hour sim) are obtainable by
scaling ChaosScenario parameters without changing the harness or
primitives.
"""

from __future__ import annotations

import pytest

from tests.chaos.harness import ChaosScenario, SimNetwork


def _run(**overrides):
    scenario = ChaosScenario(**overrides)
    net = SimNetwork(scenario)
    net.setup()
    return net.run()


# -----------------------------------------------------------------------------
# Liveness under dead-peer chaos
# -----------------------------------------------------------------------------


def test_dead_peers_are_evicted_within_three_intervals():
    report = _run(
        node_count=20,
        dead_peer_count=3,
        adversarial_count=0,
        ticks=5,  # 5 intervals; eviction should occur by tick 3
        churn_fraction_per_tick=0.0,
    )
    assert report.dead_peer_eviction_rate == 1.0, (
        f"expected all {report.scenario.dead_peer_count} dead peers to be "
        f"evicted, got {report.dead_peers_evicted}"
    )
    # Each dead peer evicted by tick 3 at latest (ticks are 0-indexed, and
    # the eviction window requires the ping to age past interval first, so
    # a 3-miss threshold resolves by tick 2 or 3 in this harness).
    assert all(t <= 3 for t in report.dead_peer_ticks_to_eviction)


def test_dead_peer_eviction_robust_under_churn():
    """Churn on legitimate peers should not delay dead-peer eviction."""
    report = _run(
        node_count=30,
        dead_peer_count=3,
        adversarial_count=0,
        ticks=5,
        churn_fraction_per_tick=0.20,
    )
    assert report.dead_peer_eviction_rate == 1.0


# -----------------------------------------------------------------------------
# Rate limiter under adversarial-peer chaos
# -----------------------------------------------------------------------------


def test_adversarial_peers_are_banned_by_victims():
    report = _run(
        node_count=20,
        adversarial_count=2,
        dead_peer_count=0,
        adversarial_requests_per_tick=30,  # well over 10-req limit
        legitimate_requests_per_tick=2,
        ticks=10,
        churn_fraction_per_tick=0.0,
    )
    # Both adversaries should have been banned by at least one victim
    # during the 10-tick run.
    assert report.adversarial_peers_banned == report.scenario.adversarial_count


def test_adversarial_spam_does_not_starve_legitimate_traffic():
    """Victim rate-limiters are per-source, so a banned adversary must
    not prevent legitimate peers from successfully submitting requests."""
    report = _run(
        node_count=30,
        adversarial_count=3,
        dead_peer_count=0,
        adversarial_requests_per_tick=40,
        legitimate_requests_per_tick=2,
        ticks=8,
        churn_fraction_per_tick=0.0,
    )
    # Legitimate traffic under cap should see ~100% allow rate.
    assert report.legit_allow_rate >= 0.95, (
        f"legitimate allow rate degraded to {report.legit_allow_rate:.2%}"
    )


def test_adversarial_requests_mostly_rejected():
    report = _run(
        node_count=20,
        adversarial_count=2,
        dead_peer_count=0,
        adversarial_requests_per_tick=50,
        legitimate_requests_per_tick=1,
        ticks=10,
        churn_fraction_per_tick=0.0,
    )
    # With 50 req/tick at a 10/60s cap against each victim, most
    # adversarial requests land post-throttle/ban.
    assert report.adversarial_reject_rate >= 0.80, (
        f"adversarial reject rate only {report.adversarial_reject_rate:.2%}"
    )


# -----------------------------------------------------------------------------
# Combined chaos — dead + adversarial + churn
# -----------------------------------------------------------------------------


def test_combined_chaos_meets_all_criteria():
    """Plan §4.3's full chaos mix at CI-friendly scale."""
    report = _run(
        node_count=30,
        adversarial_count=3,
        dead_peer_count=4,
        churn_fraction_per_tick=0.15,
        ticks=8,
        adversarial_requests_per_tick=25,
        legitimate_requests_per_tick=2,
    )
    assert report.dead_peer_eviction_rate == 1.0
    assert report.adversarial_peers_banned == report.scenario.adversarial_count
    assert report.legit_allow_rate >= 0.95


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_same_seed_produces_identical_report():
    r1 = _run(seed=42, node_count=20, ticks=8)
    r2 = _run(seed=42, node_count=20, ticks=8)
    assert r1.legit_requests_total == r2.legit_requests_total
    assert r1.adversarial_requests_total == r2.adversarial_requests_total
    assert r1.dead_peers_evicted == r2.dead_peers_evicted
    assert r1.adversarial_peers_banned == r2.adversarial_peers_banned


def test_different_seeds_can_differ():
    r1 = _run(seed=1, node_count=20, ticks=8)
    r2 = _run(seed=2, node_count=20, ticks=8)
    # With random request targeting, the totals may diverge across seeds.
    # We only assert that the harness HAS a non-trivial random surface —
    # i.e., the outcomes aren't seed-invariant.
    assert (
        r1.legit_requests_total != r2.legit_requests_total
        or r1.adversarial_requests_total != r2.adversarial_requests_total
    )
