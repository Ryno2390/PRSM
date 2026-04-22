"""Unit tests for prsm.node.liveness + prsm.node.rate_limit.

Per docs/2026-04-22-phase6-p2p-hardening-design-plan.md §6 Task 5.

Both modules are clock-injected, so tests drive time via a mutable
[now] list rather than sleeping. This keeps the suite deterministic
and fast.
"""

from __future__ import annotations

import pytest

from prsm.node.liveness import LivenessMonitor, TickResult
from prsm.node.rate_limit import (
    DEFAULT_LIMITS,
    RateLimit,
    RateLimiter,
    RateLimitResult,
)


# =============================================================================
# LivenessMonitor
# =============================================================================


@pytest.fixture
def clock():
    return [1000.0]


@pytest.fixture
def monitor(clock):
    return LivenessMonitor(
        ping_interval_sec=30.0,
        dead_threshold=3,
        clock=lambda: clock[0],
    )


def test_register_and_is_alive(monitor):
    monitor.register("peer-a")
    assert monitor.is_alive("peer-a") is True
    assert monitor.is_alive("unknown") is False


def test_unregister_removes_peer(monitor):
    monitor.register("peer-a")
    monitor.unregister("peer-a")
    assert monitor.is_alive("peer-a") is False


def test_newly_registered_peer_is_due_for_first_ping(monitor):
    monitor.register("peer-a")
    result = monitor.tick()
    assert "peer-a" in result.due_for_ping
    assert result.evicted == []


def test_pong_before_interval_keeps_peer_alive(monitor, clock):
    monitor.register("peer-a")
    monitor.tick()  # due list advances
    monitor.record_ping_sent("peer-a")

    clock[0] += 10  # within interval
    monitor.record_pong_received("peer-a")

    clock[0] += 25
    result = monitor.tick()
    # Pong was within interval → no miss, but it's been >30s total since
    # the ping so peer is due for another.
    assert monitor.is_alive("peer-a") is True
    assert "peer-a" in result.due_for_ping


def test_three_missed_pings_evicts_peer(monitor, clock):
    monitor.register("peer-a")
    monitor.tick()
    monitor.record_ping_sent("peer-a")

    # Miss 1
    clock[0] += 30
    result = monitor.tick()
    assert "peer-a" not in result.evicted
    # After miss, tick also clears outstanding → peer now due again.
    assert "peer-a" in result.due_for_ping
    monitor.record_ping_sent("peer-a")

    # Miss 2
    clock[0] += 30
    result = monitor.tick()
    assert "peer-a" not in result.evicted
    monitor.record_ping_sent("peer-a")

    # Miss 3 → eviction
    clock[0] += 30
    result = monitor.tick()
    assert "peer-a" in result.evicted
    assert monitor.is_alive("peer-a") is False


def test_pong_resets_miss_counter(monitor, clock):
    monitor.register("peer-a")
    monitor.tick()
    monitor.record_ping_sent("peer-a")

    clock[0] += 30
    monitor.tick()  # miss 1
    monitor.record_ping_sent("peer-a")

    clock[0] += 30
    monitor.tick()  # miss 2
    monitor.record_ping_sent("peer-a")

    # Late pong lands before miss 3.
    clock[0] += 10
    monitor.record_pong_received("peer-a")

    clock[0] += 20
    monitor.tick()  # no outstanding ping, still alive
    assert monitor.is_alive("peer-a")


def test_evicted_peer_is_not_re_pinged(monitor, clock):
    monitor.register("peer-a")
    monitor.tick()
    monitor.record_ping_sent("peer-a")
    for _ in range(3):
        clock[0] += 30
        monitor.tick()
        if monitor.is_alive("peer-a"):
            monitor.record_ping_sent("peer-a")

    assert not monitor.is_alive("peer-a")
    result = monitor.tick()
    assert "peer-a" not in result.due_for_ping


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        LivenessMonitor(dead_threshold=0)


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        LivenessMonitor(ping_interval_sec=0)


# =============================================================================
# RateLimiter
# =============================================================================


@pytest.fixture
def rl_clock():
    return [2000.0]


@pytest.fixture
def limiter(rl_clock):
    return RateLimiter(
        limits={
            "dht": RateLimit(max_per_window=5, window_sec=60.0),
            "direct": RateLimit(max_per_window=10, window_sec=60.0),
        },
        throttle_duration_sec=60.0,
        ban_duration_sec=3600.0,
        violations_for_ban=3,
        violation_memory_sec=600.0,
        clock=lambda: rl_clock[0],
    )


def test_under_limit_returns_allowed(limiter):
    for _ in range(5):
        assert limiter.check_and_consume("peer-a", "dht") is RateLimitResult.ALLOWED


def test_at_limit_returns_over_limit_and_engages_throttle(limiter, rl_clock):
    for _ in range(5):
        limiter.check_and_consume("peer-a", "dht")
    # 6th request in same window → OVER_LIMIT + throttle engaged.
    assert limiter.check_and_consume("peer-a", "dht") is RateLimitResult.OVER_LIMIT
    # Subsequent requests during throttle window → THROTTLED.
    assert limiter.check_and_consume("peer-a", "dht") is RateLimitResult.THROTTLED


def test_throttle_expires(limiter, rl_clock):
    for _ in range(6):
        limiter.check_and_consume("peer-a", "dht")
    # Advance past throttle + window.
    rl_clock[0] += 120
    assert limiter.check_and_consume("peer-a", "dht") is RateLimitResult.ALLOWED


def test_different_categories_tracked_independently(limiter):
    for _ in range(5):
        limiter.check_and_consume("peer-a", "dht")
    # dht is maxed; direct still has room.
    assert limiter.check_and_consume("peer-a", "direct") is RateLimitResult.ALLOWED


def test_sliding_window_expires_old_requests(limiter, rl_clock):
    for _ in range(5):
        limiter.check_and_consume("peer-a", "dht")
    rl_clock[0] += 65  # past window, not yet throttled because no over-limit yet
    # Old requests should have aged out.
    assert limiter.check_and_consume("peer-a", "dht") is RateLimitResult.ALLOWED


def test_three_violations_trigger_ban(limiter, rl_clock):
    # Violation 1
    for _ in range(6):
        limiter.check_and_consume("peer-a", "dht")
    # Wait for throttle to expire, do it again.
    rl_clock[0] += 130

    # Violation 2
    for _ in range(6):
        limiter.check_and_consume("peer-a", "dht")
    rl_clock[0] += 130

    # Violation 3 → BAN on the over-limit request itself.
    for _ in range(5):
        limiter.check_and_consume("peer-a", "dht")
    result = limiter.check_and_consume("peer-a", "dht")
    assert result is RateLimitResult.BANNED


def test_banned_peer_stays_banned(limiter, rl_clock):
    # Force a ban quickly.
    for _ in range(3):
        for _ in range(6):
            limiter.check_and_consume("peer-a", "dht")
        rl_clock[0] += 130
    # Walk forward 30 minutes — still inside 1-hour ban.
    rl_clock[0] += 1800
    assert limiter.state_of("peer-a") is RateLimitResult.BANNED


def test_unban_clears_state(limiter, rl_clock):
    for _ in range(3):
        for _ in range(6):
            limiter.check_and_consume("peer-a", "dht")
        rl_clock[0] += 130
    assert limiter.state_of("peer-a") is RateLimitResult.BANNED

    limiter.unban("peer-a")
    assert limiter.state_of("peer-a") is RateLimitResult.ALLOWED


def test_unknown_category_raises(limiter):
    with pytest.raises(ValueError):
        limiter.check_and_consume("peer-a", "no-such-category")


def test_unknown_peer_state_is_allowed(limiter):
    assert limiter.state_of("never-seen") is RateLimitResult.ALLOWED


def test_default_limits_cover_plan_categories():
    """Sanity: plan §3.6 names dht, direct_message, shard_dispatch —
    DEFAULT_LIMITS should include all three."""
    assert "dht" in DEFAULT_LIMITS
    assert "direct_message" in DEFAULT_LIMITS
    assert "shard_dispatch" in DEFAULT_LIMITS
    assert DEFAULT_LIMITS["dht"].max_per_window == 100
    assert DEFAULT_LIMITS["direct_message"].max_per_window == 500
    assert DEFAULT_LIMITS["shard_dispatch"].max_per_window == 50


def test_violation_memory_expires_old_violations(rl_clock):
    """A peer that violated long ago + stayed clean should not be one
    trip away from ban."""
    limiter = RateLimiter(
        limits={"dht": RateLimit(max_per_window=5, window_sec=60.0)},
        throttle_duration_sec=60.0,
        ban_duration_sec=3600.0,
        violations_for_ban=3,
        violation_memory_sec=600.0,  # 10 minutes
        clock=lambda: rl_clock[0],
    )
    # Violations 1 + 2.
    for _ in range(6):
        limiter.check_and_consume("peer-a", "dht")
    rl_clock[0] += 130
    for _ in range(6):
        limiter.check_and_consume("peer-a", "dht")
    # Big gap — violations age out of memory window.
    rl_clock[0] += 1200

    # Violation 3 → should NOT ban because previous violations aged out.
    for _ in range(5):
        limiter.check_and_consume("peer-a", "dht")
    result = limiter.check_and_consume("peer-a", "dht")
    assert result is RateLimitResult.OVER_LIMIT  # throttle, not ban
