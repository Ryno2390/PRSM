"""Unit tests for ReputationTracker.

Phase 3 Task 6. Verifies:
  - New (unknown) providers → neutral 0.5 (no cold-start dead zone).
  - Providers with <10 samples → also neutral 0.5.
  - 10 successes, 0 failures → 1.0.
  - 5 successes, 5 failures → 0.5.
  - 9 successes, 1 failure → 0.9.
  - Preempted events do NOT enter the score denominator.
  - Rolling window of 1000 forgets the oldest.
  - Latency p50 and p95 computation over stored samples.
  - Latency returns None when no samples recorded.
  - known_providers() + get_reputation() introspection.
"""
from __future__ import annotations

from prsm.marketplace.reputation import ReputationTracker


def test_new_provider_is_neutral():
    tracker = ReputationTracker()
    assert tracker.score_for("unknown-provider") == 0.5


def test_below_min_samples_is_neutral():
    """Even after a few successes, a provider with <10 total samples
    returns NEUTRAL — not enough data to distinguish signal from noise."""
    tracker = ReputationTracker()
    for _ in range(5):
        tracker.record_success("p1", latency_ms=100.0)
    assert tracker.score_for("p1") == 0.5


def test_all_successes_is_1_0():
    tracker = ReputationTracker()
    for _ in range(20):
        tracker.record_success("p1", latency_ms=100.0)
    assert tracker.score_for("p1") == 1.0


def test_half_half_is_0_5():
    tracker = ReputationTracker()
    for _ in range(10):
        tracker.record_success("p1", latency_ms=100.0)
    for _ in range(10):
        tracker.record_failure("p1")
    assert tracker.score_for("p1") == 0.5


def test_nine_successes_one_failure_is_0_9():
    tracker = ReputationTracker()
    for _ in range(9):
        tracker.record_success("p1", latency_ms=100.0)
    tracker.record_failure("p1")
    assert tracker.score_for("p1") == 0.9


def test_preemption_does_not_affect_score():
    """Preemption is honest-work failure (Phase 2.1 Line A) — it must
    NOT lower the reputation score. A preempted-heavy provider keeps
    its existing score based solely on success/failure ratio."""
    tracker = ReputationTracker()
    for _ in range(10):
        tracker.record_success("p1", latency_ms=100.0)
    # Many preemptions, no failures.
    for _ in range(100):
        tracker.record_preemption("p1")
    assert tracker.score_for("p1") == 1.0

    rep = tracker.get_reputation("p1")
    assert len(rep.preempted_dispatches) == 100
    assert len(rep.successful_dispatches) == 10
    assert len(rep.failed_dispatches) == 0


def test_rolling_window_forgets_oldest():
    """Deque(maxlen=1000) drops the oldest element when a 1001st arrives.
    Over 1500 successes, we see exactly the last 1000."""
    tracker = ReputationTracker()
    for _ in range(1500):
        tracker.record_success("p1", latency_ms=100.0)
    rep = tracker.get_reputation("p1")
    assert len(rep.successful_dispatches) == 1000


def test_rolling_window_score_reflects_recent():
    """After 500 successes then 1500 failures, the score reflects the
    1000-sample window (all failures): score → 0.0."""
    tracker = ReputationTracker()
    for _ in range(500):
        tracker.record_success("p1", latency_ms=100.0)
    for _ in range(1500):
        tracker.record_failure("p1")
    # Successes capped at 500 (deque not full); failures capped at 1000.
    # Score = 500 / (500 + 1000) = 0.333...
    assert abs(tracker.score_for("p1") - 500 / 1500) < 1e-9


def test_latency_p50_simple():
    """Median of [10, 20, 30, 40, 50] is 30."""
    tracker = ReputationTracker()
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0]:
        tracker.record_success("p1", latency_ms=lat)
    assert tracker.latency_p50("p1") == 30.0


def test_latency_p95():
    """p95 of [10, 20, ..., 100] (10 samples). pos = 0.95 * 9 = 8.55 →
    interpolate between sample[8]=90 and sample[9]=100: 90 + 0.55*10 = 95.5."""
    tracker = ReputationTracker()
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        tracker.record_success("p1", latency_ms=lat)
    assert abs(tracker.latency_p95("p1") - 95.5) < 1e-9


def test_latency_returns_none_for_unknown():
    tracker = ReputationTracker()
    assert tracker.latency_p50("unknown") is None
    assert tracker.latency_p95("unknown") is None


def test_latency_returns_none_when_only_failures_recorded():
    tracker = ReputationTracker()
    for _ in range(5):
        tracker.record_failure("p1")
    assert tracker.latency_p50("p1") is None


def test_failures_do_not_add_latency():
    """Latency percentiles are from success samples only. Recording
    failures alongside successes doesn't dilute the latency stats."""
    tracker = ReputationTracker()
    for _ in range(5):
        tracker.record_success("p1", latency_ms=50.0)
    for _ in range(100):
        tracker.record_failure("p1")
    assert tracker.latency_p50("p1") == 50.0  # unaffected


def test_multiple_providers_tracked_independently():
    tracker = ReputationTracker()
    for _ in range(10):
        tracker.record_success("p1", latency_ms=100.0)
    for _ in range(10):
        tracker.record_failure("p2")

    assert tracker.score_for("p1") == 1.0
    assert tracker.score_for("p2") == 0.0
    known = set(tracker.known_providers())
    assert known == {"p1", "p2"}


def test_touch_sets_first_seen_and_last_seen():
    tracker = ReputationTracker()
    tracker.record_success("p1", latency_ms=100.0)
    rep = tracker.get_reputation("p1")
    assert rep.first_seen_unix > 0
    assert rep.last_seen_unix >= rep.first_seen_unix

    # Subsequent records update last_seen but not first_seen.
    first_seen_initial = rep.first_seen_unix
    tracker.record_success("p1", latency_ms=100.0)
    rep = tracker.get_reputation("p1")
    assert rep.first_seen_unix == first_seen_initial
