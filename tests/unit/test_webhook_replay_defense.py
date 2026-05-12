"""Sprint 284 — webhook replay protection.

Sprint 283 verified that a webhook signature came from the
vendor. Sprint 284 closes the next attack: capture-and-replay.
An attacker who intercepts a valid signed payload can replay
it indefinitely because the signature stays valid forever.

Two defenses (both work; both ship; both required):

  1. Timestamp window. Persona embeds the signing timestamp
     in its header (t=<unix>). If |now - t| > tolerance, the
     webhook is too old to accept. Default 300s tolerance
     matches Stripe / Coinbase Commerce convention.

  2. Signature-hash dedup ring. Vendor-agnostic. The
     signature value itself is a perfect replay token (it's
     cryptographically unique per body+timestamp). Bounded
     in-memory ring tracks recently-seen signatures; second
     occurrence → reject. Works for Onfido (no embedded ts)
     and as a belt-and-suspenders for Persona.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.webhook_replay_defense import (
    is_timestamp_fresh,
    WebhookReplayRing,
)


# ── Timestamp freshness ──────────────────────────────────


def test_timestamp_fresh_within_window():
    ok, reason = is_timestamp_fresh(
        ts_str="1700000000",
        current_time=1700000100.0,  # 100s later
        tolerance_sec=300,
    )
    assert ok is True
    assert reason == ""


def test_timestamp_too_old():
    ok, reason = is_timestamp_fresh(
        ts_str="1700000000",
        current_time=1700000400.0,  # 400s later, tolerance 300
        tolerance_sec=300,
    )
    assert ok is False
    assert "window" in reason.lower() or "old" in reason.lower()


def test_timestamp_too_far_in_future():
    """A timestamp in the future is also suspicious — protects
    against clock-skew exploitation."""
    ok, reason = is_timestamp_fresh(
        ts_str="1700000400",  # 400s in future
        current_time=1700000000.0,
        tolerance_sec=300,
    )
    assert ok is False


def test_timestamp_exactly_at_window():
    """Tolerance is inclusive — exactly tolerance_sec away
    is still fresh."""
    ok, _ = is_timestamp_fresh(
        ts_str="1700000000",
        current_time=1700000300.0,
        tolerance_sec=300,
    )
    assert ok is True


def test_timestamp_malformed():
    ok, reason = is_timestamp_fresh(
        ts_str="not-a-number",
        current_time=1700000000.0,
        tolerance_sec=300,
    )
    assert ok is False
    assert "parse" in reason.lower() or "malformed" in reason.lower()


def test_timestamp_empty():
    ok, _ = is_timestamp_fresh(
        ts_str="",
        current_time=1700000000.0,
        tolerance_sec=300,
    )
    assert ok is False


def test_timestamp_zero_tolerance_only_accepts_exact():
    """Zero tolerance is a degenerate but legal config —
    useful for deterministic test scenarios."""
    ok, _ = is_timestamp_fresh(
        ts_str="1700000000",
        current_time=1700000000.0,
        tolerance_sec=0,
    )
    assert ok is True
    ok, _ = is_timestamp_fresh(
        ts_str="1700000000",
        current_time=1700000001.0,
        tolerance_sec=0,
    )
    assert ok is False


def test_timestamp_float_string_ok():
    """Some vendors send fractional timestamps; should accept."""
    ok, _ = is_timestamp_fresh(
        ts_str="1700000000.5",
        current_time=1700000001.0,
        tolerance_sec=300,
    )
    assert ok is True


# ── WebhookReplayRing ────────────────────────────────────


def test_ring_first_record_returns_true():
    r = WebhookReplayRing(max_entries=10)
    assert r.record("sig-abc") is True


def test_ring_second_record_returns_false():
    r = WebhookReplayRing(max_entries=10)
    r.record("sig-abc")
    assert r.record("sig-abc") is False


def test_ring_distinct_tokens_independent():
    r = WebhookReplayRing(max_entries=10)
    assert r.record("sig-abc") is True
    assert r.record("sig-def") is True
    assert r.record("sig-abc") is False


def test_ring_seen_predicate():
    r = WebhookReplayRing(max_entries=10)
    assert r.seen("sig-abc") is False
    r.record("sig-abc")
    assert r.seen("sig-abc") is True


def test_ring_max_entries_evicts_oldest():
    """Bounded ring — when full, oldest entry is evicted.
    Re-seeing a long-ago signature after eviction is allowed
    (defense covers the recent-replay window; ancient
    signatures are presumed expired by the timestamp-window
    check too)."""
    r = WebhookReplayRing(max_entries=2)
    r.record("a")
    r.record("b")
    r.record("c")  # evicts "a"
    # "a" should be record-able again
    assert r.record("a") is True


def test_ring_validates_max_entries():
    with pytest.raises(ValueError):
        WebhookReplayRing(max_entries=0)


def test_ring_count():
    r = WebhookReplayRing(max_entries=10)
    assert r.count() == 0
    r.record("a")
    assert r.count() == 1
    r.record("a")  # duplicate; count unchanged
    assert r.count() == 1
    r.record("b")
    assert r.count() == 2


def test_ring_default_max_entries_practical():
    r = WebhookReplayRing()
    assert r._max_entries >= 1000


def test_ring_empty_token_rejected():
    """Empty token would deduplicate against everything — a
    misuse that would silently make the ring useless."""
    r = WebhookReplayRing(max_entries=10)
    with pytest.raises(ValueError):
        r.record("")


def test_ring_non_string_token_rejected():
    r = WebhookReplayRing(max_entries=10)
    with pytest.raises(ValueError):
        r.record(12345)  # type: ignore
