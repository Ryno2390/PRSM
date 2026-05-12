"""Sprint 287 — CreatorReputationTracker.

Vision §14 "Data quality and Sybil resistance" mitigation
item (1): "Reputation scoring on creators based on access
frequency and repeat purchases." Distinct from the
provider-side ReputationTracker (Phase 3 Task 6) which scores
compute providers — this is the content-marketplace side,
scoring uploaders.

Spam pattern this defends against: creator uploads 1000
pieces of content, each accessed once by a different account
(no repeats, no engagement). Spam uploads earn FTNS but
should NOT earn reputation. The repeat-purchase signal is
the discriminator.

Score formula (v1, tunable):
  - cold-start: < MIN_SAMPLES_FOR_SCORE access events →
    NEUTRAL_SCORE (0.5)
  - weighted average:
      0.6 * reach_score    (log10 of distinct purchasers,
                            normalized so 100 unique → 1.0)
      0.4 * repeat_score   (fraction of purchasers who
                            accessed ≥2 pieces)

Memory bound: per-creator purchaser_counts capped at 1000
distinct ids; eviction is FIFO (oldest seen evicted first).
"""
from __future__ import annotations

import math
import pytest

from prsm.marketplace.creator_reputation import (
    CreatorReputationEntry,
    CreatorReputationTracker,
)


# ── Cold-start ───────────────────────────────────────────


def test_score_unknown_creator_is_neutral():
    t = CreatorReputationTracker()
    assert t.score_for("nobody") == 0.5


def test_score_below_min_samples_is_neutral():
    t = CreatorReputationTracker()
    # Record only a few accesses — under MIN_SAMPLES
    for i in range(5):
        t.record_access(
            creator_id="alice",
            purchaser_id=f"p{i}",
            content_id=f"c{i}",
        )
    assert t.score_for("alice") == 0.5


# ── Record access ────────────────────────────────────────


def test_record_access_updates_counters():
    t = CreatorReputationTracker()
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1",
    )
    assert t.access_count("alice") == 1
    assert t.distinct_purchasers("alice") == 1


def test_record_access_validates_required_fields():
    t = CreatorReputationTracker()
    with pytest.raises(ValueError):
        t.record_access(
            creator_id="", purchaser_id="p", content_id="c",
        )
    with pytest.raises(ValueError):
        t.record_access(
            creator_id="a", purchaser_id="", content_id="c",
        )
    with pytest.raises(ValueError):
        t.record_access(
            creator_id="a", purchaser_id="p", content_id="",
        )


def test_record_access_tracks_repeat_purchasers():
    t = CreatorReputationTracker()
    # Bob accesses 3 of alice's pieces
    for cid in ["c1", "c2", "c3"]:
        t.record_access(
            creator_id="alice", purchaser_id="bob",
            content_id=cid,
        )
    # Carol accesses 1 piece
    t.record_access(
        creator_id="alice", purchaser_id="carol",
        content_id="c4",
    )
    assert t.access_count("alice") == 4
    assert t.distinct_purchasers("alice") == 2
    # 1 of 2 purchasers (bob) is a repeat
    assert t.repeat_purchaser_count("alice") == 1


def test_record_access_distinct_content_id_required_for_repeat():
    """A purchaser accessing the SAME content multiple times
    counts as repeated access but does NOT inflate the
    'distinct content explored' signal. The tracker should
    still distinguish."""
    t = CreatorReputationTracker()
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1",
    )
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1",
    )
    # Total accesses incremented
    assert t.access_count("alice") == 2
    # But still 1 distinct purchaser
    assert t.distinct_purchasers("alice") == 1


# ── Score formula ────────────────────────────────────────


def test_score_high_for_broad_repeat_engagement():
    """50 distinct purchasers, 30 are repeats → near-max
    score."""
    t = CreatorReputationTracker()
    for i in range(50):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c0",
        )
    # 30 of them come back for a second piece
    for i in range(30):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c1",
        )
    score = t.score_for("alice")
    assert 0.65 < score <= 1.0


def test_score_low_for_spam_pattern():
    """50 accesses, all distinct purchasers, ZERO repeats.
    Spam pattern: broad shallow engagement, no real value.
    Score should be capped at the reach-only ceiling
    (0.6 weight on reach + 0 weight on repeat)."""
    t = CreatorReputationTracker()
    for i in range(50):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id=f"c{i}",
        )
    score = t.score_for("alice")
    # No repeats → repeat_score = 0 → max possible = 0.6
    assert score <= 0.61


def test_score_zero_repeats_explicit():
    """Spam pattern: 100 unique purchasers, all single-use.
    repeat_score=0, reach_score=1 → 0.6 * 1 + 0.4 * 0 = 0.6"""
    t = CreatorReputationTracker()
    for i in range(100):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id=f"c{i}",
        )
    score = t.score_for("alice")
    assert abs(score - 0.6) < 0.05


def test_score_small_audience_perfect_repeat():
    """10 purchasers, ALL repeats (everyone came back).
    reach is small (log10(10) / 2 = 0.5) but repeat=1.0.
    Expected: 0.6 * 0.5 + 0.4 * 1.0 = 0.7"""
    t = CreatorReputationTracker()
    for i in range(10):
        # First access
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c0",
        )
    for i in range(10):
        # All come back
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c1",
        )
    score = t.score_for("alice")
    assert 0.65 < score < 0.75


def test_score_clipped_to_unit_interval():
    """Pathological case: thousands of repeat purchasers.
    Score must stay in [0, 1]."""
    t = CreatorReputationTracker()
    for i in range(500):
        for c in ["c0", "c1", "c2"]:
            t.record_access(
                creator_id="alice", purchaser_id=f"p{i}",
                content_id=c,
            )
    score = t.score_for("alice")
    assert 0.0 <= score <= 1.0


# ── Repeat-purchase rate ─────────────────────────────────


def test_repeat_purchase_rate_zero_when_no_repeats():
    t = CreatorReputationTracker()
    for i in range(20):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id=f"c{i}",
        )
    assert t.repeat_purchase_rate("alice") == 0.0


def test_repeat_purchase_rate_one_when_all_repeat():
    t = CreatorReputationTracker()
    for i in range(10):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c0",
        )
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c1",
        )
    assert t.repeat_purchase_rate("alice") == 1.0


def test_repeat_purchase_rate_unknown_creator_zero():
    t = CreatorReputationTracker()
    assert t.repeat_purchase_rate("nobody") == 0.0


# ── Distinct purchasers / access count ───────────────────


def test_access_count_unknown_creator_zero():
    t = CreatorReputationTracker()
    assert t.access_count("nobody") == 0


def test_distinct_purchasers_unknown_creator_zero():
    t = CreatorReputationTracker()
    assert t.distinct_purchasers("nobody") == 0


# ── Known creators enumeration ───────────────────────────


def test_known_creators_empty():
    t = CreatorReputationTracker()
    assert t.known_creators() == []


def test_known_creators_returns_all():
    t = CreatorReputationTracker()
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1",
    )
    t.record_access(
        creator_id="carol", purchaser_id="dave",
        content_id="c2",
    )
    creators = sorted(t.known_creators())
    assert creators == ["alice", "carol"]


# ── Entry detail ─────────────────────────────────────────


def test_get_entry_returns_record():
    t = CreatorReputationTracker()
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1", timestamp=100.0,
    )
    e = t.get_entry("alice")
    assert isinstance(e, CreatorReputationEntry)
    assert e.creator_id == "alice"
    assert e.total_accesses == 1
    assert e.first_seen_unix == 100
    assert e.last_seen_unix == 100


def test_get_entry_unknown_returns_none():
    t = CreatorReputationTracker()
    assert t.get_entry("nobody") is None


def test_entry_to_dict_round_trip():
    e = CreatorReputationEntry(
        creator_id="alice",
        total_accesses=10,
        purchaser_counts={"bob": 3, "carol": 1},
        first_seen_unix=100,
        last_seen_unix=200,
    )
    d = e.to_dict()
    assert d["creator_id"] == "alice"
    assert d["total_accesses"] == 10
    assert d["distinct_purchasers"] == 2
    # purchaser_counts NOT serialized (privacy + size); only
    # aggregates surface
    assert "purchaser_counts" not in d
    assert d["repeat_purchaser_count"] == 1


# ── Purchaser-counts bounded ─────────────────────────────


def test_purchaser_counts_bounded():
    """Per-creator purchaser_counts must be bounded to avoid
    unbounded memory growth from a creator with millions of
    one-time purchasers."""
    t = CreatorReputationTracker(max_purchasers_per_creator=10)
    for i in range(20):
        t.record_access(
            creator_id="alice", purchaser_id=f"p{i}",
            content_id="c0",
        )
    # Only the most-recent 10 purchasers retained
    e = t.get_entry("alice")
    assert len(e.purchaser_counts) <= 10
    # access_count still reflects the full 20 events (running
    # counter; bounded only the distinct-purchaser map)
    assert t.access_count("alice") == 20


def test_purchaser_counts_default_bound_practical():
    t = CreatorReputationTracker()
    assert t._max_purchasers_per_creator >= 1000


# ── Timestamp tracking ───────────────────────────────────


def test_timestamps_default_to_now():
    import time as _time
    t = CreatorReputationTracker()
    before = int(_time.time())
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1",
    )
    after = int(_time.time())
    e = t.get_entry("alice")
    assert before <= e.first_seen_unix <= after
    assert before <= e.last_seen_unix <= after


def test_explicit_timestamp_threaded_through():
    t = CreatorReputationTracker()
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1", timestamp=42.0,
    )
    e = t.get_entry("alice")
    assert e.first_seen_unix == 42
    assert e.last_seen_unix == 42


def test_subsequent_access_updates_last_seen():
    t = CreatorReputationTracker()
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c1", timestamp=100.0,
    )
    t.record_access(
        creator_id="alice", purchaser_id="bob",
        content_id="c2", timestamp=200.0,
    )
    e = t.get_entry("alice")
    assert e.first_seen_unix == 100
    assert e.last_seen_unix == 200


# ── Score weights tunable via module constants ───────────


def test_score_constants_exposed():
    """Module-level weights so future sprints can tune
    without diving into score_for."""
    from prsm.marketplace import creator_reputation
    assert hasattr(creator_reputation, "REACH_WEIGHT")
    assert hasattr(creator_reputation, "REPEAT_WEIGHT")
    assert hasattr(creator_reputation, "MIN_SAMPLES_FOR_SCORE")
    assert hasattr(creator_reputation, "NEUTRAL_SCORE")
    # Weights sum to 1.0
    assert abs(
        creator_reputation.REACH_WEIGHT
        + creator_reputation.REPEAT_WEIGHT
        - 1.0
    ) < 1e-9
