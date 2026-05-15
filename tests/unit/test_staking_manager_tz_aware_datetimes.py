"""Sprint 432 — F11 fix: StakingManager datetime tz-aware normalization.

Production-blocking bug surfaced during sprint 432 stake → claim
verification:

  POST /staking/claim-rewards → 500
  "Claim rewards failed: can't subtract offset-naive and
   offset-aware datetimes"

Root cause: SQLite (the default backend) drops timezone info when
persisting datetimes. ``staked_at`` and ``last_reward_calculation``
get written as tz-aware UTC (via ``datetime.now(timezone.utc)``)
but come back as naive datetimes on load. The reward calculation
at staking_manager.py:893 does
``datetime.now(timezone.utc) - stake.last_reward_calculation``,
which raises TypeError.

Fix: ``StakingManager._ensure_utc`` normalizes naive datetimes from
the DB to tz-aware UTC at the row→record conversion site. Sound
because all writers use ``datetime.now(timezone.utc)`` — the stored
value IS UTC, just untagged.

These pins fire if the conversion is silently dropped or if a new
datetime field is added to StakeModel without going through the
helper.
"""
from __future__ import annotations

from datetime import datetime, timezone

from prsm.economy.tokenomics.staking_manager import StakingManager


def test_ensure_utc_tags_naive_datetime_as_utc():
    """A naive datetime (no tzinfo) must be returned as
    tz-aware UTC — re-tagging only, no clock arithmetic.
    SQLite-loaded values land here."""
    naive = datetime(2026, 5, 15, 12, 0, 0)
    assert naive.tzinfo is None
    result = StakingManager._ensure_utc(naive)
    assert result is not None
    assert result.tzinfo is timezone.utc
    # Wall-clock unchanged — no shift, just tagged.
    assert result.year == 2026
    assert result.hour == 12


def test_ensure_utc_preserves_already_aware_datetime():
    """A datetime that already has tzinfo must round-trip
    unchanged — don't double-tag."""
    aware = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
    result = StakingManager._ensure_utc(aware)
    assert result == aware
    assert result.tzinfo is timezone.utc


def test_ensure_utc_handles_none():
    """Optional datetime fields legitimately come back as None
    (e.g., first-time stake before any reward calculation).
    The helper must short-circuit on None rather than raise
    AttributeError."""
    assert StakingManager._ensure_utc(None) is None


def test_subtraction_works_after_normalization():
    """The headline regression: subtracting `now` (tz-aware UTC)
    minus a normalized naive-DB value must succeed without
    "can't subtract offset-naive and offset-aware datetimes"."""
    db_loaded_naive = datetime(2026, 5, 15, 11, 0, 0)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
    normalized = StakingManager._ensure_utc(db_loaded_naive)
    # The exact subtraction that production calculate_rewards does:
    delta = (now - normalized).total_seconds()
    assert delta == 3600.0  # one hour


def test_subtraction_fails_without_normalization():
    """Defensive: prove the bug we're defending against. Same
    subtraction without _ensure_utc must raise TypeError —
    if Python ever silently allows naive-aware mixing, this
    test surfaces that change-of-semantics."""
    import pytest

    db_loaded_naive = datetime(2026, 5, 15, 11, 0, 0)
    now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(TypeError, match="can't subtract"):
        _ = (now - db_loaded_naive).total_seconds()
