"""Sprint 755 — operator-controlled active-window schedule.

Foundation for the multi-sprint scheduling arc. Tests:

- Parse 'HH:MM-HH:MM' into ActiveWindow
- Timezone resolution (default UTC; explicit named tz like
  'America/New_York')
- is_active() across:
  - non-cross-midnight windows (09:00-17:00)
  - cross-midnight windows (22:00-08:00)
  - half-open interval edges (at start = active; at end = inactive)
  - aware-datetime conversion to schedule's tz
  - naive-datetime treated as schedule's tz
- Env resolution: unset → None; set with malformed spec → ValueError
- Operator UX edge case: start == end rejected (zero-length window)
- Frozen dataclass: ActiveWindow values are hashable + comparable
"""
from __future__ import annotations

import os
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest


# ---- Parser tests ---------------------------------------------------


def test_parse_non_cross_midnight_window():
    """'09:00-17:00' → start=09:00, end=17:00."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("09:00-17:00", "UTC")
    assert w.start == time(9, 0)
    assert w.end == time(17, 0)
    assert w.tz_name == "UTC"


def test_parse_cross_midnight_window():
    """'22:00-08:00' → start=22:00, end=08:00 (overnight)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "America/New_York")
    assert w.start == time(22, 0)
    assert w.end == time(8, 0)
    assert w.tz_name == "America/New_York"


def test_parse_tolerates_whitespace():
    """Operators copy-paste from configs with stray whitespace."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("  22:00 - 08:00  ", "UTC")
    assert w.start == time(22, 0)
    assert w.end == time(8, 0)


def test_parse_rejects_empty_spec():
    """Empty spec → ValueError (operator must unset env for
    backward-compat always-active behavior)."""
    from prsm.node.schedule import parse_active_window
    with pytest.raises(ValueError, match="non-empty"):
        parse_active_window("", "UTC")


def test_parse_rejects_missing_dash():
    """'22:00 08:00' (space-separated) → ValueError."""
    from prsm.node.schedule import parse_active_window
    with pytest.raises(ValueError, match="HH:MM-HH:MM"):
        parse_active_window("22:00 08:00", "UTC")


def test_parse_rejects_malformed_time():
    """'25:00-08:00' (hour out of range) → ValueError."""
    from prsm.node.schedule import parse_active_window
    with pytest.raises(ValueError, match="hour"):
        parse_active_window("25:00-08:00", "UTC")


def test_parse_rejects_non_numeric_time():
    """'twentytwo:00-08:00' → ValueError."""
    from prsm.node.schedule import parse_active_window
    with pytest.raises(ValueError, match="integers"):
        parse_active_window("twentytwo:00-08:00", "UTC")


def test_parse_rejects_unknown_timezone():
    """Bogus tz → ValueError naming the offending tz."""
    from prsm.node.schedule import parse_active_window
    with pytest.raises(ValueError, match="Bogus/Timezone"):
        parse_active_window("22:00-08:00", "Bogus/Timezone")


def test_parse_rejects_zero_length_window():
    """start == end is ambiguous (zero-length OR always-active);
    operator gets a clear error instead of silent surprise."""
    from prsm.node.schedule import parse_active_window
    with pytest.raises(ValueError, match="start == end"):
        parse_active_window("12:00-12:00", "UTC")


# ---- is_active() evaluation tests ----------------------------------


def test_is_active_non_cross_midnight_inside():
    """09:00-17:00 + now=12:00 → True."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("09:00-17:00", "UTC")
    assert w.is_active(datetime(2026, 5, 23, 12, 0))


def test_is_active_non_cross_midnight_before_start():
    """09:00-17:00 + now=08:00 → False."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("09:00-17:00", "UTC")
    assert not w.is_active(datetime(2026, 5, 23, 8, 0))


def test_is_active_non_cross_midnight_at_end():
    """09:00-17:00 + now=17:00 → False (half-open interval)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("09:00-17:00", "UTC")
    assert not w.is_active(datetime(2026, 5, 23, 17, 0))


def test_is_active_non_cross_midnight_at_start():
    """09:00-17:00 + now=09:00 → True (half-open interval)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("09:00-17:00", "UTC")
    assert w.is_active(datetime(2026, 5, 23, 9, 0))


def test_is_active_cross_midnight_overnight():
    """22:00-08:00 + now=02:00 → True."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "UTC")
    assert w.is_active(datetime(2026, 5, 23, 2, 0))


def test_is_active_cross_midnight_evening():
    """22:00-08:00 + now=23:30 → True."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "UTC")
    assert w.is_active(datetime(2026, 5, 23, 23, 30))


def test_is_active_cross_midnight_during_working_hours():
    """22:00-08:00 + now=12:00 → False (the user's example: Mac
    NOT in the pool during the workday)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "UTC")
    assert not w.is_active(datetime(2026, 5, 23, 12, 0))


def test_is_active_cross_midnight_at_end():
    """22:00-08:00 + now=08:00 → False (half-open)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "UTC")
    assert not w.is_active(datetime(2026, 5, 23, 8, 0))


def test_is_active_with_aware_datetime_converts_timezone():
    """Schedule in America/New_York; `now` in UTC → converted
    before comparison. 22:00 NY = 02:00 UTC next day; so a UTC
    timestamp of 02:00 should be evaluated against NY's 22:00
    (which IS in 22:00-08:00 window)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "America/New_York")
    # 02:00 UTC = 22:00 NY (during EDT, which is May)
    utc_now = datetime(2026, 5, 24, 2, 0, tzinfo=ZoneInfo("UTC"))
    assert w.is_active(utc_now)


def test_is_active_with_naive_datetime_treated_as_schedule_tz():
    """When `now` has no tzinfo, treat it as already in the
    schedule's timezone. Lets simple unit tests pass naive times."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("09:00-17:00", "America/New_York")
    # Naive 14:00 → treated as 14:00 NY → in 09:00-17:00 window
    assert w.is_active(datetime(2026, 5, 23, 14, 0))


def test_is_active_defaults_to_now():
    """is_active() with no argument uses datetime.now(tz)."""
    from prsm.node.schedule import parse_active_window
    # Schedule that's always active relative to "now": create a
    # 1-minute wide window centered on the current minute in UTC.
    now_utc = datetime.now(ZoneInfo("UTC"))
    # Construct a window covering [now-30min, now+30min] (non-cross-
    # midnight assumption — could fail near midnight UTC, accept).
    start_min = (now_utc.hour * 60 + now_utc.minute - 30) % (24 * 60)
    end_min = (now_utc.hour * 60 + now_utc.minute + 30) % (24 * 60)
    spec = (
        f"{start_min // 60:02d}:{start_min % 60:02d}-"
        f"{end_min // 60:02d}:{end_min % 60:02d}"
    )
    w = parse_active_window(spec, "UTC")
    # Should be active because window spans current time
    assert w.is_active()


# ---- Env-resolution tests ------------------------------------------


def test_env_resolution_unset_returns_none():
    """No PRSM_ACTIVE_HOURS → None (backward-compatible always-on)."""
    from prsm.node.schedule import resolve_active_window_from_env
    old = os.environ.pop("PRSM_ACTIVE_HOURS", None)
    try:
        assert resolve_active_window_from_env() is None
    finally:
        if old is not None:
            os.environ["PRSM_ACTIVE_HOURS"] = old


def test_env_resolution_set_returns_window():
    """PRSM_ACTIVE_HOURS=22:00-08:00 → ActiveWindow with UTC default."""
    from prsm.node.schedule import resolve_active_window_from_env
    os.environ["PRSM_ACTIVE_HOURS"] = "22:00-08:00"
    os.environ.pop("PRSM_ACTIVE_TIMEZONE", None)
    try:
        w = resolve_active_window_from_env()
        assert w is not None
        assert w.start == time(22, 0)
        assert w.end == time(8, 0)
        assert w.tz_name == "UTC"
    finally:
        del os.environ["PRSM_ACTIVE_HOURS"]


def test_env_resolution_with_timezone():
    """PRSM_ACTIVE_HOURS + PRSM_ACTIVE_TIMEZONE → uses explicit tz."""
    from prsm.node.schedule import resolve_active_window_from_env
    os.environ["PRSM_ACTIVE_HOURS"] = "22:00-08:00"
    os.environ["PRSM_ACTIVE_TIMEZONE"] = "America/New_York"
    try:
        w = resolve_active_window_from_env()
        assert w is not None
        assert w.tz_name == "America/New_York"
    finally:
        del os.environ["PRSM_ACTIVE_HOURS"]
        del os.environ["PRSM_ACTIVE_TIMEZONE"]


def test_env_resolution_malformed_raises():
    """Malformed PRSM_ACTIVE_HOURS → ValueError at startup
    (operator deserves a clear error rather than silent ignore)."""
    from prsm.node.schedule import resolve_active_window_from_env
    os.environ["PRSM_ACTIVE_HOURS"] = "not-a-time"
    try:
        with pytest.raises(ValueError):
            resolve_active_window_from_env()
    finally:
        del os.environ["PRSM_ACTIVE_HOURS"]


def test_render_human_readable():
    """ActiveWindow.render() → 'HH:MM-HH:MM TZ' for operator UX."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "America/New_York")
    assert w.render() == "22:00-08:00 America/New_York"


def test_active_window_is_frozen_and_hashable():
    """Frozen dataclass → can be used as dict key / in sets, and
    two equivalent schedules compare equal (enables operator UX
    'is the running schedule the one I set?')."""
    from prsm.node.schedule import parse_active_window
    w1 = parse_active_window("22:00-08:00", "UTC")
    w2 = parse_active_window("22:00-08:00", "UTC")
    assert w1 == w2
    assert hash(w1) == hash(w2)
    assert {w1, w2} == {w1}
