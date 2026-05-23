"""Sprint 760 — scenario integration test for active-window boundaries.

Sprint 755-758 ship the parser/evaluator/gates/CLI/fast-reannounce.
Sprint 759 documents in the runbook. Sprint 760 closes the arc
with a scenario test that walks the clock across a window
boundary and pins the transition behavior end-to-end.

Scenario:
  Schedule = "12:00-13:00 UTC" (1-hour window)

  11:30 → inactive (before)
  11:59:59 → inactive (right before start)
  12:00:00 → ACTIVE (transition: start IS inclusive, half-open)
  12:00:01 → active
  12:30:00 → active
  12:59:59 → active
  13:00:00 → INACTIVE (transition: end IS exclusive, half-open)
  13:30:00 → inactive (after)

Tests both non-cross-midnight + cross-midnight windows.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest


def _at(window, h, m, s=0):
    """Helper: was the schedule active at HH:MM:SS in its tz?"""
    tz = ZoneInfo(window.tz_name)
    return window.is_active(datetime(2026, 5, 23, h, m, s, tzinfo=tz))


def test_non_cross_midnight_window_boundary_sequence():
    """12:00-13:00 UTC walked across the boundary."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("12:00-13:00", "UTC")

    # Before window
    assert _at(w, 11, 30) is False
    assert _at(w, 11, 59, 59) is False

    # Start boundary (inclusive)
    assert _at(w, 12, 0, 0) is True
    assert _at(w, 12, 0, 1) is True

    # Inside window
    assert _at(w, 12, 30) is True
    assert _at(w, 12, 59, 59) is True

    # End boundary (exclusive)
    assert _at(w, 13, 0, 0) is False
    assert _at(w, 13, 30) is False


def test_cross_midnight_window_boundary_sequence():
    """22:00-08:00 UTC walked across both transitions."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "UTC")

    # During the day (outside window)
    assert _at(w, 12, 0) is False
    assert _at(w, 17, 0) is False
    assert _at(w, 21, 59, 59) is False

    # Evening start
    assert _at(w, 22, 0, 0) is True
    assert _at(w, 23, 30) is True

    # Through midnight
    assert _at(w, 0, 0) is True
    assert _at(w, 3, 0) is True
    assert _at(w, 7, 59, 59) is True

    # Morning end (exclusive)
    assert _at(w, 8, 0, 0) is False
    assert _at(w, 9, 0) is False


def test_timezone_boundary_with_dst_aware_check():
    """A schedule in America/New_York remains stable across DST
    transitions (the wall-clock semantics are what operators
    care about; the underlying UTC offset shifts but is_active
    follows local time)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "America/New_York")
    ny_tz = ZoneInfo("America/New_York")
    # Just after EDT begins (March 9 2026 in US): 22:00 NY = 02:00 UTC
    # Just before EDT ends (Nov 1 2026): same wall-clock semantics
    edt_time = datetime(2026, 7, 15, 23, 0, tzinfo=ny_tz)
    est_time = datetime(2026, 12, 15, 23, 0, tzinfo=ny_tz)
    assert w.is_active(edt_time) is True
    assert w.is_active(est_time) is True


def test_is_active_state_machine_walk():
    """Simulate sprint 758's _announce_loop state-machine across
    a boundary crossing. Pins that the transition detection logic
    behaves correctly when is_active flips."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("12:00-13:00", "UTC")
    utc = ZoneInfo("UTC")

    timestamps = [
        datetime(2026, 5, 23, 11, 50, tzinfo=utc),  # inactive
        datetime(2026, 5, 23, 11, 59, tzinfo=utc),  # inactive
        datetime(2026, 5, 23, 12, 0, tzinfo=utc),   # ← transition!
        datetime(2026, 5, 23, 12, 30, tzinfo=utc),  # active
        datetime(2026, 5, 23, 13, 0, tzinfo=utc),   # ← transition!
        datetime(2026, 5, 23, 13, 10, tzinfo=utc),  # inactive
    ]

    was_active = False
    transitions_to_active = 0
    transitions_to_inactive = 0
    for ts in timestamps:
        now_active = w.is_active(ts)
        if now_active and not was_active:
            transitions_to_active += 1
        if not now_active and was_active:
            transitions_to_inactive += 1
        was_active = now_active

    assert transitions_to_active == 1, (
        "must observe exactly one inactive→active transition "
        "(at 12:00)"
    )
    assert transitions_to_inactive == 1, (
        "must observe exactly one active→inactive transition "
        "(at 13:00)"
    )


def test_render_format_matches_runbook_examples():
    """The render() output must match the documentation examples
    in the operator runbook (so operators copy-pasting from the
    CLI see the same format they're documented to expect)."""
    from prsm.node.schedule import parse_active_window
    w = parse_active_window("22:00-08:00", "America/New_York")
    assert w.render() == "22:00-08:00 America/New_York"
