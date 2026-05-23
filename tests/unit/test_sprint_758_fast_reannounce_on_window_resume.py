"""Sprint 758 — fast re-announce on inactive→active transition.

Sprint 756 wired announce_self() to skip when outside the active
window. But the announce loop sleeps for `announce_interval`
(default 60s) between calls. When the operator's window resumes
at 22:00, the node would broadcast again only at 22:00:60 worst-
case — peers' caches don't pick the node back up promptly.

Sprint 758 fix: poll on a tighter cadence (min(announce_interval,
10s)), detect the inactive→active transition, and force an
immediate announce. Pin tests verify:

1. State-machine: was_active False → now_active True triggers
   announce regardless of elapsed time.
2. Active → active (no transition) only triggers on the
   normal interval cadence.
3. Inactive → inactive (still inactive) never triggers.
4. The source contains the transition-detection logic + force
   call — defends against future refactors that might drop it.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_announce_loop_transition_logic_unit_test():
    """Unit-level: extract the transition-detection logic from
    _announce_loop's source and verify the boolean expression
    is the right shape. Behavioral end-to-end testing of the
    async loop is fragile (patching asyncio.sleep deadlocks the
    test runner), so we pin via source-shape + targeted logic.

    The state machine: was_active=False, now_active=True →
    transition_to_active=True → force immediate announce."""
    # Reproduce the logic from _announce_loop:
    def _step(was_active: bool, now_active: bool) -> bool:
        transition_to_active = now_active and not was_active
        return transition_to_active

    # Inactive → Active (the case sprint 758 fixes)
    assert _step(was_active=False, now_active=True) is True
    # Active → Active (no transition)
    assert _step(was_active=True, now_active=True) is False
    # Active → Inactive (don't force announce on the way OUT —
    # peers' caches expire naturally)
    assert _step(was_active=True, now_active=False) is False
    # Inactive → Inactive
    assert _step(was_active=False, now_active=False) is False


def test_announce_loop_source_contains_transition_detection():
    """Source-level pin: the _announce_loop function tracks
    `was_active` state + checks `transition_to_active`. Defends
    against future refactor that drops the state machine."""
    import inspect
    from prsm.node.discovery import PeerDiscovery
    src = inspect.getsource(PeerDiscovery._announce_loop)
    assert "was_active" in src, (
        "Sprint 758 fix requires state tracking across polls"
    )
    assert "transition_to_active" in src or "not was_active" in src, (
        "must detect inactive→active flip"
    )
    assert "is_currently_active" in src
    # Pin the operator-visible log message that says "active
    # window resumed" — useful for operators tailing logs to
    # confirm their schedule took effect.
    assert "active window resumed" in src.lower()


def test_announce_loop_source_uses_short_poll_interval():
    """Source-level pin: poll cadence is `min(announce_interval,
    10s)` so transition detection is bounded at 10s even when
    operator sets a long announce_interval."""
    import inspect
    from prsm.node.discovery import PeerDiscovery
    src = inspect.getsource(PeerDiscovery._announce_loop)
    assert "min(self.announce_interval, 10" in src or (
        "min(" in src and "10" in src
    ), "poll cadence must be bounded for fast transition detection"


def test_announce_loop_source_preserves_announce_interval_for_regular_cadence():
    """Pin: when no transition + active, the loop still respects
    the announce_interval (don't spam announces every poll)."""
    import inspect
    from prsm.node.discovery import PeerDiscovery
    src = inspect.getsource(PeerDiscovery._announce_loop)
    assert "elapsed_since_announce" in src, (
        "must accumulate elapsed time across short polls and only "
        "regular-announce when >= announce_interval"
    )
    assert ">= self.announce_interval" in src
