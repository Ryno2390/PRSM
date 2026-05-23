"""Sprint 755 — operator-controlled active-window scheduling.

Operators want to express "my Mac is in the pool overnight but
not during working hours". Pre-755 the daemon had no way to
honor that without manual systemd start/stop scripts.

This module ships the foundation: parse a simple `HH:MM-HH:MM`
spec + a timezone name, evaluate `is_active(now)`. Sprint 756
will wire this into the daemon to gate inference dispatch +
discovery announces.

Wire format:
    PRSM_ACTIVE_HOURS=22:00-08:00
    PRSM_ACTIVE_TIMEZONE=America/New_York

Semantics:
    Non-cross-midnight: "09:00-17:00" → active 9am to 5pm
    Cross-midnight:     "22:00-08:00" → active 10pm to 8am
    Half-open interval: [start, end) — at exactly `start` →
        active; at exactly `end` → inactive. Matches the
        Python `range()` convention + avoids the off-by-one
        ambiguity at midnight.

Operators with no schedule (env unset) get the existing
always-on behavior. `resolve_active_window_from_env()` returns
None when unset → the daemon should treat None as "always
active" (backward-compatible default).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional


try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — Python 3.8 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore


@dataclass(frozen=True)
class ActiveWindow:
    """A recurring daily time-of-day window when the node is
    active in the network.

    Frozen so two schedules can be compared by value (useful for
    operator UX: "is the running schedule the one I set?")."""

    start: time
    end: time
    tz_name: str  # store the name (not the ZoneInfo) for hashability

    def is_active(self, now: Optional[datetime] = None) -> bool:
        """Return True iff `now` falls within the window.

        If `now` is None, uses the current time in the schedule's
        timezone. If `now` is provided as an aware datetime, it's
        converted to the schedule's timezone before comparison.
        Naive datetimes are treated as being in the schedule's
        timezone already (lets tests pass simple naive times).
        """
        tz = ZoneInfo(self.tz_name)
        if now is None:
            now = datetime.now(tz)
        elif now.tzinfo is None:
            # Naive — assume schedule's timezone
            now = now.replace(tzinfo=tz)
        else:
            now = now.astimezone(tz)
        current = now.time()
        if self.start <= self.end:
            # Non-cross-midnight: e.g., 09:00-17:00 → active
            # iff start <= current < end.
            return self.start <= current < self.end
        # Cross-midnight: e.g., 22:00-08:00 → active iff
        # current >= start OR current < end.
        return current >= self.start or current < self.end

    def render(self) -> str:
        """Human-readable form: 'HH:MM-HH:MM TZ'."""
        return (
            f"{self.start.strftime('%H:%M')}-"
            f"{self.end.strftime('%H:%M')} {self.tz_name}"
        )


def _parse_time(s: str) -> time:
    """Parse 'HH:MM' into datetime.time. Raises ValueError."""
    if ":" not in s:
        raise ValueError(
            f"time must be 'HH:MM', got {s!r}"
        )
    h_str, _, m_str = s.partition(":")
    try:
        h = int(h_str)
        m = int(m_str)
    except ValueError as exc:
        raise ValueError(
            f"time digits must be integers, got {s!r}"
        ) from exc
    if not (0 <= h <= 23):
        raise ValueError(
            f"hour must be in [0, 23], got {h} from {s!r}"
        )
    if not (0 <= m <= 59):
        raise ValueError(
            f"minute must be in [0, 59], got {m} from {s!r}"
        )
    return time(h, m)


def parse_active_window(spec: str, tz_name: str = "UTC") -> ActiveWindow:
    """Parse an `HH:MM-HH:MM` spec + timezone name into an
    ActiveWindow.

    Raises ValueError on malformed spec or unknown timezone.
    """
    if not spec or not spec.strip():
        raise ValueError(
            "active-window spec must be non-empty"
        )
    spec = spec.strip()
    if "-" not in spec:
        raise ValueError(
            f"spec must be 'HH:MM-HH:MM', got {spec!r}"
        )
    start_str, _, end_str = spec.partition("-")
    start = _parse_time(start_str.strip())
    end = _parse_time(end_str.strip())
    # Validate timezone — raises ZoneInfoNotFoundError if unknown.
    try:
        ZoneInfo(tz_name)
    except Exception as exc:
        raise ValueError(
            f"invalid timezone {tz_name!r}: {exc}"
        ) from exc
    # Edge case: start == end. Ambiguous (zero-length window or
    # always-active?). Reject — operators can express "always
    # active" by unsetting the env, and zero-length is useless.
    if start == end:
        raise ValueError(
            f"active window start == end ({spec!r}); use empty "
            f"PRSM_ACTIVE_HOURS to get always-active"
        )
    return ActiveWindow(start=start, end=end, tz_name=tz_name)


def resolve_active_window_from_env() -> Optional[ActiveWindow]:
    """Read PRSM_ACTIVE_HOURS + PRSM_ACTIVE_TIMEZONE.

    Returns:
        None — env unset → backward-compatible always-active
        ActiveWindow — schedule parsed successfully

    Raises:
        ValueError — env is set but malformed; operator deserves
        a startup-time error rather than silently ignoring their
        intent.
    """
    spec = os.environ.get("PRSM_ACTIVE_HOURS", "").strip()
    if not spec:
        return None
    tz_name = (
        os.environ.get("PRSM_ACTIVE_TIMEZONE", "").strip()
        or "UTC"
    )
    return parse_active_window(spec, tz_name)
