"""Sprint 191 — shared _iso_ts helper handles datetime, float, None.

Multiple dashboard handlers crashed with `AttributeError: 'float'
object has no attribute 'isoformat'` because the underlying
domain objects (LocalLedger transactions, compute_requester
submitted_jobs) carry timestamps as Unix floats. Sprint 190
fixed one handler inline; sprint 191 extracts the coercion to a
shared helper + applies to /api/jobs + /api/jobs/{job_id}.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_iso_ts_none():
    from prsm.dashboard.app import _iso_ts
    assert _iso_ts(None) is None


def test_iso_ts_datetime():
    from prsm.dashboard.app import _iso_ts
    dt = datetime(2026, 5, 11, 15, 8, 59, 540958, tzinfo=timezone.utc)
    iso = _iso_ts(dt)
    assert iso == "2026-05-11T15:08:59.540958+00:00"


def test_iso_ts_float_unix():
    from prsm.dashboard.app import _iso_ts
    ts = 1778559140.0  # arbitrary unix ts in 2026
    iso = _iso_ts(ts)
    assert iso.startswith("2026-")
    assert iso.endswith("+00:00")


def test_iso_ts_int_unix():
    from prsm.dashboard.app import _iso_ts
    iso = _iso_ts(1778559140)
    assert iso.startswith("2026-")
    assert iso.endswith("+00:00")


def test_iso_ts_invalid_returns_none():
    """Defensive — non-coercible input doesn't crash, just None."""
    from prsm.dashboard.app import _iso_ts
    assert _iso_ts("not-a-timestamp") is None
    assert _iso_ts([1, 2, 3]) is None
    assert _iso_ts({"a": 1}) is None


def test_iso_ts_handles_naive_datetime():
    """datetime without tzinfo still produces ISO output (without
    the +00:00 suffix, but no crash)."""
    from prsm.dashboard.app import _iso_ts
    dt = datetime(2026, 5, 11, 15, 0, 0)  # naive
    iso = _iso_ts(dt)
    assert "2026" in iso
