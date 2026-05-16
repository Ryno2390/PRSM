"""Sprint 480 — F22 ProvenanceQueries hydration regression pin.

F22: ProvenanceQueries.load_all_for_node failed at every daemon
startup with `'str' object has no attribute 'timestamp'`.

Root cause (same class as F11 — sprint 432, StakingManager
tz-aware datetime subtract): SQLite stores DATETIME columns as
ISO-format strings (`'2026-05-15 12:52:27.363797'`), not native
Python `datetime` objects. The list-comprehension at line 1587
called `row.created_at.timestamp()` which raised AttributeError
on the first row.

The defensive try/except at line 1591 caught the exception and
returned [] silently. **Live impact**: uploaded_content dict
empty after every daemon restart → operator can't see
previously-uploaded content via `/content/mine` until they
re-upload (or hit the per-CID query path).

Sprint 480 fix: `_row_created_at_to_epoch()` helper normalizes
both native datetime objects (Postgres path) and ISO strings
(SQLite path) to a Unix-epoch float. Live-verified: daemon
restart → /content/mine returned 0 → 14 entries (full content
history restored).

These pins defend the helper + its integration with
load_all_for_node so a future refactor can't silently
reintroduce the str→.timestamp() crash.
"""
from __future__ import annotations

from datetime import datetime, timezone


from prsm.core.database import _row_created_at_to_epoch


def test_none_returns_zero():
    assert _row_created_at_to_epoch(None) == 0.0


def test_native_datetime_returns_epoch():
    """Postgres path — DATETIME columns come back as native
    datetime objects with .timestamp() callable."""
    dt = datetime(2026, 5, 15, 12, 52, 27, 363797)
    result = _row_created_at_to_epoch(dt)
    assert result == dt.timestamp()
    assert isinstance(result, float)


def test_iso_string_with_microseconds_returns_epoch():
    """SQLite path with microseconds — the canonical SQLite
    DATETIME format on PRSM's deployment."""
    s = "2026-05-15 12:52:27.363797"
    result = _row_created_at_to_epoch(s)
    expected = datetime.strptime(
        s, "%Y-%m-%d %H:%M:%S.%f",
    ).timestamp()
    assert result == expected
    assert isinstance(result, float)


def test_iso_string_without_microseconds():
    """SQLite may emit datetimes without fractional seconds
    on some operations — the helper must accept both."""
    s = "2026-05-15 12:52:27"
    result = _row_created_at_to_epoch(s)
    expected = datetime.strptime(
        s, "%Y-%m-%d %H:%M:%S",
    ).timestamp()
    assert result == expected


def test_iso_string_with_T_separator():
    """ISO 8601 format with `T` separator — defensive
    against future migrations that store ISO 8601 instead
    of SQLite's space-separated default."""
    s = "2026-05-15T12:52:27.363797"
    result = _row_created_at_to_epoch(s)
    expected = datetime.fromisoformat(s).timestamp()
    assert result == expected


def test_unparseable_string_returns_zero():
    """Defensive: junk strings return 0.0 (matches pre-F22
    None-handling). Logging the bad value is a future
    enhancement but the function must not crash."""
    assert _row_created_at_to_epoch("not a date") == 0.0
    assert _row_created_at_to_epoch("") == 0.0
    assert _row_created_at_to_epoch("   ") == 0.0


def test_load_all_for_node_no_longer_crashes_on_string_dates(
    tmp_path, monkeypatch,
):
    """Integration pin: the list-comprehension at line 1587
    must NOT raise AttributeError when `row.created_at` is a
    string. We can't easily mock SQLAlchemy rows; instead,
    verify the helper handles the literal string values
    SQLite is known to return.

    The sprint 480 fix replaces `row.created_at.timestamp()`
    with `_row_created_at_to_epoch(row.created_at)`. If a
    future refactor calls `.timestamp()` directly again, this
    pin won't catch it — but the test on the helper itself
    (above) defends the centralized normalization."""
    # Reproduce the exact pattern from line 1587 — must NOT
    # raise.
    sqlite_value = "2026-05-15 12:52:27.363797"
    result = _row_created_at_to_epoch(sqlite_value)
    assert result > 0
    assert isinstance(result, float)


def test_load_all_for_node_source_uses_helper_not_direct_timestamp():
    """Pin: the source must use _row_created_at_to_epoch(),
    not call .timestamp() directly. A refactor that reverts
    to row.created_at.timestamp() would silently break SQLite
    deployments again."""
    from pathlib import Path
    db_src = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "core" / "database.py"
    ).read_text()
    # Find the load_all_for_node body.
    idx = db_src.find("async def load_all_for_node")
    assert idx >= 0
    # Next 5000 chars cover the function body.
    body = db_src[idx:idx + 5000]
    # The helper MUST be the integration point.
    assert "_row_created_at_to_epoch" in body, (
        "load_all_for_node must use _row_created_at_to_epoch "
        "helper for created_at — pre-fix direct .timestamp() "
        "call crashed on SQLite str values"
    )
    # The buggy pattern must NOT reappear.
    assert "row.created_at.timestamp()" not in body, (
        "F22 regression: row.created_at.timestamp() was "
        "re-introduced — SQLite strings will crash hydration"
    )
