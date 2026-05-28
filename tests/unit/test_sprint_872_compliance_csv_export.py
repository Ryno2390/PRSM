"""Sprint 872 — compliance audit ring CSV export pin tests."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from prsm.economy.web3.compliance_csv_export import (
    COMPLIANCE_CSV_HEADER,
    _entry_passes_filter,
    _entry_to_csv_row,
    _iso8601,
    export_summary,
    export_to_csv,
)


@dataclass
class _FakeEntry:
    entry_id: str = "e1"
    timestamp: float = 1700000000.0
    kind: str = "onramp_quote"
    user_id: str = "alice"
    usd_amount: float = 100.0
    ftns_amount: float = 0.0
    status: str = "OK"
    kyc_status: Optional[str] = "VERIFIED"
    tx_hash: Optional[str] = None
    vendor_ref: Optional[str] = None
    address: Optional[str] = None
    jurisdiction: Optional[str] = "AU"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Canonical schema ─────────────────────────────────────────

def test_csv_header_pinned():
    """Schema-breaking changes here cascade to downstream
    regulator-format transforms — pin the column order."""
    assert COMPLIANCE_CSV_HEADER == [
        "entry_id",
        "timestamp_unix",
        "timestamp_iso8601_utc",
        "kind",
        "user_id",
        "usd_amount",
        "ftns_amount",
        "status",
        "kyc_status",
        "tx_hash",
        "vendor_ref",
        "address",
        "jurisdiction",
        "metadata_json",
    ]


# ── ISO 8601 timestamp ───────────────────────────────────────

def test_iso8601_zero_returns_empty():
    assert _iso8601(0) == ""
    assert _iso8601(-1) == ""


def test_iso8601_real_timestamp_round_trip():
    """2026-05-28 00:00 UTC → real ISO string."""
    # 2026-05-28T00:00:00 UTC = 1779667200 (approx)
    ts = 1779667200.0
    iso = _iso8601(ts)
    assert "2026-05" in iso
    assert iso.endswith("+00:00")


# ── Row encoding ─────────────────────────────────────────────

def test_row_canonical_shape():
    e = _FakeEntry()
    row = _entry_to_csv_row(e)
    assert len(row) == len(COMPLIANCE_CSV_HEADER)
    assert row[0] == "e1"  # entry_id
    assert row[3] == "onramp_quote"  # kind
    assert row[4] == "alice"  # user_id


def test_row_usd_formatted_with_6_decimals():
    e = _FakeEntry(usd_amount=99.99)
    row = _entry_to_csv_row(e)
    # column 5 = usd_amount
    assert row[5] == "99.990000"


def test_row_none_fields_become_empty_strings():
    """None values must serialize as empty cells, NOT 'None'
    literal — downstream regulator parsers fail on 'None'."""
    e = _FakeEntry(
        kyc_status=None, tx_hash=None, vendor_ref=None,
        address=None, jurisdiction=None,
    )
    row = _entry_to_csv_row(e)
    # columns 8-12 = optional fields
    for i in range(8, 13):
        assert row[i] == ""


def test_row_metadata_json_encoded_with_stable_key_order():
    e = _FakeEntry(metadata={"z": 1, "a": 2, "m": 3})
    row = _entry_to_csv_row(e)
    # Last column = metadata_json
    assert row[-1] == '{"a":2,"m":3,"z":1}'


def test_row_empty_metadata_is_empty_string():
    e = _FakeEntry(metadata={})
    row = _entry_to_csv_row(e)
    assert row[-1] == ""


def test_row_zero_amounts_serialize_as_0_not_empty():
    """A real $0 entry (rare but valid) must serialize as '0',
    not empty — distinguishes 'recorded but zeroed' from
    'missing data'."""
    e = _FakeEntry(usd_amount=0, ftns_amount=0)
    row = _entry_to_csv_row(e)
    assert row[5] == "0"
    assert row[6] == "0"


# ── Filter logic ─────────────────────────────────────────────

def test_filter_since_excludes_old_entries():
    e_old = _FakeEntry(timestamp=100)
    e_new = _FakeEntry(timestamp=200)
    assert _entry_passes_filter(e_old, since=150) is False
    assert _entry_passes_filter(e_new, since=150) is True


def test_filter_until_excludes_new_entries():
    """`until` is exclusive — entries with ts >= until skipped."""
    e_before = _FakeEntry(timestamp=100)
    e_at = _FakeEntry(timestamp=200)
    e_after = _FakeEntry(timestamp=300)
    assert _entry_passes_filter(e_before, until=200) is True
    assert _entry_passes_filter(e_at, until=200) is False
    assert _entry_passes_filter(e_after, until=200) is False


def test_filter_user_id_exact_match():
    e_alice = _FakeEntry(user_id="alice")
    e_bob = _FakeEntry(user_id="bob")
    assert _entry_passes_filter(e_alice, user_id="alice") is True
    assert _entry_passes_filter(e_bob, user_id="alice") is False


def test_filter_kinds_set_membership():
    e_quote = _FakeEntry(kind="onramp_quote")
    e_exec = _FakeEntry(kind="onramp_execute")
    e_off = _FakeEntry(kind="offramp_quote")
    targets = {"onramp_quote", "offramp_quote"}
    assert _entry_passes_filter(e_quote, kinds=targets) is True
    assert _entry_passes_filter(e_off, kinds=targets) is True
    assert _entry_passes_filter(e_exec, kinds=targets) is False


def test_filter_min_usd_fincen_ctr_threshold():
    """The FinCEN CTR threshold ($10k) is the canonical use case
    for min_usd. Below threshold = excluded from CTR report."""
    e_small = _FakeEntry(usd_amount=500)
    e_big = _FakeEntry(usd_amount=10_001)
    assert _entry_passes_filter(e_small, min_usd=10_000) is False
    assert _entry_passes_filter(e_big, min_usd=10_000) is True


def test_filter_no_filters_passes_all():
    assert _entry_passes_filter(_FakeEntry()) is True


# ── Full CSV serialization ───────────────────────────────────

def _read_csv(csv_text: str):
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    return rows[0], rows[1:]


def test_export_includes_header_row_first():
    entries = [_FakeEntry()]
    csv_text = export_to_csv(entries)
    header, body = _read_csv(csv_text)
    assert header == COMPLIANCE_CSV_HEADER
    assert len(body) == 1


def test_export_empty_ring_produces_header_only():
    csv_text = export_to_csv([])
    header, body = _read_csv(csv_text)
    assert header == COMPLIANCE_CSV_HEADER
    assert body == []


def test_export_applies_since_filter():
    entries = [
        _FakeEntry(entry_id="old", timestamp=100),
        _FakeEntry(entry_id="new", timestamp=200),
    ]
    csv_text = export_to_csv(entries, since=150)
    header, body = _read_csv(csv_text)
    assert len(body) == 1
    assert body[0][0] == "new"


def test_export_applies_user_id_filter():
    entries = [
        _FakeEntry(entry_id="a", user_id="alice"),
        _FakeEntry(entry_id="b", user_id="bob"),
    ]
    csv_text = export_to_csv(entries, user_id="alice")
    header, body = _read_csv(csv_text)
    assert len(body) == 1
    assert body[0][0] == "a"


def test_export_applies_kinds_filter():
    entries = [
        _FakeEntry(entry_id="q", kind="onramp_quote"),
        _FakeEntry(entry_id="e", kind="onramp_execute"),
    ]
    csv_text = export_to_csv(entries, kinds=["onramp_quote"])
    header, body = _read_csv(csv_text)
    assert len(body) == 1
    assert body[0][0] == "q"


def test_export_rfc4180_lineterm():
    """RFC 4180 says CRLF; we use LF for cross-platform
    consistency — modern parsers handle both. Pinning so a future
    refactor doesn't switch back to CRLF unexpectedly + break
    downstream pipelines."""
    entries = [_FakeEntry()]
    csv_text = export_to_csv(entries)
    assert "\n" in csv_text
    # No CRLF in our output
    assert "\r" not in csv_text


# ── Summary aggregation ──────────────────────────────────────

def test_summary_aggregates_by_kind():
    entries = [
        _FakeEntry(kind="onramp_quote", usd_amount=100),
        _FakeEntry(kind="onramp_quote", usd_amount=200),
        _FakeEntry(kind="offramp_quote", usd_amount=50),
    ]
    s = export_summary(entries)
    assert s["total_count"] == 3
    assert s["total_usd"] == 350.0
    assert s["by_kind"]["onramp_quote"]["count"] == 2
    assert s["by_kind"]["onramp_quote"]["usd"] == 300.0
    assert s["by_kind"]["offramp_quote"]["count"] == 1


def test_summary_window_filter():
    entries = [
        _FakeEntry(timestamp=100, usd_amount=10),
        _FakeEntry(timestamp=200, usd_amount=20),
        _FakeEntry(timestamp=300, usd_amount=30),
    ]
    s = export_summary(entries, since=150, until=250)
    assert s["total_count"] == 1
    assert s["total_usd"] == 20
    assert s["filter_since"] == 150
    assert s["filter_until"] == 250


def test_summary_empty_ring():
    s = export_summary([])
    assert s["total_count"] == 0
    assert s["total_usd"] == 0
    assert s["by_kind"] == {}
