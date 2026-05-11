"""Sprint 282 — FiatComplianceRing tests.

Compliance audit trail for fiat on/off-ramp events. Records
every quote + execute attempt across the Phase 5 surface so
operators have a single queryable log for AUSTRAC / FinCEN /
IRS reporting once ramps are live.

Persistence is required, not opt-in: regulators expect 5-7
year retention. The ring writes to disk by default when
PRSM_FIAT_COMPLIANCE_LOG_DIR is set; without the env var, the
ring operates in-memory only with bounded retention so a
misconfigured node doesn't silently drop audit data into the
void.
"""
from __future__ import annotations

import json
import pytest

from prsm.economy.web3.fiat_compliance_ring import (
    FiatComplianceEntry, FiatComplianceRing,
)


# ── Entry round-trip ─────────────────────────────────────


def test_entry_to_dict_round_trip():
    e = FiatComplianceEntry(
        entry_id="e-1", timestamp=100.0,
        kind="onramp_quote",
        user_id="alice",
        usd_amount=100.0, ftns_amount=100.0,
        status="PENDING_COMMISSION",
        kyc_status="VERIFIED",
        tx_hash=None,
        vendor_ref="persona-alice",
        jurisdiction="US",
        metadata={"k": "v"},
    )
    d = e.to_dict()
    restored = FiatComplianceEntry.from_dict(d)
    assert restored == e


# ── Record + validate ────────────────────────────────────


def test_record_assigns_entry_id_and_timestamp():
    r = FiatComplianceRing()
    e = r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0,
        status="PENDING_COMMISSION",
    )
    assert e.entry_id
    assert e.timestamp > 0
    assert e.kind == "onramp_quote"
    assert r.count() == 1


def test_record_validates_kind():
    r = FiatComplianceRing()
    with pytest.raises(ValueError):
        r.record(
            kind="bogus", user_id="alice",
            usd_amount=100.0, ftns_amount=100.0,
            status="OK",
        )


def test_record_kind_taxonomy():
    """The valid-kinds set is the union of fiat surfaces this
    sprint covers."""
    r = FiatComplianceRing()
    for kind in [
        "onramp_quote", "onramp_execute",
        "offramp_quote", "offramp_execute",
        "gasless_transfer_quote", "gasless_transfer_execute",
        "kyc_initiate", "kyc_status_change",
    ]:
        r.record(
            kind=kind, user_id="alice",
            usd_amount=0.0, ftns_amount=0.0, status="X",
        )
    assert r.count() == 8


def test_record_validates_non_negative_amounts():
    r = FiatComplianceRing()
    with pytest.raises(ValueError):
        r.record(
            kind="onramp_quote", user_id="alice",
            usd_amount=-1.0, ftns_amount=10.0,
            status="OK",
        )
    with pytest.raises(ValueError):
        r.record(
            kind="onramp_quote", user_id="alice",
            usd_amount=10.0, ftns_amount=-1.0,
            status="OK",
        )


def test_record_user_id_can_be_empty_for_explicit_address_flows():
    """Explicit destination_address (no WaaS user_id) is a
    valid path — operator should still record it."""
    r = FiatComplianceRing()
    e = r.record(
        kind="onramp_quote", user_id="",
        address="0xrecipient",
        usd_amount=100.0, ftns_amount=100.0,
        status="OK",
    )
    assert e.user_id == ""
    assert e.address == "0xrecipient"


# ── Query: recent ────────────────────────────────────────


def test_recent_newest_first():
    r = FiatComplianceRing()
    e1 = r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0,
        status="OK", timestamp=100.0,
    )
    e2 = r.record(
        kind="onramp_quote", user_id="bob",
        usd_amount=50.0, ftns_amount=50.0,
        status="OK", timestamp=200.0,
    )
    recent = r.recent(limit=10)
    assert [e.entry_id for e in recent] == [
        e2.entry_id, e1.entry_id,
    ]


def test_recent_kind_filter():
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    r.record(
        kind="offramp_quote", user_id="alice",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
    )
    onramps = r.recent(limit=10, kind="onramp_quote")
    assert len(onramps) == 1
    assert onramps[0].kind == "onramp_quote"


def test_recent_user_id_filter():
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    r.record(
        kind="onramp_quote", user_id="bob",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
    )
    alice_entries = r.recent(limit=10, user_id="alice")
    assert len(alice_entries) == 1
    assert alice_entries[0].user_id == "alice"


def test_recent_invalid_limit_rejected():
    r = FiatComplianceRing()
    with pytest.raises(ValueError):
        r.recent(limit=0)
    with pytest.raises(ValueError):
        r.recent(limit=10001)


# ── Summary ──────────────────────────────────────────────


def test_summary_by_kind():
    r = FiatComplianceRing()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    r.record(
        kind="onramp_quote", user_id="bob",
        usd_amount=200.0, ftns_amount=200.0, status="OK",
    )
    r.record(
        kind="offramp_quote", user_id="alice",
        usd_amount=50.0, ftns_amount=50.0, status="OK",
    )
    s = r.summary_by_kind()
    assert s["onramp_quote"]["count"] == 2
    assert s["onramp_quote"]["total_usd"] == 300.0
    assert s["offramp_quote"]["count"] == 1
    assert s["offramp_quote"]["total_usd"] == 50.0


def test_summary_empty_ring():
    r = FiatComplianceRing()
    assert r.summary_by_kind() == {}


def test_summary_zero_amount_events_counted():
    """KYC events have 0 USD amount but should still surface
    in counts for audit visibility."""
    r = FiatComplianceRing()
    r.record(
        kind="kyc_initiate", user_id="alice",
        usd_amount=0.0, ftns_amount=0.0, status="OK",
    )
    s = r.summary_by_kind()
    assert s["kyc_initiate"]["count"] == 1
    assert s["kyc_initiate"]["total_usd"] == 0.0


# ── Persistence ──────────────────────────────────────────


def test_persistence_round_trip(tmp_path):
    r1 = FiatComplianceRing(persist_dir=tmp_path)
    e = r1.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
        timestamp=100.0,
    )
    r2 = FiatComplianceRing(persist_dir=tmp_path)
    assert r2.count() == 1
    fetched = r2.get(e.entry_id)
    assert fetched is not None
    assert fetched.user_id == "alice"


def test_persistence_corrupt_file_fail_soft(tmp_path):
    (tmp_path / "garbage.json").write_text("{not valid json")
    r = FiatComplianceRing(persist_dir=tmp_path)
    assert r.count() == 0


def test_persistence_disk_load_ordered_by_timestamp(tmp_path):
    r1 = FiatComplianceRing(persist_dir=tmp_path)
    r1.record(
        kind="onramp_quote", user_id="b",
        usd_amount=0.0, ftns_amount=0.0, status="OK",
        timestamp=200.0,
    )
    r1.record(
        kind="onramp_quote", user_id="a",
        usd_amount=0.0, ftns_amount=0.0, status="OK",
        timestamp=100.0,
    )
    r2 = FiatComplianceRing(persist_dir=tmp_path)
    recent = r2.recent(limit=10)
    # newest-first
    assert recent[0].user_id == "b"
    assert recent[1].user_id == "a"


# ── from_env ─────────────────────────────────────────────


def test_from_env_no_dir(monkeypatch):
    monkeypatch.delenv(
        "PRSM_FIAT_COMPLIANCE_LOG_DIR", raising=False,
    )
    r = FiatComplianceRing.from_env()
    assert r._persist_dir is None
    assert r.count() == 0


def test_from_env_with_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "PRSM_FIAT_COMPLIANCE_LOG_DIR", str(tmp_path),
    )
    r = FiatComplianceRing.from_env()
    assert r._persist_dir == tmp_path


def test_from_env_with_jurisdiction(monkeypatch):
    """Operator can pin a default jurisdiction tag so every
    record is labeled at intake."""
    monkeypatch.setenv("PRSM_OPERATOR_JURISDICTION", "US")
    r = FiatComplianceRing.from_env()
    e = r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="OK",
    )
    assert e.jurisdiction == "US"


# ── Bounded retention ────────────────────────────────────


def test_max_entries_enforced():
    r = FiatComplianceRing(max_entries=2)
    for i in range(5):
        r.record(
            kind="onramp_quote", user_id=f"u{i}",
            usd_amount=0.0, ftns_amount=0.0, status="OK",
        )
    assert r.count() == 2


def test_default_max_entries_large():
    """Default retention should be large enough that operators
    are unlikely to lose data before persistence is set up."""
    r = FiatComplianceRing()
    assert r._max_entries >= 100_000
