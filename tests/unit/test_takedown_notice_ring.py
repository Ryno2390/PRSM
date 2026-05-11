"""Sprint 272 — TakedownNoticeRing tests.

Foundation-side intake for content takedown notices. Information
distribution only — never enforces, never modifies operator
filters. Operators voluntarily act via their sprint-269
ContentFilterStore."""
from __future__ import annotations

import json

import pytest

from prsm.node.takedown_notice_log import (
    TakedownNoticeEntry, TakedownNoticeRing,
)


def test_entry_to_dict_round_trip():
    e = TakedownNoticeEntry(
        notice_id="abc-123",
        timestamp=100.0,
        target_cid="bafy-target",
        sender="legal@example.com",
        jurisdiction="US-DMCA",
        basis="DMCA §512(c)",
        notice_text="full notice body",
        status="received",
    )
    d = e.to_dict()
    assert d["notice_id"] == "abc-123"
    assert d["target_cid"] == "bafy-target"
    assert d["status"] == "received"


def test_record_returns_uuid_assigned_entry():
    r = TakedownNoticeRing()
    e = r.record(
        target_cid="bafy1",
        sender="legal@ex.com",
        jurisdiction="US-DMCA",
        basis="DMCA §512(c)",
        notice_text="notice body here",
    )
    assert e.notice_id  # auto-assigned
    assert len(e.notice_id) > 0
    assert e.target_cid == "bafy1"
    assert e.status == "received"
    assert r.count() == 1


def test_record_validates_required_fields():
    r = TakedownNoticeRing()
    with pytest.raises(ValueError):
        r.record(target_cid="", sender="x", jurisdiction="y", basis="z")
    with pytest.raises(ValueError):
        r.record(target_cid="x", sender="", jurisdiction="y", basis="z")
    with pytest.raises(ValueError):
        r.record(target_cid="x", sender="y", jurisdiction="", basis="z")
    with pytest.raises(ValueError):
        r.record(target_cid="x", sender="y", jurisdiction="z", basis="")


def test_record_caps_notice_text():
    r = TakedownNoticeRing()
    huge = "x" * 100_000
    e = r.record(
        target_cid="bafy1", sender="x", jurisdiction="y",
        basis="z", notice_text=huge,
    )
    assert len(e.notice_text) == 8192  # _MAX_NOTICE_TEXT_LEN


def test_get_by_id():
    r = TakedownNoticeRing()
    e = r.record(
        target_cid="bafy1", sender="x", jurisdiction="y", basis="z",
    )
    fetched = r.get(e.notice_id)
    assert fetched is not None
    assert fetched.target_cid == "bafy1"
    assert r.get("nonexistent") is None


def test_recent_returns_newest_first():
    r = TakedownNoticeRing()
    e1 = r.record(
        target_cid="c1", sender="s", jurisdiction="j", basis="b",
        timestamp=100.0,
    )
    e2 = r.record(
        target_cid="c2", sender="s", jurisdiction="j", basis="b",
        timestamp=200.0,
    )
    recent = r.recent(limit=10)
    assert [e.notice_id for e in recent] == [e2.notice_id, e1.notice_id]


def test_recent_status_filter():
    r = TakedownNoticeRing()
    r.record(
        target_cid="c1", sender="s", jurisdiction="j", basis="b",
    )
    received = r.recent(limit=10, status="received")
    assert len(received) == 1
    expired = r.recent(limit=10, status="expired")
    assert len(expired) == 0


def test_recent_target_cid_filter():
    r = TakedownNoticeRing()
    r.record(
        target_cid="target-a", sender="s", jurisdiction="j", basis="b",
    )
    r.record(
        target_cid="target-b", sender="s", jurisdiction="j", basis="b",
    )
    r.record(
        target_cid="target-a", sender="s", jurisdiction="j", basis="b",
    )
    matches = r.recent(limit=10, target_cid="target-a")
    assert len(matches) == 2
    assert all(e.target_cid == "target-a" for e in matches)


def test_recent_invalid_limit_rejected():
    r = TakedownNoticeRing()
    with pytest.raises(ValueError):
        r.recent(limit=0)
    with pytest.raises(ValueError):
        r.recent(limit=1001)


def test_recent_invalid_status_rejected():
    r = TakedownNoticeRing()
    with pytest.raises(ValueError):
        r.recent(limit=10, status="bogus")


def test_invalid_max_entries_rejected():
    with pytest.raises(ValueError):
        TakedownNoticeRing(max_entries=0)


def test_persistence_round_trip(tmp_path):
    r1 = TakedownNoticeRing(persist_dir=tmp_path)
    e = r1.record(
        target_cid="bafy-persisted",
        sender="legal@ex.com",
        jurisdiction="US-DMCA",
        basis="DMCA §512(c)",
        notice_text="full body",
        timestamp=100.0,
    )
    r2 = TakedownNoticeRing(persist_dir=tmp_path)
    assert r2.count() == 1
    fetched = r2.get(e.notice_id)
    assert fetched is not None
    assert fetched.target_cid == "bafy-persisted"


def test_persistence_corrupt_file_fail_soft(tmp_path):
    (tmp_path / "garbage.json").write_text("{not valid json")
    r = TakedownNoticeRing(persist_dir=tmp_path)
    assert r.count() == 0


def test_max_entries_enforced():
    r = TakedownNoticeRing(max_entries=2)
    for i in range(5):
        r.record(
            target_cid=f"c{i}", sender="s",
            jurisdiction="j", basis="b",
        )
    assert r.count() == 2
