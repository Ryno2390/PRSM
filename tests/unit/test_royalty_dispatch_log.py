"""Sprint 249 — RoyaltyDispatchRing audit ring.

Captures every dispatch attempt fired by the sprint-248
on-chain content-royalty activation block so operators can see
per-job per-shard outcomes via GET /admin/royalty-dispatch-history.

Symmetric to HeartbeatRecordedRing + SlashEventRing in design:
in-memory deque, optional filesystem persistence via env
(PRSM_ROYALTY_DISPATCH_LOG_DIR), recent() pagination.
"""
from __future__ import annotations

import json

import pytest

from prsm.node.royalty_dispatch_log import (
    RoyaltyDispatchEntry,
    RoyaltyDispatchRing,
)


class TestEntry:
    def test_to_dict_round_trip(self):
        e = RoyaltyDispatchEntry(
            timestamp=1715000000.0,
            job_id="job-1",
            cid="cid-a",
            status="sent",
            tx_hash="0xtx1",
            gross_wei=1_000_000_000_000_000,
            error=None,
        )
        d = e.to_dict()
        assert d["job_id"] == "job-1"
        assert d["status"] == "sent"
        assert d["gross_wei"] == 1_000_000_000_000_000
        assert d["tx_hash"] == "0xtx1"


class TestAppend:
    def test_basic_append(self):
        ring = RoyaltyDispatchRing()
        ring.append(
            job_id="job-1", cid="cid-a", status="sent",
            tx_hash="0xtx", gross_wei=100,
        )
        assert ring.count() == 1

    def test_max_entries_enforced(self):
        ring = RoyaltyDispatchRing(max_entries=2)
        for i in range(5):
            ring.append(
                job_id=f"job-{i}", cid=f"cid-{i}",
                status="sent", tx_hash=None, gross_wei=1,
            )
        assert ring.count() == 2

    def test_invalid_max_entries_rejected(self):
        with pytest.raises(ValueError):
            RoyaltyDispatchRing(max_entries=0)
        with pytest.raises(ValueError):
            RoyaltyDispatchRing(max_entries=-3)


class TestRecent:
    def test_returns_most_recent_first(self):
        ring = RoyaltyDispatchRing()
        ring.append(
            job_id="j1", cid="c1", status="sent",
            tx_hash="0xtx1", gross_wei=1, timestamp=100.0,
        )
        ring.append(
            job_id="j2", cid="c2", status="sent",
            tx_hash="0xtx2", gross_wei=1, timestamp=200.0,
        )
        snap = ring.recent(limit=10)
        assert [e.job_id for e in snap] == ["j2", "j1"]

    def test_status_filter(self):
        ring = RoyaltyDispatchRing()
        ring.append(
            job_id="j1", cid="c1", status="sent",
            tx_hash="0xtx", gross_wei=1,
        )
        ring.append(
            job_id="j2", cid="c2", status="failed",
            tx_hash=None, gross_wei=1, error="rpc",
        )
        ring.append(
            job_id="j3", cid="c3", status="skipped_no_record",
            tx_hash=None, gross_wei=1,
        )
        only_failed = ring.recent(limit=10, status="failed")
        assert [e.job_id for e in only_failed] == ["j2"]

    def test_job_id_filter(self):
        ring = RoyaltyDispatchRing()
        ring.append(
            job_id="job-A", cid="c1", status="sent",
            tx_hash="0x1", gross_wei=1,
        )
        ring.append(
            job_id="job-B", cid="c2", status="sent",
            tx_hash="0x2", gross_wei=1,
        )
        ring.append(
            job_id="job-A", cid="c3", status="sent",
            tx_hash="0x3", gross_wei=1,
        )
        scoped = ring.recent(limit=10, job_id="job-A")
        assert {e.cid for e in scoped} == {"c1", "c3"}

    def test_pagination(self):
        ring = RoyaltyDispatchRing()
        for i in range(5):
            ring.append(
                job_id=f"j{i}", cid=f"c{i}",
                status="sent", tx_hash=f"0x{i}",
                gross_wei=1, timestamp=float(i),
            )
        page2 = ring.recent(limit=2, offset=2)
        # newest-first: 4, 3, 2, 1, 0 → offset=2 limit=2 → [2, 1]
        assert [e.job_id for e in page2] == ["j2", "j1"]

    def test_invalid_limit_rejected(self):
        ring = RoyaltyDispatchRing()
        with pytest.raises(ValueError):
            ring.recent(limit=0)
        with pytest.raises(ValueError):
            ring.recent(limit=1001)

    def test_invalid_offset_rejected(self):
        ring = RoyaltyDispatchRing()
        with pytest.raises(ValueError):
            ring.recent(limit=10, offset=-1)


class TestFilesystemPersistence:
    def test_persisted_entries_load_on_restart(self, tmp_path):
        r1 = RoyaltyDispatchRing(persist_dir=tmp_path)
        r1.append(
            job_id="j1", cid="c1", status="sent",
            tx_hash="0xtx", gross_wei=42, timestamp=100.0,
        )
        # Fresh instance loads from disk.
        r2 = RoyaltyDispatchRing(persist_dir=tmp_path)
        assert r2.count() == 1
        assert r2.recent(limit=10)[0].job_id == "j1"

    def test_corrupt_file_skipped_no_crash(self, tmp_path):
        (tmp_path / "garbage.json").write_text("{not json")
        ring = RoyaltyDispatchRing(persist_dir=tmp_path)
        # Constructor must not raise; ring is just empty.
        assert ring.count() == 0
