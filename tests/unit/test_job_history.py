"""B8 async-dispatch follow-on — JobHistoryRecord + JobHistoryStore.

Closes the ``/compute/status/{job_id}`` honest-scope gap: status
endpoint previously returned escrow lifecycle only. With JobHistory
wired, callers see route + response + aggregator + participants
+ start/complete timing — the load-bearing surface for late-retrieval
of long-running jobs once async dispatch lands.

v1 is in-memory LRU-bounded: simpler, fast, sufficient for the
synchronous-from-caller path that's currently the only path live.
Filesystem persistence is the right v2.
"""
from __future__ import annotations

import time

import pytest

from prsm.node.job_history import (
    JobHistoryRecord,
    JobHistoryStore,
    JobStatus,
)


# ──────────────────────────────────────────────────────────────────────
# JobHistoryRecord — schema + validation
# ──────────────────────────────────────────────────────────────────────


class TestJobHistoryRecord:
    def test_minimal_construction(self):
        rec = JobHistoryRecord(
            job_id="forge-abc",
            query="Count records",
            status=JobStatus.IN_PROGRESS,
            started_at=1_700_000_000.0,
        )
        assert rec.job_id == "forge-abc"
        assert rec.status == JobStatus.IN_PROGRESS
        assert rec.completed_at is None
        assert rec.route is None
        assert rec.response is None
        assert rec.error is None

    def test_completed_record_carries_route_and_response(self):
        rec = JobHistoryRecord(
            job_id="forge-abc",
            query="q",
            status=JobStatus.COMPLETED,
            started_at=1_700_000_000.0,
            completed_at=1_700_000_010.0,
            route="qo_swarm",
            response='{"count": 7}',
            aggregator_node_id="agg-7",
            contributing_shards=("cid-a", "cid-b"),
            participants=({"shard_cid": "cid-a"}, {"shard_cid": "cid-b"}),
            traces_collected=2,
        )
        assert rec.route == "qo_swarm"
        assert rec.response == '{"count": 7}'
        assert rec.aggregator_node_id == "agg-7"
        assert rec.contributing_shards == ("cid-a", "cid-b")
        assert len(rec.participants) == 2
        assert rec.traces_collected == 2

    def test_failed_record_carries_error(self):
        rec = JobHistoryRecord(
            job_id="forge-fail",
            query="q",
            status=JobStatus.FAILED,
            started_at=1_700_000_000.0,
            completed_at=1_700_000_005.0,
            error="aggregator timeout",
        )
        assert rec.status == JobStatus.FAILED
        assert rec.error == "aggregator timeout"

    def test_empty_job_id_rejected(self):
        with pytest.raises(ValueError, match="job_id"):
            JobHistoryRecord(
                job_id="",
                query="q",
                status=JobStatus.IN_PROGRESS,
                started_at=1.0,
            )

    def test_negative_started_at_rejected(self):
        with pytest.raises(ValueError, match="started_at"):
            JobHistoryRecord(
                job_id="x",
                query="q",
                status=JobStatus.IN_PROGRESS,
                started_at=-1.0,
            )

    def test_completed_at_before_started_at_rejected(self):
        with pytest.raises(ValueError, match="completed_at"):
            JobHistoryRecord(
                job_id="x",
                query="q",
                status=JobStatus.COMPLETED,
                started_at=100.0,
                completed_at=50.0,  # before start
            )

    def test_to_dict_round_trip(self):
        rec = JobHistoryRecord(
            job_id="forge-abc",
            query="q",
            status=JobStatus.COMPLETED,
            started_at=1.0,
            completed_at=2.0,
            route="qo_swarm",
            response="ok",
        )
        d = rec.to_dict()
        # All load-bearing fields surface in the dict.
        assert d["job_id"] == "forge-abc"
        assert d["status"] == "completed"
        assert d["started_at"] == 1.0
        assert d["completed_at"] == 2.0
        assert d["route"] == "qo_swarm"
        assert d["response"] == "ok"


# ──────────────────────────────────────────────────────────────────────
# JobHistoryStore
# ──────────────────────────────────────────────────────────────────────


class TestJobHistoryStore:
    def test_record_and_retrieve(self):
        store = JobHistoryStore(max_entries=100)
        rec = JobHistoryRecord(
            job_id="forge-1", query="q", status=JobStatus.IN_PROGRESS,
            started_at=time.time(),
        )
        store.put(rec)
        retrieved = store.get("forge-1")
        assert retrieved == rec

    def test_unknown_job_returns_none(self):
        store = JobHistoryStore(max_entries=10)
        assert store.get("nonexistent") is None

    def test_put_overwrites_existing(self):
        # Job lifecycle: in_progress → completed. The store must
        # accept the in-place update so /compute/status surfaces
        # the latest state.
        store = JobHistoryStore(max_entries=10)
        t = time.time()
        store.put(JobHistoryRecord(
            job_id="j", query="q", status=JobStatus.IN_PROGRESS,
            started_at=t,
        ))
        store.put(JobHistoryRecord(
            job_id="j", query="q", status=JobStatus.COMPLETED,
            started_at=t, completed_at=t + 1, route="qo_swarm",
            response="ok",
        ))
        rec = store.get("j")
        assert rec.status == JobStatus.COMPLETED
        assert rec.response == "ok"

    def test_lru_eviction_when_over_max_entries(self):
        # Bounded memory: oldest entries evicted past max_entries.
        store = JobHistoryStore(max_entries=3)
        for i in range(5):
            store.put(JobHistoryRecord(
                job_id=f"j-{i}", query="q",
                status=JobStatus.IN_PROGRESS,
                started_at=float(i),
            ))
        # j-0 + j-1 evicted; j-2..j-4 retained.
        assert store.get("j-0") is None
        assert store.get("j-1") is None
        assert store.get("j-2") is not None
        assert store.get("j-3") is not None
        assert store.get("j-4") is not None

    def test_get_promotes_to_most_recent_in_lru(self):
        # An accessed entry should NOT be the next eviction
        # candidate. Standard LRU semantics.
        store = JobHistoryStore(max_entries=3)
        for i in range(3):
            store.put(JobHistoryRecord(
                job_id=f"j-{i}", query="q",
                status=JobStatus.IN_PROGRESS,
                started_at=float(i),
            ))
        # Access j-0 (oldest) — promotes it.
        store.get("j-0")
        # Insert a new entry: should evict j-1 (now oldest after
        # j-0 was promoted).
        store.put(JobHistoryRecord(
            job_id="j-3", query="q",
            status=JobStatus.IN_PROGRESS, started_at=3.0,
        ))
        assert store.get("j-1") is None
        assert store.get("j-0") is not None

    def test_max_entries_rejected_at_construction_when_zero(self):
        with pytest.raises(ValueError, match="max_entries"):
            JobHistoryStore(max_entries=0)

    def test_max_entries_rejected_at_construction_when_negative(self):
        with pytest.raises(ValueError, match="max_entries"):
            JobHistoryStore(max_entries=-1)

    def test_size_reports_current_entry_count(self):
        store = JobHistoryStore(max_entries=10)
        assert store.size() == 0
        store.put(JobHistoryRecord(
            job_id="j", query="q", status=JobStatus.IN_PROGRESS,
            started_at=time.time(),
        ))
        assert store.size() == 1

    def test_default_max_entries_is_reasonable(self):
        # Default should accommodate a few hundred jobs without
        # operator tuning; pin a load-bearing minimum.
        store = JobHistoryStore()
        assert store._max_entries >= 256


# ──────────────────────────────────────────────────────────────────────
# Filesystem persistence (v2, ships 2026-05-09)
# ──────────────────────────────────────────────────────────────────────


def _record(job_id: str, *, started_at: float = None,
            status: JobStatus = JobStatus.IN_PROGRESS,
            response: str = None) -> JobHistoryRecord:
    """Helper for terse test setup."""
    return JobHistoryRecord(
        job_id=job_id,
        query=f"q-{job_id}",
        status=status,
        started_at=started_at if started_at is not None else time.time(),
        response=response,
    )


class TestJobHistoryStorePersistence:
    """v2 closes the B8 deferred sub-item: filesystem persistence
    so node restart doesn't wipe job history. Pattern: put writes
    through to disk; get falls back to disk on in-memory miss;
    startup scan repopulates the in-memory LRU from disk.
    persist_dir=None preserves v1 in-memory-only behavior.
    """

    def test_v1_behavior_when_persist_dir_none(self, tmp_path):
        """Without persist_dir, store is in-memory only — no disk
        writes, no disk reads."""
        store = JobHistoryStore()
        store.put(_record("forge-1"))
        # No files anywhere; behavior identical to v1.
        assert list(tmp_path.iterdir()) == []
        assert store.get("forge-1") is not None

    def test_put_writes_to_disk_when_persist_dir_set(self, tmp_path):
        store = JobHistoryStore(persist_dir=tmp_path)
        rec = _record("forge-1", response="42")
        store.put(rec)
        # File written under persist_dir.
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        # Filename uses job_id (sanitized); contents round-trip.
        import json
        loaded = json.loads(files[0].read_text())
        assert loaded["job_id"] == "forge-1"
        assert loaded["response"] == "42"

    def test_put_overwrites_existing_disk_record(self, tmp_path):
        store = JobHistoryStore(persist_dir=tmp_path)
        store.put(_record("forge-1", status=JobStatus.IN_PROGRESS))
        store.put(_record(
            "forge-1", status=JobStatus.COMPLETED, response="done",
        ))
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        import json
        loaded = json.loads(files[0].read_text())
        assert loaded["status"] == "completed"
        assert loaded["response"] == "done"

    def test_get_falls_back_to_disk_on_memory_miss(self, tmp_path):
        """Job evicted from in-memory LRU still retrievable from
        disk."""
        # Tiny store to force eviction.
        store = JobHistoryStore(max_entries=2, persist_dir=tmp_path)
        store.put(_record("forge-1"))
        store.put(_record("forge-2"))
        store.put(_record("forge-3"))  # evicts forge-1 from memory
        # In-memory size respects LRU bound.
        assert store.size() == 2
        # forge-1 evicted from memory but still on disk → get
        # finds it via disk lookup.
        rec = store.get("forge-1")
        assert rec is not None
        assert rec.job_id == "forge-1"

    def test_startup_scan_loads_existing_disk_records(self, tmp_path):
        """Constructor scans persist_dir on init and populates
        in-memory LRU from disk so /compute/status finds prior-run
        jobs immediately on node restart."""
        # Pre-seed disk with two records (simulates prior node run).
        store_a = JobHistoryStore(persist_dir=tmp_path)
        store_a.put(_record("forge-1", started_at=100.0))
        store_a.put(_record(
            "forge-2", started_at=200.0,
            status=JobStatus.COMPLETED, response="r",
        ))
        # Fresh store on the same dir — should pick up the records.
        store_b = JobHistoryStore(persist_dir=tmp_path)
        assert store_b.get("forge-1") is not None
        assert store_b.get("forge-2") is not None
        assert store_b.get("forge-2").response == "r"

    def test_corrupt_disk_file_skipped_fail_soft(self, tmp_path, caplog):
        """A corrupt JSON file under persist_dir must NOT crash
        startup. Log + skip + continue with valid records."""
        # Write a junk file directly to the persist_dir.
        (tmp_path / "forge-bad.json").write_text("{not valid json")
        # Also write a valid one to verify scan continues.
        store_a = JobHistoryStore(persist_dir=tmp_path)
        store_a.put(_record("forge-good"))

        # Reload — must not raise.
        store_b = JobHistoryStore(persist_dir=tmp_path)
        # Valid record loaded.
        assert store_b.get("forge-good") is not None
        # Bad record absent (skipped, not crashed on).
        assert store_b.get("forge-bad") is None

    def test_persist_dir_created_if_missing(self, tmp_path):
        """If persist_dir doesn't exist yet, the store creates it
        — no operator pre-mkdir required."""
        target = tmp_path / "subdir" / "jobs"
        assert not target.exists()
        store = JobHistoryStore(persist_dir=target)
        assert target.exists() and target.is_dir()
        store.put(_record("forge-1"))
        assert list(target.glob("*.json"))

    def test_persist_with_special_chars_in_job_id(self, tmp_path):
        """job_id with characters that aren't filesystem-safe
        (e.g., slashes from a foreign caller) should still persist
        cleanly via filename sanitization."""
        store = JobHistoryStore(persist_dir=tmp_path)
        # Path-traversal-shaped job_id should NOT escape the
        # persist_dir — it's a security boundary.
        store.put(_record("../../etc/passwd"))
        # Round-trip works regardless.
        rec = store.get("../../etc/passwd")
        assert rec is not None
        # No file written outside persist_dir.
        for child in tmp_path.parent.iterdir():
            if child.is_file() and "passwd" in child.name:
                pytest.fail(
                    f"Filename traversal escaped persist_dir: {child}"
                )


# ──────────────────────────────────────────────────────────────────────
# Enumeration / list surface (v2 — for /compute/jobs)
# ──────────────────────────────────────────────────────────────────────


class TestJobHistoryStoreList:
    """v2 enumeration surface: list() returns records most-recent-
    first with optional status filter + pagination."""

    def test_list_empty_returns_empty_list(self):
        store = JobHistoryStore()
        result = store.list()
        assert result == []

    def test_list_returns_most_recent_first(self):
        store = JobHistoryStore()
        # put in chronological order
        store.put(_record("forge-1", started_at=100.0))
        store.put(_record("forge-2", started_at=200.0))
        store.put(_record("forge-3", started_at=300.0))
        records = store.list()
        assert [r.job_id for r in records] == [
            "forge-3", "forge-2", "forge-1",
        ]

    def test_list_filter_by_status(self):
        store = JobHistoryStore()
        store.put(_record("forge-1", status=JobStatus.IN_PROGRESS))
        store.put(_record("forge-2", status=JobStatus.COMPLETED))
        store.put(_record("forge-3", status=JobStatus.FAILED))
        store.put(_record("forge-4", status=JobStatus.IN_PROGRESS))
        in_progress = store.list(status_filter=JobStatus.IN_PROGRESS)
        assert {r.job_id for r in in_progress} == {"forge-1", "forge-4"}
        completed = store.list(status_filter=JobStatus.COMPLETED)
        assert [r.job_id for r in completed] == ["forge-2"]

    def test_list_limit_caps_output_size(self):
        store = JobHistoryStore()
        for i in range(10):
            store.put(_record(f"forge-{i}", started_at=float(i)))
        result = store.list(limit=3)
        assert len(result) == 3
        # Most-recent-first: forge-9, forge-8, forge-7.
        assert [r.job_id for r in result] == ["forge-9", "forge-8", "forge-7"]

    def test_list_offset_skips_first_n(self):
        store = JobHistoryStore()
        for i in range(5):
            store.put(_record(f"forge-{i}", started_at=float(i)))
        # Skip first 2 → returns forge-2, forge-1, forge-0.
        result = store.list(offset=2)
        assert [r.job_id for r in result] == ["forge-2", "forge-1", "forge-0"]

    def test_list_offset_and_limit_compose(self):
        store = JobHistoryStore()
        for i in range(10):
            store.put(_record(f"forge-{i}", started_at=float(i)))
        # offset=3, limit=2 → forge-6, forge-5.
        result = store.list(offset=3, limit=2)
        assert [r.job_id for r in result] == ["forge-6", "forge-5"]

    def test_list_filter_combined_with_pagination(self):
        store = JobHistoryStore()
        for i in range(10):
            status = (
                JobStatus.COMPLETED if i % 2 == 0
                else JobStatus.IN_PROGRESS
            )
            store.put(_record(
                f"forge-{i}", status=status, started_at=float(i),
            ))
        # 5 IN_PROGRESS records (1, 3, 5, 7, 9). offset=1, limit=2.
        result = store.list(
            status_filter=JobStatus.IN_PROGRESS, offset=1, limit=2,
        )
        assert [r.job_id for r in result] == ["forge-7", "forge-5"]

    def test_count_returns_total_matching_filter(self):
        """count() returns the total number of records matching
        the filter (for pagination total)."""
        store = JobHistoryStore()
        for i in range(5):
            store.put(_record(
                f"forge-{i}",
                status=(
                    JobStatus.COMPLETED if i < 3 else JobStatus.IN_PROGRESS
                ),
            ))
        assert store.count() == 5
        assert store.count(status_filter=JobStatus.COMPLETED) == 3
        assert store.count(status_filter=JobStatus.IN_PROGRESS) == 2
        assert store.count(status_filter=JobStatus.FAILED) == 0
