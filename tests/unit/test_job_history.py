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
