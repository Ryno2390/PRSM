"""Phase 3.x.11 Task 2 — unit tests for ``KVCacheManager``.

Covers:
  - allocate / get / evict / evict_idle lifecycle
  - LRU eviction at ``max_cached_requests`` cap
  - TTL eviction past ``ttl_seconds``
  - Concurrent-allocation thread safety
  - Constructor validation
"""

from __future__ import annotations

import threading
import time
from typing import List

import pytest

from prsm.compute.chain_rpc.kv_cache import (
    CacheAlreadyAllocatedError,
    KVCacheHandle,
    KVCacheManager,
)


# ──────────────────────────────────────────────────────────────────────────
# Mock clock for deterministic TTL testing
# ──────────────────────────────────────────────────────────────────────────


class _ManualClock:
    """Deterministic monotonic clock; ``.now`` is read by
    ``KVCacheManager`` via the ``clock`` callable kwarg.
    Tests advance time by setting ``.now`` directly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


# ──────────────────────────────────────────────────────────────────────────
# Constructor validation
# ──────────────────────────────────────────────────────────────────────────


class TestConstructorValidation:
    def test_rejects_non_int_max(self):
        with pytest.raises(RuntimeError, match="max_cached_requests"):
            KVCacheManager(max_cached_requests="64")  # type: ignore[arg-type]

    def test_rejects_zero_max(self):
        with pytest.raises(RuntimeError, match="positive"):
            KVCacheManager(max_cached_requests=0)

    def test_rejects_bool_max(self):
        with pytest.raises(RuntimeError, match="max_cached_requests"):
            KVCacheManager(max_cached_requests=True)  # type: ignore[arg-type]

    def test_rejects_zero_ttl(self):
        with pytest.raises(RuntimeError, match="positive"):
            KVCacheManager(ttl_seconds=0)

    def test_rejects_negative_ttl(self):
        with pytest.raises(RuntimeError, match="positive"):
            KVCacheManager(ttl_seconds=-1)

    def test_rejects_bool_ttl(self):
        with pytest.raises(RuntimeError, match="ttl_seconds"):
            KVCacheManager(ttl_seconds=True)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────
# Allocate
# ──────────────────────────────────────────────────────────────────────────


class TestAllocate:
    def test_allocate_creates_new_handle(self):
        clock = _ManualClock()
        mgr = KVCacheManager(clock=clock)
        handle = mgr.allocate("req-1", n_layers=12)
        assert isinstance(handle, KVCacheHandle)
        assert handle.request_id == "req-1"
        assert handle.n_layers == 12
        assert handle.last_touch_time == 1000.0
        assert handle.payload is None  # runner sets later

    def test_allocate_twice_same_id_raises(self):
        mgr = KVCacheManager()
        mgr.allocate("req-1", n_layers=12)
        with pytest.raises(CacheAlreadyAllocatedError, match="req-1"):
            mgr.allocate("req-1", n_layers=12)

    def test_allocate_rejects_empty_request_id(self):
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="request_id"):
            mgr.allocate("", n_layers=12)

    def test_allocate_rejects_zero_layers(self):
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="n_layers"):
            mgr.allocate("req-1", n_layers=0)

    def test_allocate_rejects_bool_layers(self):
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="n_layers"):
            mgr.allocate("req-1", n_layers=True)  # type: ignore[arg-type]

    def test_allocate_writes_handle_to_pool(self):
        mgr = KVCacheManager()
        assert "req-1" not in mgr
        mgr.allocate("req-1", n_layers=12)
        assert "req-1" in mgr
        assert len(mgr) == 1


# ──────────────────────────────────────────────────────────────────────────
# Get
# ──────────────────────────────────────────────────────────────────────────


class TestGet:
    def test_get_returns_existing(self):
        mgr = KVCacheManager()
        allocated = mgr.allocate("req-1", n_layers=12)
        retrieved = mgr.get("req-1")
        assert retrieved is allocated

    def test_get_unknown_returns_none(self):
        mgr = KVCacheManager()
        assert mgr.get("nonexistent") is None

    def test_get_touches_last_touch_time(self):
        clock = _ManualClock(start=1000.0)
        mgr = KVCacheManager(clock=clock)
        handle = mgr.allocate("req-1", n_layers=12)
        assert handle.last_touch_time == 1000.0
        # Advance time and re-touch.
        clock.now = 1050.0
        retrieved = mgr.get("req-1")
        assert retrieved is not None
        assert retrieved.last_touch_time == 1050.0

    def test_get_moves_to_lru_end(self):
        # Allocate three; ``get`` on the oldest should move it
        # to the LRU end (so a subsequent allocate-at-cap
        # evicts the second-oldest, not the first-oldest).
        mgr = KVCacheManager(max_cached_requests=3)
        mgr.allocate("a", n_layers=12)
        mgr.allocate("b", n_layers=12)
        mgr.allocate("c", n_layers=12)
        # Touch "a" — moves to the LRU end.
        mgr.get("a")
        # Now allocate a fourth; LRU is "b" (not "a"), so "b"
        # gets evicted.
        mgr.allocate("d", n_layers=12)
        assert "a" in mgr
        assert "b" not in mgr  # LRU evicted
        assert "c" in mgr
        assert "d" in mgr


# ──────────────────────────────────────────────────────────────────────────
# Evict
# ──────────────────────────────────────────────────────────────────────────


class TestEvict:
    def test_evict_returns_true_when_handle_exists(self):
        mgr = KVCacheManager()
        mgr.allocate("req-1", n_layers=12)
        assert mgr.evict("req-1") is True
        assert "req-1" not in mgr

    def test_evict_returns_false_when_handle_absent(self):
        mgr = KVCacheManager()
        # Idempotent: evict-on-unknown is a no-op (matches the
        # GeneratorExit cleanup path racing with TTL eviction).
        assert mgr.evict("never-allocated") is False

    def test_evict_double_call_idempotent(self):
        mgr = KVCacheManager()
        mgr.allocate("req-1", n_layers=12)
        assert mgr.evict("req-1") is True
        assert mgr.evict("req-1") is False

    def test_evict_all(self):
        mgr = KVCacheManager()
        mgr.allocate("a", n_layers=12)
        mgr.allocate("b", n_layers=12)
        mgr.allocate("c", n_layers=12)
        evicted = mgr.evict_all()
        assert sorted(evicted) == ["a", "b", "c"]
        assert len(mgr) == 0


# ──────────────────────────────────────────────────────────────────────────
# LRU eviction
# ──────────────────────────────────────────────────────────────────────────


class TestLRUEviction:
    def test_lru_evicts_oldest_at_cap(self):
        mgr = KVCacheManager(max_cached_requests=3)
        mgr.allocate("a", n_layers=12)
        mgr.allocate("b", n_layers=12)
        mgr.allocate("c", n_layers=12)
        # Pool full. Allocate "d" → LRU "a" gets evicted.
        mgr.allocate("d", n_layers=12)
        assert "a" not in mgr
        assert "b" in mgr
        assert "c" in mgr
        assert "d" in mgr
        assert len(mgr) == 3

    def test_lru_eviction_respects_get_touches(self):
        # Touched-via-get handles survive longer than
        # un-touched ones.
        mgr = KVCacheManager(max_cached_requests=2)
        mgr.allocate("a", n_layers=12)
        mgr.allocate("b", n_layers=12)
        # Touch "a" — moves to LRU end. Now LRU is "b".
        mgr.get("a")
        mgr.allocate("c", n_layers=12)
        assert "a" in mgr
        assert "b" not in mgr  # b was LRU, evicted
        assert "c" in mgr


# ──────────────────────────────────────────────────────────────────────────
# TTL eviction
# ──────────────────────────────────────────────────────────────────────────


class TestTTLEviction:
    def test_evict_idle_drops_past_ttl_handles(self):
        clock = _ManualClock(start=1000.0)
        mgr = KVCacheManager(ttl_seconds=10.0, clock=clock)
        mgr.allocate("a", n_layers=12)
        mgr.allocate("b", n_layers=12)
        # Advance past TTL.
        clock.now = 1011.0
        evicted = mgr.evict_idle()
        assert sorted(evicted) == ["a", "b"]
        assert len(mgr) == 0

    def test_evict_idle_keeps_recently_touched_handles(self):
        clock = _ManualClock(start=1000.0)
        mgr = KVCacheManager(ttl_seconds=10.0, clock=clock)
        mgr.allocate("a", n_layers=12)
        mgr.allocate("b", n_layers=12)
        # Advance time, then touch "b" — its last_touch
        # updates to 1005.0.
        clock.now = 1005.0
        mgr.get("b")
        # Advance past "a"'s TTL but not "b"'s.
        clock.now = 1011.0
        evicted = mgr.evict_idle()
        assert evicted == ["a"]
        assert "b" in mgr

    def test_evict_idle_returns_empty_when_no_idles(self):
        clock = _ManualClock(start=1000.0)
        mgr = KVCacheManager(ttl_seconds=300.0, clock=clock)
        mgr.allocate("a", n_layers=12)
        # No time advance — no eviction.
        assert mgr.evict_idle() == []
        assert "a" in mgr


# ──────────────────────────────────────────────────────────────────────────
# Concurrent allocation thread safety
# ──────────────────────────────────────────────────────────────────────────


class TestConcurrentAllocation:
    def test_concurrent_allocate_distinct_ids(self):
        # 16 threads each allocate 8 distinct request_ids = 128
        # total. All should land in the manager (cap large
        # enough to hold them all).
        mgr = KVCacheManager(max_cached_requests=256)
        n_threads = 16
        per_thread = 8
        errors: List[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(per_thread):
                    mgr.allocate(
                        f"t{tid}-r{i}", n_layers=12,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(mgr) == n_threads * per_thread

    def test_concurrent_allocate_same_id_one_wins(self):
        # 16 threads race to allocate the same request_id;
        # exactly ONE allocation succeeds; the other 15 get
        # CacheAlreadyAllocatedError.
        mgr = KVCacheManager(max_cached_requests=64)
        successes: List[KVCacheHandle] = []
        already_errors: List[CacheAlreadyAllocatedError] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                handle = mgr.allocate("contended", n_layers=12)
                with lock:
                    successes.append(handle)
            except CacheAlreadyAllocatedError as exc:
                with lock:
                    already_errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1
        assert len(already_errors) == 15

    def test_concurrent_allocate_evict_no_corruption(self):
        # Mixed allocate + evict workload. Each thread does
        # 50 allocate-then-evict cycles. After all threads,
        # the manager should be empty + no exceptions raised.
        mgr = KVCacheManager(max_cached_requests=256)
        errors: List[Exception] = []

        def worker(tid: int) -> None:
            try:
                for i in range(50):
                    rid = f"t{tid}-i{i}"
                    mgr.allocate(rid, n_layers=12)
                    mgr.evict(rid)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(mgr) == 0
