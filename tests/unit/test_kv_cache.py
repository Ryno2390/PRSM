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


# ──────────────────────────────────────────────────────────────────────────
# Phase 3.x.11.y Task 2 — KVCacheManager.rollback
# ──────────────────────────────────────────────────────────────────────────


class TestRollback:
    def _setup_handle_with_tokens(
        self, mgr, request_id="req-1", tokens=4,
    ):
        """Helper: allocate handle + simulate that N tokens have
        been generated. Payload is a list of strings tagged by
        position so tests can verify truncation."""
        handle = mgr.allocate(request_id, n_layers=2)
        handle.payload = [f"pos_{i}" for i in range(tokens)]
        handle.tokens_generated = tokens
        return handle

    def test_rollback_happy_path_drops_n(self):
        mgr = KVCacheManager()
        handle = self._setup_handle_with_tokens(mgr, tokens=5)

        def truncate_fn(payload, n):
            return payload[:-n]

        rolled, dropped = mgr.rollback("req-1", 2, truncate_fn)
        assert rolled is True
        assert dropped == 2
        assert handle.payload == ["pos_0", "pos_1", "pos_2"]
        assert handle.tokens_generated == 3

    def test_rollback_unknown_request_id_returns_false_zero(self):
        mgr = KVCacheManager()
        rolled, dropped = mgr.rollback(
            "never-allocated", 4, lambda p, n: p[:-n],
        )
        assert rolled is False
        assert dropped == 0

    def test_rollback_zero_n_is_noop(self):
        mgr = KVCacheManager()
        handle = self._setup_handle_with_tokens(mgr, tokens=4)
        truncate_calls = []

        def truncate_fn(payload, n):
            truncate_calls.append(n)
            return payload[:-n]

        rolled, dropped = mgr.rollback("req-1", 0, truncate_fn)
        assert rolled is False
        assert dropped == 0
        # truncate_fn MUST NOT be called for the no-op path.
        assert truncate_calls == []
        # Handle state unchanged.
        assert handle.tokens_generated == 4
        assert handle.payload == ["pos_0", "pos_1", "pos_2", "pos_3"]

    def test_rollback_negative_n_is_noop(self):
        mgr = KVCacheManager()
        self._setup_handle_with_tokens(mgr, tokens=4)
        rolled, dropped = mgr.rollback("req-1", -3, lambda p, n: p)
        assert rolled is False
        assert dropped == 0

    def test_rollback_past_tokens_generated_drops_all(self):
        # Asking to drop 100 when only 4 tokens generated → drops
        # all 4, returns dropped=4.
        mgr = KVCacheManager()
        handle = self._setup_handle_with_tokens(mgr, tokens=4)

        def truncate_fn(payload, n):
            return payload[:-n] if n < len(payload) else []

        rolled, dropped = mgr.rollback("req-1", 100, truncate_fn)
        assert rolled is True
        assert dropped == 4
        assert handle.payload == []
        assert handle.tokens_generated == 0

    def test_rollback_when_tokens_generated_zero_is_noop(self):
        # PREFILL alone hasn't generated any tokens — rollback
        # is a no-op even though the handle exists.
        mgr = KVCacheManager()
        handle = mgr.allocate("req-1", n_layers=2)
        handle.payload = ["pos_0", "pos_1"]
        handle.tokens_generated = 0
        truncate_calls = []
        rolled, dropped = mgr.rollback(
            "req-1", 4, lambda p, n: truncate_calls.append(n) or p,
        )
        assert rolled is False
        assert dropped == 0
        assert truncate_calls == []

    def test_rollback_truncate_fn_returned_payload_stored(self):
        # Manager stores whatever truncate_fn returns — the
        # payload may be the same reference (mutated in place)
        # or a new object. Test verifies the manager doesn't
        # silently drop the returned payload.
        mgr = KVCacheManager()
        handle = self._setup_handle_with_tokens(mgr, tokens=3)
        new_payload_marker = ["FRESH_PAYLOAD"]

        def truncate_fn(payload, n):
            return new_payload_marker

        mgr.rollback("req-1", 1, truncate_fn)
        assert handle.payload is new_payload_marker

    def test_rollback_validation_rejects_empty_request_id(self):
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="request_id"):
            mgr.rollback("", 2, lambda p, n: p)

    def test_rollback_validation_rejects_non_int_n(self):
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="n_positions"):
            mgr.rollback("req-1", "two", lambda p, n: p)

    def test_rollback_validation_rejects_bool_n(self):
        # bool is a subclass of int; explicitly rejected
        # (mirrors allocate's bool-rejection pattern).
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="n_positions"):
            mgr.rollback("req-1", True, lambda p, n: p)

    def test_rollback_validation_rejects_non_callable_truncate_fn(self):
        mgr = KVCacheManager()
        with pytest.raises(RuntimeError, match="truncate_fn"):
            mgr.rollback("req-1", 2, "not callable")

    def test_rollback_concurrent_with_get_thread_safety(self):
        # Concurrent rollback + get-incremental contention test.
        # 4 threads rollback the same handle while 4 threads call
        # get(). Manager's lock guarantees atomic rollback (no
        # torn read of handle.payload between truncate_fn return
        # and tokens_generated decrement). After all threads
        # finish, the payload+tokens_generated MUST be consistent.
        mgr = KVCacheManager()
        handle = self._setup_handle_with_tokens(mgr, tokens=20)

        def truncate_fn(payload, n):
            # Return a payload of length (current - n). Simulates
            # the actual truncation; thread-safe because we're
            # called inside the manager's lock.
            return payload[:-n] if n < len(payload) else []

        errors = []
        rollback_total = [0]
        rollback_lock = threading.Lock()

        def rollback_worker():
            try:
                for _ in range(5):
                    rolled, dropped = mgr.rollback(
                        "req-1", 1, truncate_fn,
                    )
                    if rolled:
                        with rollback_lock:
                            rollback_total[0] += dropped
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def get_worker():
            try:
                for _ in range(5):
                    h = mgr.get("req-1")
                    if h is not None:
                        # Read tokens_generated + payload length;
                        # they MUST be consistent (manager's lock
                        # holds during rollback's payload+counter
                        # update).
                        n_pay = len(h.payload)
                        n_tok = h.tokens_generated
                        # Loose invariant: payload length + dropped
                        # so far should always equal initial 20.
                        # (Tight check would require atomic snapshot
                        # of (payload, tokens_generated, dropped).)
                        # For this test, just confirm both fields
                        # are well-formed ints/lists.
                        assert isinstance(n_pay, int) and n_pay >= 0
                        assert isinstance(n_tok, int) and n_tok >= 0
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = (
            [threading.Thread(target=rollback_worker) for _ in range(4)]
            + [threading.Thread(target=get_worker) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Total dropped + remaining payload length == 20.
        final_payload_len = len(handle.payload)
        final_tokens = handle.tokens_generated
        assert rollback_total[0] + final_payload_len == 20
        # Counter consistent with payload state.
        assert final_tokens == final_payload_len
