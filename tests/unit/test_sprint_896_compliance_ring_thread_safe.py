"""Sprint 896 — FiatComplianceRing is thread-safe.

sp887 race finding, made REACHABLE by sp894. Before sp894 the funnel
auto-sweep ran inline on the single asyncio event-loop thread, so its
compliance-ring writes (sp885 records onramp_execute on every CONFIRMED
intent) were serialized with the `/wallet/onramp/execute` tier check
that READS the ring (total_usd_for_user). No true concurrency → no race.

sp894 offloaded the sweep to a worker thread to stop it blocking the
event loop. Correct fix — but it means the sweep thread now calls
ring.record() (deque.append) CONCURRENTLY with the event loop's tier
check iterating the same deque. CPython's deque raises
``RuntimeError: deque mutated during iteration`` when iterated during a
concurrent append — so the financial enforcement gate (and the
summary / recent / export readers) can crash under load.

sp896 adds a mutex: record() mutates under the lock; readers snapshot
the deque under the lock then iterate the snapshot lock-free (keeps the
hot-path lock hold short). The node wires the SAME ring into both the
event loop and the sweep thread (node.py _do_sweep), so this is the
real production topology.
"""
from __future__ import annotations

import sys
import threading

import pytest

from prsm.economy.web3.fiat_compliance_ring import FiatComplianceRing


@pytest.fixture
def fast_gil_switch():
    """Force frequent GIL hand-off so a concurrent iterate/append
    race surfaces deterministically instead of by luck."""
    prev = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(prev)


def _record_execute(ring, user="alice", n=1):
    for _ in range(n):
        ring.record(
            kind="onramp_execute",
            user_id=user,
            usd_amount=1.0,
            ftns_amount=0.0,
            status="CONFIRMED",
        )


def _run_writer_reader(ring, reader_call, *, prefill, writes):
    """Pre-fill the ring, then run a writer thread (records executes)
    concurrently with a reader thread (reader_call(ring) in a loop).
    Returns the list of exceptions caught in EITHER thread."""
    _record_execute(ring, n=prefill)
    errors = []
    stop = threading.Event()

    def writer():
        try:
            _record_execute(ring, n=writes)
        except Exception as exc:  # noqa: BLE001
            errors.append(("writer", exc))
        finally:
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                reader_call(ring)
        except Exception as exc:  # noqa: BLE001
            errors.append(("reader", exc))

    r = threading.Thread(target=reader)
    w = threading.Thread(target=writer)
    r.start()
    w.start()
    w.join()
    r.join()
    return errors


# ── No crash under concurrent write + read ───────────────────

def test_total_usd_no_crash_under_concurrent_record(fast_gil_switch):
    """The tier-check read path (total_usd_for_user) must not crash
    while the sweep thread records executes. Pre-sp896 this raises
    'deque mutated during iteration'."""
    ring = FiatComplianceRing()
    errors = _run_writer_reader(
        ring,
        lambda r: r.total_usd_for_user("alice"),
        prefill=15_000,
        writes=120_000,
    )
    assert errors == [], f"concurrency crashed: {errors[:1]}"


def test_summary_no_crash_under_concurrent_record(fast_gil_switch):
    ring = FiatComplianceRing()
    errors = _run_writer_reader(
        ring,
        lambda r: r.summary_by_kind(),
        prefill=15_000,
        writes=120_000,
    )
    assert errors == [], f"concurrency crashed: {errors[:1]}"


def test_recent_no_crash_under_concurrent_record(fast_gil_switch):
    ring = FiatComplianceRing()
    errors = _run_writer_reader(
        ring,
        lambda r: r.recent(limit=100),
        prefill=15_000,
        writes=120_000,
    )
    assert errors == [], f"concurrency crashed: {errors[:1]}"


# ── Correctness preserved under concurrency ──────────────────

def test_total_usd_correct_after_concurrent_writes(fast_gil_switch):
    """All concurrent records must land — no lost appends. After the
    writers settle, the rolling total is the exact sum."""
    ring = FiatComplianceRing()
    threads = [
        threading.Thread(target=_record_execute, args=(ring, "bob", 5_000))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 4 threads × 5_000 executes × $1.00 each.
    assert ring.count() == 20_000
    assert ring.total_usd_for_user("bob") == pytest.approx(20_000.0)


def test_single_threaded_behavior_unchanged():
    """Regression: the lock must not change single-threaded results."""
    ring = FiatComplianceRing()
    ring.record(
        kind="onramp_execute", user_id="carol",
        usd_amount=250.0, ftns_amount=0.0, status="CONFIRMED",
    )
    ring.record(
        kind="onramp_quote", user_id="carol",
        usd_amount=999.0, ftns_amount=0.0, status="QUOTED",
    )
    # Only executes burn the rolling total (sp885).
    assert ring.total_usd_for_user("carol") == 250.0
    assert ring.count() == 2
    assert ring.summary_by_kind()["onramp_execute"]["count"] == 1
