"""Unit tests for prsm.economy.web3.tx_lock_registry — Phase 7 §8.8."""
from __future__ import annotations

import threading
import time

import pytest

from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY, TxLockRegistry


class TestGetLock:
    def test_returns_lock_instance(self):
        r = TxLockRegistry()
        lock = r.get_lock("0xdeadbeef")
        assert isinstance(lock, type(threading.Lock()))

    def test_same_address_same_lock(self):
        r = TxLockRegistry()
        addr = "0xAbCdEf1234567890aBcDeF1234567890AbCdEf12"
        assert r.get_lock(addr) is r.get_lock(addr)

    def test_checksum_variation_same_lock(self):
        """EIP-55 checksum upper/lowercase MUST resolve to the same lock."""
        r = TxLockRegistry()
        upper = "0xABCDEF1234567890ABCDEF1234567890ABCDEF12"
        lower = "0xabcdef1234567890abcdef1234567890abcdef12"
        mixed = "0xAbCdEf1234567890aBcDeF1234567890AbCdEf12"
        lock_upper = r.get_lock(upper)
        lock_lower = r.get_lock(lower)
        lock_mixed = r.get_lock(mixed)
        assert lock_upper is lock_lower
        assert lock_upper is lock_mixed
        assert r._known_addresses() == 1

    def test_different_addresses_different_locks(self):
        r = TxLockRegistry()
        addr_a = "0x0000000000000000000000000000000000000001"
        addr_b = "0x0000000000000000000000000000000000000002"
        lock_a = r.get_lock(addr_a)
        lock_b = r.get_lock(addr_b)
        assert lock_a is not lock_b
        assert r._known_addresses() == 2

    def test_rejects_empty_address(self):
        r = TxLockRegistry()
        with pytest.raises(ValueError):
            r.get_lock("")

    def test_rejects_non_string(self):
        r = TxLockRegistry()
        with pytest.raises(ValueError):
            r.get_lock(None)  # type: ignore[arg-type]


class TestConcurrentAccess:
    def test_registry_thread_safe_creates_single_lock(self):
        """100 threads concurrently asking for the same address must all
        get the same lock — registry-lock serializes creation."""
        r = TxLockRegistry()
        addr = "0x1111111111111111111111111111111111111111"
        locks = []
        start_barrier = threading.Barrier(100)

        def worker():
            start_barrier.wait()  # synchronize the race
            locks.append(r.get_lock(addr))

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = locks[0]
        assert all(lock is first for lock in locks), (
            "registry created multiple locks for the same address"
        )
        assert r._known_addresses() == 1

    def test_same_lock_serializes_across_clients(self):
        """The core §8.8 property: two client-like actors holding locks
        obtained from the registry for the SAME address cannot both enter
        the critical section simultaneously."""
        r = TxLockRegistry()
        addr = "0x2222222222222222222222222222222222222222"

        enter_order = []
        exit_order = []
        inside_lock = threading.Event()

        def client_a():
            lock = r.get_lock(addr)
            with lock:
                enter_order.append("A")
                inside_lock.set()
                # Hold the lock long enough for B to try and fail
                time.sleep(0.05)
                exit_order.append("A")

        def client_b():
            inside_lock.wait()  # ensure A is inside
            lock = r.get_lock(addr)
            with lock:
                enter_order.append("B")
                exit_order.append("B")

        t_a = threading.Thread(target=client_a)
        t_b = threading.Thread(target=client_b)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        # B must not have entered until A exited
        assert enter_order == ["A", "B"]
        assert exit_order == ["A", "B"]

    def test_different_addresses_do_not_block(self):
        """Sanity: two different accounts' txs run concurrently."""
        r = TxLockRegistry()
        addr_a = "0x3333333333333333333333333333333333333333"
        addr_b = "0x4444444444444444444444444444444444444444"

        results = []

        def worker(addr, label):
            lock = r.get_lock(addr)
            with lock:
                start = time.time()
                time.sleep(0.03)
                results.append((label, start, time.time()))

        t_a = threading.Thread(target=worker, args=(addr_a, "A"))
        t_b = threading.Thread(target=worker, args=(addr_b, "B"))
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        # Both started within the same critical-section window
        a = next(r for r in results if r[0] == "A")
        b = next(r for r in results if r[0] == "B")
        # If they'd been serialized, the later one's start would be after
        # the earlier one's end (> 0.03s gap). Allow some scheduling slack.
        gap = abs(a[1] - b[1])
        assert gap < 0.025, (
            f"different-address locks serialized; start times {gap:.4f}s apart"
        )


class TestModuleGlobal:
    def test_module_global_singleton(self):
        """Module-level TX_LOCK_REGISTRY returns stable locks across imports."""
        addr = "0x5555555555555555555555555555555555555555"
        lock1 = TX_LOCK_REGISTRY.get_lock(addr)
        lock2 = TX_LOCK_REGISTRY.get_lock(addr)
        assert lock1 is lock2

    def test_real_eth_account_address_works(self):
        """Integration sanity: eth_account generates mixed-case addresses;
        registry must handle them without raising."""
        try:
            from eth_account import Account
        except ImportError:
            pytest.skip("eth_account not installed")

        acct1 = Account.create()
        acct2 = Account.create()
        lock1 = TX_LOCK_REGISTRY.get_lock(acct1.address)
        lock2 = TX_LOCK_REGISTRY.get_lock(acct2.address)
        assert lock1 is not lock2
        # Re-fetch with the same address variably cased
        assert TX_LOCK_REGISTRY.get_lock(acct1.address.lower()) is lock1
        assert TX_LOCK_REGISTRY.get_lock(acct1.address.upper().replace("0X", "0x")) is lock1
