"""Process-wide per-account tx lock registry — resolves Phase 7 §8.8.

Different web3 clients (StakeManager, ProvenanceRegistry, BatchSettlement, etc.)
that share the same signing account MUST serialize their
``build_transaction → sign_transaction → send_raw_transaction`` sequence to
avoid nonce collisions.

Prior to this module each client held its own ``threading.Lock`` (see
``stake_manager.py:309`` and ``provenance_registry.py:209`` pre-§8.8). Per-client
locks correctly serialize txs submitted through a single client, but two
clients of the same account hold two independent locks and can both read
the same ``"pending"`` nonce from ``eth.get_transaction_count`` — the second
client's tx is then broadcast with a nonce already in-flight and fails with
``replacement transaction underpriced`` or ``nonce too low``.

The shared-key scenario is common in PRSM operator deployments: a single
provider node typically runs both StakeBond interactions (bond/unbond/
claim_bounty) AND ProvenanceRegistry interactions (register_content) from
the same signing key.

This module resolves that within a single process. Cross-process coordination
remains the operator's responsibility (see OPERATOR_GUIDE.md §On-chain
Keypairs).

Usage from inside a web3 client::

    from prsm.economy.web3.tx_lock_registry import TX_LOCK_REGISTRY

    class StakeManager:
        def __init__(self, ...):
            self._tx_lock = TX_LOCK_REGISTRY.get_lock(self._account.address)

        def bond(self, ...):
            with self._tx_lock:
                tx = self.contract.functions.bond(...).build_transaction(...)
                return self._sign_and_send(tx)

Invariants:
    - ``get_lock(addr)`` returns the SAME ``threading.Lock`` instance on every
      call for the same logical address (EIP-55 case-insensitive).
    - Different addresses return different locks.
    - The registry itself is thread-safe (guarded by an internal registry lock).
    - No garbage collection: once a lock is created for an address, it lives
      for the lifetime of the process. This is intentional — an active client
      holding a reference while the registry drops the entry would create
      two independent locks for the same address, re-introducing the bug.
"""
from __future__ import annotations

import threading
from typing import Dict


class TxLockRegistry:
    """Process-wide registry of per-account locks for on-chain tx submission."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}

    def get_lock(self, address: str) -> threading.Lock:
        if not isinstance(address, str) or not address:
            raise ValueError(f"address must be a non-empty string; got {address!r}")
        key = address.lower()
        with self._registry_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def _known_addresses(self) -> int:
        """Test-only observability hook: count of distinct accounts seen."""
        with self._registry_lock:
            return len(self._locks)


TX_LOCK_REGISTRY = TxLockRegistry()
