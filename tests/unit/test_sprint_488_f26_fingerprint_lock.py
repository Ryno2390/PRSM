"""Sprint 488 — F26 fingerprint dedup race fix pins.

F26 had two root causes:

1. ContentFingerprintRegistry was constructed INSIDE
   `_build_query_orchestrator_or_none` (node.py), which
   early-returns when `PRSM_QUERY_ORCHESTRATOR_ENABLED` is
   unset (the default). Result: every non-QO daemon had
   `_content_fingerprint_registry = None` → the §14
   anti-Sybil first-creator-wins NEVER fired regardless
   of concurrency.

2. The `register()` method's get → insert → disk-write
   sequence was not atomic. Concurrent callers all saw
   `existing is None` and all inserted.

Sprint 488 fix:
  - Registry construction MOVED to unconditional node init
    (in `__init__` before the `_build_query_orchestrator_or_none`
    call).
  - `threading.Lock` added around the register() critical
    section.

These pins defend both invariants in the source so a future
refactor can't silently re-introduce either bug.
"""
from __future__ import annotations

import threading
from pathlib import Path

from prsm.marketplace.content_fingerprint_registry import (
    ContentFingerprintRegistry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_registry_has_threading_lock():
    """The registry must hold a process-wide lock for the
    check-then-insert window. Without this, concurrent
    callers all see `existing is None` and all insert."""
    reg = ContentFingerprintRegistry()
    assert hasattr(reg, "_lock"), (
        "ContentFingerprintRegistry must have a _lock "
        "attribute — F26 regression risk"
    )
    # Confirm it's a real lock (has acquire/release).
    assert hasattr(reg._lock, "acquire")
    assert hasattr(reg._lock, "release")


def test_register_method_uses_lock():
    """The `register` method body must use the lock. Source
    pin — a refactor that splits the get + insert without
    holding the lock would silently re-introduce the race."""
    import inspect
    src = inspect.getsource(ContentFingerprintRegistry.register)
    # The lock acquisition must wrap the entire body.
    assert "self._lock" in src, (
        "register() must use self._lock — F26 regression"
    )
    assert "with self._lock" in src, (
        "register() must use `with self._lock:` context "
        "manager — bare acquire/release risks leaking on "
        "exception"
    )


def test_registry_wired_unconditionally_in_node_init():
    """The ContentFingerprintRegistry must be wired in
    Node.__init__ UNCONDITIONALLY — not inside the
    `_build_query_orchestrator_or_none` method which
    early-returns when QO isn't enabled.

    Pre-fix: registry construction was inside
    `_build_query_orchestrator_or_none`. With
    `PRSM_QUERY_ORCHESTRATOR_ENABLED` unset (default), the
    function returned None at line 4033 before reaching
    line 4133 → registry stayed None → /content/upload
    silently skipped the dedup check → §14 anti-Sybil
    invariant compromised on every default-configured
    daemon."""
    node_src = (
        REPO_ROOT / "prsm" / "node" / "node.py"
    ).read_text()
    # The unconditional init must appear AT TOP-LEVEL inside
    # __init__ — find the marker comment.
    # Marker can span lines — strip whitespace + comment hashes
    # before search.
    normalized = " ".join(
        line.lstrip("#").strip()
        for line in node_src.splitlines()
    )
    assert (
        "wired UNCONDITIONALLY"
    ) in normalized, (
        "Sprint 488 F26 fix marker missing — the registry "
        "init may have been moved back into "
        "_build_query_orchestrator_or_none"
    )

    # Belt-and-suspenders: the registry construction must
    # appear BEFORE the line `def _build_query_orchestrator_or_none`.
    init_idx = node_src.find(
        "self._content_fingerprint_registry = (\n"
        "                ContentFingerprintRegistry.from_env()\n"
        "            )"
    )
    qo_def_idx = node_src.find(
        "def _build_query_orchestrator_or_none"
    )
    # At least one occurrence must be BEFORE the method
    # definition (the unconditional one); the legacy
    # duplicate inside the method body is harmlessly left
    # in place per sprint 488's comment.
    first_init_idx = node_src.find(
        "self._content_fingerprint_registry"
    )
    assert 0 < first_init_idx < qo_def_idx, (
        "ContentFingerprintRegistry init must appear before "
        "the _build_query_orchestrator_or_none method body"
    )


def test_register_first_caller_returns_is_new_true():
    """Single-call sanity: the first registration for a
    content_hash returns (creator, True)."""
    reg = ContentFingerprintRegistry()
    canonical, is_new = reg.register(
        content_hash="a" * 64,
        creator_eth_address="0x" + "1" * 40,
    )
    assert canonical == "0x" + "1" * 40
    assert is_new is True


def test_register_second_caller_returns_is_new_false():
    """Second caller with a DIFFERENT address for the SAME
    content_hash returns (canonical_first, False) +
    increments duplicate_attempt_count. This is the
    first-creator-wins guarantee."""
    reg = ContentFingerprintRegistry()
    first = "0x" + "1" * 40
    second = "0x" + "2" * 40
    reg.register(content_hash="a" * 64, creator_eth_address=first)
    canonical, is_new = reg.register(
        content_hash="a" * 64, creator_eth_address=second,
    )
    assert canonical == first  # first creator wins
    assert is_new is False
    # The duplicate_attempt_count should have incremented.
    entry = reg._entries["a" * 64]
    assert entry.duplicate_attempt_count == 1


def test_register_under_thread_race_only_one_canonical():
    """The load-bearing integration pin: launch N threads
    that race to register the same content_hash with
    different addresses. Exactly ONE thread must observe
    `is_new=True`. The rest must observe `is_new=False`
    AND get back the SAME canonical creator.

    Pre-fix: dictionary-level race produced multiple
    `is_new=True` returns → all callers thought they
    were canonical."""
    reg = ContentFingerprintRegistry()
    content_hash = "b" * 64
    n_threads = 20
    results = []
    barrier = threading.Barrier(n_threads)

    def _race(idx: int) -> None:
        addr = "0x" + (f"{idx + 1:040x}")
        barrier.wait()  # synchronize starts
        canonical, is_new = reg.register(
            content_hash=content_hash,
            creator_eth_address=addr,
        )
        results.append((canonical, is_new, addr))

    threads = [
        threading.Thread(target=_race, args=(i,))
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    is_new_count = sum(1 for _, is_new, _ in results if is_new)
    canonicals = {c for c, is_new, _ in results if is_new}
    all_canonicals = {c for c, _, _ in results}

    assert is_new_count == 1, (
        f"exactly ONE thread should see is_new=True; got "
        f"{is_new_count} (anti-Sybil race not fixed)"
    )
    assert len(all_canonicals) == 1, (
        f"all threads must see the SAME canonical creator; "
        f"got {len(all_canonicals)} distinct: {all_canonicals}"
    )
    assert len(canonicals) == 1
