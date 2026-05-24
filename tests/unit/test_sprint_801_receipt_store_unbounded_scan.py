"""Sprint 801 — uncapped receipt scans by node_id for long-running operators.

Sprint 793 noted the load-bearing limitation: the
_ReceiptStoreAdapter (sprints 791/792's earnings-by-node-id
flow) called `receipt_store.list(offset=0, limit=1000)`, but
ReceiptStore's in-memory cache caps at 1024 entries
(_DEFAULT_MAX_ENTRIES). Even with persist_dir set, the
on-disk receipts beyond cache eviction are unreachable —
queries return only the LRU window.

For a long-running operator (months of receipts, 100s per
day), this means earnings-by-node history silently truncates.
A device that earned 80% of its FTNS in months 1-5 shows as
zero by month 6 because all those receipts have evicted from
the cache.

Sprint 801 closes that with a new ReceiptStore method:

  list_for_node_ids(node_ids) -> List[Dict[str, Any]]

When `persist_dir` is set, iterates the on-disk file glob
(unbounded by cache size) + filters by settler_node_id ∈
node_ids. When `persist_dir` is None, falls back to scanning
the cache (preserving pre-801 behavior for in-memory-only
test fixtures + dev configs).

Sprint-793's `_ReceiptStoreAdapter` switches from
`list(limit=1000)` to this new method, so the
/devices/earnings endpoint now sees the full history when
persistence is enabled.

Pin tests:
- list_for_node_ids method exists on ReceiptStore.
- Empty node_ids → [].
- No-persist (cache-only): filters cache by settler_node_id.
- With persist: scans disk + returns ALL matches (test
  populates 50 receipts then evicts to 5-entry cache —
  query MUST still return all 50 matches).
- _ReceiptStoreAdapter delegates to list_for_node_ids
  (source-shape regression guard).
- Sprint 793's existing adapter tests still pass (regression).
"""
from __future__ import annotations

import inspect
from pathlib import Path


def _make_receipt_dict(job_id: str, settler_node_id: str):
    return {
        "job_id": job_id,
        "request_id": f"r-{job_id}",
        "model_id": "gpt2",
        "content_tier": "A",
        "privacy_tier": "none",
        "epsilon_spent": 0.0,
        "tee_type": "software",
        "tee_attestation": "6174",
        "output_hash": "dead",
        "duration_seconds": 1.0,
        "cost_ftns": "1.0",
        "settler_signature": "",
        "settler_node_id": settler_node_id,
        "streamed_output": False,
    }


# ---- Method exists --------------------------------------------


def test_list_for_node_ids_method_exists():
    from prsm.node.receipt_store import ReceiptStore
    assert hasattr(ReceiptStore, "list_for_node_ids")


# ---- Empty input ----------------------------------------------


def test_empty_node_ids_returns_empty():
    from prsm.node.receipt_store import ReceiptStore
    store = ReceiptStore(persist_dir=None)
    store.put("j1", _make_receipt_dict("j1", "a" * 32))
    assert store.list_for_node_ids([]) == []


# ---- Cache-only filtering -------------------------------------


def test_no_persist_filters_cache():
    from prsm.node.receipt_store import ReceiptStore
    store = ReceiptStore(persist_dir=None)
    store.put("j1", _make_receipt_dict("j1", "a" * 32))
    store.put("j2", _make_receipt_dict("j2", "b" * 32))
    store.put("j3", _make_receipt_dict("j3", "a" * 32))
    out = store.list_for_node_ids(["a" * 32])
    assert len(out) == 2
    for r in out:
        assert r["settler_node_id"] == "a" * 32


# ---- LOAD-BEARING: disk scan beyond cache cap ----------------


def test_disk_scan_beats_cache_cap(tmp_path: Path):
    """Populate 50 receipts with cache cap = 5. After eviction,
    cache holds only the 5 most-recent. list_for_node_ids MUST
    still return all 50 matches from disk."""
    from prsm.node.receipt_store import ReceiptStore
    store = ReceiptStore(
        max_entries=5, persist_dir=tmp_path,
    )
    for i in range(50):
        store.put(
            f"j{i:03d}",
            _make_receipt_dict(f"j{i:03d}", "a" * 32),
        )
    # After eviction, cache holds only 5
    assert len(store._cache) == 5
    # But disk has all 50
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 50

    # The load-bearing assertion: list_for_node_ids returns ALL
    # 50, not just the cache window.
    out = store.list_for_node_ids(["a" * 32])
    assert len(out) == 50


def test_disk_scan_filters_by_node_id(tmp_path: Path):
    """Mixed node_ids on disk → only matches returned."""
    from prsm.node.receipt_store import ReceiptStore
    store = ReceiptStore(max_entries=3, persist_dir=tmp_path)
    for i in range(10):
        nid = ("a" * 32) if i % 2 == 0 else ("b" * 32)
        store.put(
            f"j{i:03d}",
            _make_receipt_dict(f"j{i:03d}", nid),
        )
    out_a = store.list_for_node_ids(["a" * 32])
    assert len(out_a) == 5
    for r in out_a:
        assert r["settler_node_id"] == "a" * 32

    out_both = store.list_for_node_ids(["a" * 32, "b" * 32])
    assert len(out_both) == 10


def test_disk_scan_unknown_node_id_returns_empty(tmp_path: Path):
    from prsm.node.receipt_store import ReceiptStore
    store = ReceiptStore(max_entries=3, persist_dir=tmp_path)
    store.put("j1", _make_receipt_dict("j1", "a" * 32))
    out = store.list_for_node_ids(["c" * 32])
    assert out == []


# ---- _ReceiptStoreAdapter delegates ---------------------------


def test_adapter_uses_list_for_node_ids():
    """Source-shape: _ReceiptStoreAdapter.list_receipts_for_node_ids
    calls store.list_for_node_ids (not the old .list pattern)."""
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter
    src = inspect.getsource(
        _ReceiptStoreAdapter.list_receipts_for_node_ids,
    )
    assert "list_for_node_ids" in src, (
        "Sprint 801: adapter must delegate to "
        "ReceiptStore.list_for_node_ids so disk-backed stores "
        "return the full history"
    )


def test_adapter_with_persist_returns_full_history(tmp_path: Path):
    """End-to-end: build store w/ persist + small cache + 30
    receipts → adapter returns ALL 30, not just cache window."""
    from prsm.node.receipt_store import ReceiptStore
    from prsm.node.wallet_api_wiring import _ReceiptStoreAdapter

    store = ReceiptStore(max_entries=3, persist_dir=tmp_path)
    for i in range(30):
        store.put(
            f"j{i:03d}",
            _make_receipt_dict(f"j{i:03d}", "a" * 32),
        )
    adapter = _ReceiptStoreAdapter(receipt_store=store)
    out = adapter.list_receipts_for_node_ids(["a" * 32])
    assert len(out) == 30
