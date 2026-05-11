"""Sprint 250 — ReceiptStore.list() + count() for receipt enumeration.

Sprint 242 only exposed get(job_id) lookup. Auditors + operators
need to enumerate all stored receipts (paginated). This sprint
adds:
  - ReceiptStore.list(offset, limit) → most-recently-put first
  - ReceiptStore.count() → already exists as __len__, but
    expose as a method for parity with JobHistoryStore.count()
  - Optional model_id filter so operators can audit "all
    receipts for mock-llama-3-8b"
"""
from __future__ import annotations

from prsm.node.receipt_store import ReceiptStore


def _r(job_id, model_id="m1"):
    return {
        "job_id": job_id,
        "model_id": model_id,
        "cost_ftns": "0.10",
    }


def test_list_returns_most_recent_first():
    s = ReceiptStore()
    s.put("a", _r("a"))
    s.put("b", _r("b"))
    s.put("c", _r("c"))
    listing = s.list(offset=0, limit=10)
    assert [r["job_id"] for r in listing] == ["c", "b", "a"]


def test_list_pagination():
    s = ReceiptStore()
    for i in range(5):
        s.put(f"j{i}", _r(f"j{i}"))
    # Newest-first: j4, j3, j2, j1, j0
    page1 = s.list(offset=0, limit=2)
    assert [r["job_id"] for r in page1] == ["j4", "j3"]
    page2 = s.list(offset=2, limit=2)
    assert [r["job_id"] for r in page2] == ["j2", "j1"]


def test_list_empty():
    s = ReceiptStore()
    assert s.list(offset=0, limit=10) == []


def test_count_method():
    s = ReceiptStore()
    assert s.count() == 0
    s.put("a", _r("a"))
    s.put("b", _r("b"))
    assert s.count() == 2


def test_model_id_filter():
    s = ReceiptStore()
    s.put("a", _r("a", "m1"))
    s.put("b", _r("b", "m2"))
    s.put("c", _r("c", "m1"))
    filtered = s.list(model_id="m1", offset=0, limit=10)
    assert {r["job_id"] for r in filtered} == {"a", "c"}


def test_invalid_limit_rejected():
    import pytest
    s = ReceiptStore()
    with pytest.raises(ValueError):
        s.list(offset=0, limit=0)
    with pytest.raises(ValueError):
        s.list(offset=0, limit=1001)


def test_invalid_offset_rejected():
    import pytest
    s = ReceiptStore()
    with pytest.raises(ValueError):
        s.list(offset=-1, limit=10)
