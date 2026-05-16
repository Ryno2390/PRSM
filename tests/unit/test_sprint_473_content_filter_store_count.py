"""Sprint 473 — F21 regression pin.

F21: ContentFilterStore lacked the `count()` method that
`/health/detailed`'s subsystem-record-count probe (added in
sprint 343) expects. The probe raised AttributeError, which
the defensive try/except in `_orchestrator_subsystem` caught
and translated into `status: error, available: false`.

Because `content_filter_store` is in the `optional` subsystem
list (sprint 343), having it report `available: false` (without
the `not_wired`/`disabled` opt-out status) flipped the
top-level health to `degraded`.

Live impact: operator dashboards + alerting rules keyed on
`status == "degraded"` would fire continuously, drowning out
real alerts. Sprint 473 added a `count()` method that returns
the sum of CIDs + tags + patterns.

These pins fire if a future refactor removes the method again.
"""
from __future__ import annotations

from prsm.node.content_filter_store import ContentFilterStore


def test_content_filter_store_has_count_method():
    """The method must exist + be callable. Without it, the
    health probe crashes."""
    assert hasattr(ContentFilterStore, "count")
    assert callable(getattr(ContentFilterStore, "count"))


def test_content_filter_store_count_returns_zero_on_empty():
    """Empty store reports 0. Verifies the method works
    without state."""
    store = ContentFilterStore()
    assert store.count() == 0


def test_content_filter_store_count_sums_all_categories():
    """The count is the sum of CIDs + tags + patterns. A
    refactor that drops one of the three categories from the
    count would be a regression in operator visibility — the
    health endpoint should report a non-zero record_count if
    ANY filter entry exists."""
    store = ContentFilterStore()
    store.add_cids(["cid-a", "cid-b"])
    assert store.count() == 2
    store.add_tags(["spam"])
    assert store.count() == 3
    store.add_patterns([r"\bbad\b"])
    assert store.count() == 4


def test_content_filter_store_count_matches_to_dict_sums():
    """The to_dict() snapshot already exposes count_cids +
    count_tags + count_patterns. The count() method must
    agree with their sum, so dashboards reading either path
    can't disagree about how many filter entries exist."""
    store = ContentFilterStore()
    store.add_cids(["cid-a"])
    store.add_tags(["t1", "t2"])
    store.add_patterns([r"x"])
    d = store.to_dict()
    expected = (
        d["count_cids"] + d["count_tags"]
        + d["count_patterns"]
    )
    assert store.count() == expected
