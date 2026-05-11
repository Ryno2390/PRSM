"""Sprint 246 — resolver maps contributing_shards → per-shard
metadata for the on-chain content-access royalty leg.

Pre-sprint: settlement-side code has the shard_cid list but no
clean path from that to (content_hash, creator_eth_address,
royalty_rate). Each settlement caller would write its own
ContentIndex iteration.

This sprint extracts `resolve_content_royalty_targets()` from
the settlement-call-site perspective:
  - Input: contributing_shards: List[str], content_index
  - Output: List[ContentRoyaltyTarget] with (shard_cid,
    content_hash, creator_eth_address, royalty_rate). Skips
    shards with missing records OR missing eth_address (the
    on-chain dispatch needs an address to distribute to).
  - Returns the kept-targets list AND a `skipped` count for
    operator telemetry.

Pure function. No side effects. No on-chain dispatch yet.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.content_royalty_resolver import (
    ContentRoyaltyTarget,
    resolve_content_royalty_targets,
)


def _fake_record(
    cid: str,
    content_hash: str = "ab" * 32,
    royalty_rate: float = 0.05,
    creator_eth_address: str | None = None,
):
    r = MagicMock()
    r.cid = cid
    r.content_hash = content_hash
    r.royalty_rate = royalty_rate
    r.creator_eth_address = creator_eth_address
    return r


def _fake_index(records_by_cid):
    idx = MagicMock()
    idx.lookup = lambda cid: records_by_cid.get(cid)
    return idx


def test_resolves_full_records():
    records = {
        "c1": _fake_record(
            "c1", creator_eth_address="0x" + "a" * 40,
        ),
        "c2": _fake_record(
            "c2", creator_eth_address="0x" + "b" * 40,
        ),
    }
    targets, skipped = resolve_content_royalty_targets(
        contributing_shards=["c1", "c2"],
        content_index=_fake_index(records),
    )
    assert skipped == 0
    assert len(targets) == 2
    assert isinstance(targets[0], ContentRoyaltyTarget)
    assert {t.shard_cid for t in targets} == {"c1", "c2"}
    assert {t.creator_eth_address for t in targets} == {
        "0x" + "a" * 40, "0x" + "b" * 40,
    }


def test_skips_missing_record():
    records = {
        "c1": _fake_record(
            "c1", creator_eth_address="0x" + "a" * 40,
        ),
    }
    targets, skipped = resolve_content_royalty_targets(
        contributing_shards=["c1", "c-missing"],
        content_index=_fake_index(records),
    )
    assert len(targets) == 1
    assert skipped == 1
    assert targets[0].shard_cid == "c1"


def test_skips_missing_eth_address():
    """v1 backwards-compat: pre-sprint-243 uploads have no
    eth_address. Skip them quietly — on-chain dispatch can't
    pay an unknown address."""
    records = {
        "c1": _fake_record(
            "c1", creator_eth_address="0x" + "a" * 40,
        ),
        "c2": _fake_record("c2", creator_eth_address=None),
        "c3": _fake_record("c3", creator_eth_address=""),
    }
    targets, skipped = resolve_content_royalty_targets(
        contributing_shards=["c1", "c2", "c3"],
        content_index=_fake_index(records),
    )
    assert len(targets) == 1
    assert skipped == 2


def test_no_shards_returns_empty():
    targets, skipped = resolve_content_royalty_targets(
        contributing_shards=[],
        content_index=_fake_index({}),
    )
    assert targets == []
    assert skipped == 0


def test_preserves_per_shard_royalty_rate():
    records = {
        "c1": _fake_record(
            "c1", royalty_rate=0.05,
            creator_eth_address="0x" + "a" * 40,
        ),
        "c2": _fake_record(
            "c2", royalty_rate=0.10,
            creator_eth_address="0x" + "b" * 40,
        ),
    }
    targets, _ = resolve_content_royalty_targets(
        contributing_shards=["c1", "c2"],
        content_index=_fake_index(records),
    )
    rates = {t.shard_cid: t.royalty_rate for t in targets}
    assert rates == {"c1": 0.05, "c2": 0.10}


def test_lookup_exception_treated_as_missing():
    """Defensive: ContentIndex.lookup raising mid-call shouldn't
    crash settlement. Treat as missing record."""
    idx = MagicMock()
    idx.lookup = MagicMock(side_effect=RuntimeError("db down"))
    targets, skipped = resolve_content_royalty_targets(
        contributing_shards=["c1"],
        content_index=idx,
    )
    assert targets == []
    assert skipped == 1
