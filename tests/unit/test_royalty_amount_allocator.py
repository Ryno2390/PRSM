"""Sprint 256 — royalty amount allocator (uniform | rate_weighted).

Sprint 248 wired the on-chain content-royalty leg with uniform
per-shard allocation (every shard gets the same `gross_per_
shard_wei` from env). A smarter policy weights by each shard's
recorded `royalty_rate` — content with a higher per-access rate
pays out proportionally more from a fixed pool.

This sprint ships a pure allocator function. The dispatcher
integration (sprint TBD) consumes the returned `{cid -> wei}`
map and lets the operator pick mode via env.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prsm.economy.onchain_content_royalty import (
    allocate_royalty_amounts,
)


def _rec(cid: str, royalty_rate: float):
    r = MagicMock()
    r.cid = cid
    r.royalty_rate = royalty_rate
    return r


def _index(records):
    idx = MagicMock()
    idx.lookup = lambda cid: records.get(cid)
    return idx


class TestUniform:
    def test_evenly_split(self):
        out = allocate_royalty_amounts(
            shards=["a", "b", "c", "d"],
            content_index=_index({}),  # ignored in uniform
            total_pool_wei=1000,
            mode="uniform",
        )
        # 1000 // 4 = 250 each, last absorbs remainder
        assert sum(out.values()) == 1000
        assert all(v == 250 for v in out.values())

    def test_uniform_remainder_absorbed_by_last(self):
        out = allocate_royalty_amounts(
            shards=["a", "b", "c"],
            content_index=_index({}),
            total_pool_wei=1000,
            mode="uniform",
        )
        # 1000 // 3 = 333; remainder 1 goes to last
        assert sum(out.values()) == 1000
        assert out["a"] == 333
        assert out["b"] == 333
        assert out["c"] == 334


class TestRateWeighted:
    def test_proportional_split(self):
        records = {
            "a": _rec("a", 0.05),
            "b": _rec("b", 0.10),
            "c": _rec("c", 0.05),
        }
        out = allocate_royalty_amounts(
            shards=["a", "b", "c"],
            content_index=_index(records),
            total_pool_wei=1000,
            mode="rate_weighted",
        )
        # weights: 0.05+0.10+0.05 = 0.20
        # a: 1000 * 0.05/0.20 = 250
        # b: 1000 * 0.10/0.20 = 500
        # c: 1000 * 0.05/0.20 = 250
        assert sum(out.values()) == 1000
        assert out["a"] == 250
        assert out["b"] == 500
        assert out["c"] == 250

    def test_missing_record_falls_back_to_zero_then_redistributes(self):
        """Missing record = no weight; remaining shards split
        the pool proportionally."""
        records = {
            "a": _rec("a", 0.05),
            "b": _rec("b", 0.05),
            # c missing
        }
        out = allocate_royalty_amounts(
            shards=["a", "b", "c"],
            content_index=_index(records),
            total_pool_wei=1000,
            mode="rate_weighted",
        )
        # c has zero weight → 0
        # a + b split 1000 evenly
        assert out["c"] == 0
        assert sum(out.values()) == 1000

    def test_all_zero_rates_falls_back_to_uniform(self):
        """If all weights are 0 (or no records), degrade
        gracefully to uniform allocation."""
        records = {
            "a": _rec("a", 0.0),
            "b": _rec("b", 0.0),
        }
        out = allocate_royalty_amounts(
            shards=["a", "b"],
            content_index=_index(records),
            total_pool_wei=1000,
            mode="rate_weighted",
        )
        assert sum(out.values()) == 1000
        assert out["a"] == 500
        assert out["b"] == 500


class TestValidation:
    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            allocate_royalty_amounts(
                shards=["a"],
                content_index=_index({}),
                total_pool_wei=100,
                mode="bogus",
            )

    def test_zero_pool_rejected(self):
        with pytest.raises(ValueError):
            allocate_royalty_amounts(
                shards=["a"],
                content_index=_index({}),
                total_pool_wei=0,
                mode="uniform",
            )

    def test_negative_pool_rejected(self):
        with pytest.raises(ValueError):
            allocate_royalty_amounts(
                shards=["a"],
                content_index=_index({}),
                total_pool_wei=-5,
                mode="uniform",
            )


class TestEdge:
    def test_empty_shards_returns_empty(self):
        out = allocate_royalty_amounts(
            shards=[],
            content_index=_index({}),
            total_pool_wei=1000,
            mode="uniform",
        )
        assert out == {}

    def test_single_shard_gets_full_pool(self):
        out = allocate_royalty_amounts(
            shards=["a"],
            content_index=_index({}),
            total_pool_wei=1000,
            mode="uniform",
        )
        assert out == {"a": 1000}

    def test_lookup_exception_treated_as_zero_weight(self):
        idx = MagicMock()
        idx.lookup = MagicMock(side_effect=RuntimeError("db"))
        out = allocate_royalty_amounts(
            shards=["a", "b"],
            content_index=idx,
            total_pool_wei=1000,
            mode="rate_weighted",
        )
        # Both lookup-fail → all zero weight → uniform fallback
        assert out["a"] == 500
        assert out["b"] == 500
