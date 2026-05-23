"""Sprint 791 — aggregate earnings per node_id (multi-device roster UX).

The multi-device arc (sprints 786-790) lets operators bind
multiple devices to one wallet. The closing piece is a way to
SEE per-device earnings — answer the question "which of my
devices is actually earning?".

Sprint 791 ships the pure aggregator primitive. A daemon endpoint
+ `prsm wallet devices earnings` CLI consumer would land in a
future sprint; sprint 791 is the load-bearing pure-functional
piece that any caller can drop in.

  aggregate_earnings_by_node_id(receipts) -> Dict[str, Decimal]

Inputs:
  - Receipts as an iterable of dicts (persisted shape) OR
    InferenceReceipt dataclasses (round-tripped).
  - Each receipt MUST have settler_node_id + cost_ftns +
    (optional) partial_completion. Malformed receipts are
    skipped (defensive — don't crash an earnings summary on
    one bad row).

Output:
  Dict mapping settler_node_id → total effective_cost_for_receipt
  summed across all matching receipts. Uses sprint-780
  effective_cost_for_receipt so partial_completion is honored
  (preempted partials credit proportionally; full successes
  credit fully).

Pin tests:
- Empty input → {}
- Single full-success receipt → {node_id: cost}
- Multiple receipts same node → sum
- Multiple nodes → separate entries
- Partial_completion 7/10 → 0.7 * cost credited
- Mixed full + partial → correct sum (sprint-780 math)
- Receipt missing settler_node_id → skipped, no crash
- Receipt missing cost_ftns → skipped, no crash
- Dict input round-trips through InferenceReceipt.from_dict
- Dataclass input handled directly (no double-parse)
- Decimal arithmetic preserved (no float drift)
"""
from __future__ import annotations

from decimal import Decimal


def _make_receipt_dict(
    settler_node_id="node-a",
    cost_ftns="1.0",
    partial_completion=None,
):
    """Build a persisted-shape receipt dict (matches
    InferenceReceipt.to_dict())."""
    d = {
        "job_id": "j1",
        "request_id": "r1",
        "model_id": "gpt2",
        "content_tier": "A",
        "privacy_tier": "none",
        "epsilon_spent": 0.0,
        "tee_type": "software",
        "tee_attestation": "6174",  # hex of b"at"
        "output_hash": "dead",
        "duration_seconds": 1.0,
        "cost_ftns": cost_ftns,
        "settler_signature": "",
        "settler_node_id": settler_node_id,
        "streamed_output": False,
    }
    if partial_completion is not None:
        d["partial_completion"] = partial_completion
    return d


# ---- Basic empties / singles -----------------------------------


def test_empty_input_returns_empty_dict():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    assert aggregate_earnings_by_node_id([]) == {}


def test_single_full_success_receipt():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="1.5"),
    ])
    assert out == {"node-a": Decimal("1.5")}


# ---- Aggregation across receipts -------------------------------


def test_multiple_receipts_same_node_summed():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="1.0"),
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="2.0"),
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="0.5"),
    ])
    assert out == {"node-a": Decimal("3.5")}


def test_multiple_nodes_separate_entries():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="1.0"),
        _make_receipt_dict(settler_node_id="node-b", cost_ftns="2.0"),
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="0.5"),
    ])
    assert out == {
        "node-a": Decimal("1.5"),
        "node-b": Decimal("2.0"),
    }


# ---- Partial-completion honored --------------------------------


def test_partial_completion_credited_proportionally():
    """7/10 preempted → 0.7 * cost credited (sprint-780 math)."""
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        _make_receipt_dict(
            settler_node_id="node-a",
            cost_ftns="1.0",
            partial_completion={
                "reason": "preempted",
                "tokens_completed": 7,
                "tokens_requested": 10,
                "timestamp": "2026-05-23T12:00:00Z",
            },
        ),
    ])
    assert out == {"node-a": Decimal("0.7")}


def test_mixed_full_and_partial_summed_correctly():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        # Full 1.0
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="1.0"),
        # Partial 7/10 of 1.0 = 0.7
        _make_receipt_dict(
            settler_node_id="node-a",
            cost_ftns="1.0",
            partial_completion={
                "reason": "preempted",
                "tokens_completed": 7,
                "tokens_requested": 10,
                "timestamp": "2026-05-23T12:00:00Z",
            },
        ),
    ])
    assert out == {"node-a": Decimal("1.7")}


# ---- Defensive: malformed receipts ----------------------------


def test_missing_settler_node_id_skipped():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    good = _make_receipt_dict(
        settler_node_id="node-a", cost_ftns="1.0",
    )
    bad = _make_receipt_dict(cost_ftns="2.0")
    bad.pop("settler_node_id")
    out = aggregate_earnings_by_node_id([good, bad])
    # Only the good receipt counted; bad skipped without crash
    assert out == {"node-a": Decimal("1.0")}


def test_missing_cost_ftns_skipped():
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    good = _make_receipt_dict(
        settler_node_id="node-a", cost_ftns="1.0",
    )
    bad = _make_receipt_dict(settler_node_id="node-b")
    bad.pop("cost_ftns")
    out = aggregate_earnings_by_node_id([good, bad])
    assert out == {"node-a": Decimal("1.0")}


def test_empty_settler_node_id_skipped():
    """settler_node_id="" (default before signing) — skip rather
    than create a "" key with random sum."""
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        _make_receipt_dict(settler_node_id="", cost_ftns="1.0"),
        _make_receipt_dict(settler_node_id="node-a", cost_ftns="2.0"),
    ])
    assert out == {"node-a": Decimal("2.0")}


# ---- Dataclass input path -------------------------------------


def test_inference_receipt_dataclass_input():
    """Accepts InferenceReceipt instances directly, no re-parse."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    receipt = InferenceReceipt.from_dict(
        _make_receipt_dict(
            settler_node_id="node-x", cost_ftns="2.5",
        ),
    )
    out = aggregate_earnings_by_node_id([receipt])
    assert out == {"node-x": Decimal("2.5")}


def test_mixed_dict_and_dataclass_inputs():
    """Iterable of mixed dict + dataclass — both handled."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    dataclass_input = InferenceReceipt.from_dict(
        _make_receipt_dict(
            settler_node_id="node-a", cost_ftns="1.0",
        ),
    )
    dict_input = _make_receipt_dict(
        settler_node_id="node-a", cost_ftns="2.0",
    )
    out = aggregate_earnings_by_node_id([dataclass_input, dict_input])
    assert out == {"node-a": Decimal("3.0")}


# ---- Decimal precision ----------------------------------------


def test_decimal_precision_preserved():
    """33/100 of 3.0 = exactly 0.99 (no float drift)."""
    from prsm.economy.credit_policy import (
        aggregate_earnings_by_node_id,
    )
    out = aggregate_earnings_by_node_id([
        _make_receipt_dict(
            settler_node_id="node-a",
            cost_ftns="3.0",
            partial_completion={
                "reason": "preempted",
                "tokens_completed": 33,
                "tokens_requested": 100,
                "timestamp": "2026-05-23T12:00:00Z",
            },
        ),
    ])
    assert out == {"node-a": Decimal("0.99")}
