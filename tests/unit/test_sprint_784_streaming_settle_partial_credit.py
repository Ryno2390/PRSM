"""Sprint 784 — wire settle_inference_receipt into _settle_streaming_escrow.

This is the FIRST production call site to integrate the
sprints 780-783 settler-side primitives. _settle_streaming_escrow
runs at the end of every successful streaming inference and has
BOTH `result.receipt` (an InferenceReceipt) AND the escrow in
scope — the perfect place to plumb partial-completion credit.

Behavior matrix:
- result.receipt is None (legacy / mid-stream failure path)
  → fall back to release_escrow (existing behavior; no regression)
- result.receipt present with no partial_completion (full success)
  → settle_inference_receipt → release_escrow (full payout)
- result.receipt with partial_completion 7/10 preempted
  → settle_inference_receipt → release_escrow_split with 0.7 of
  the escrow amount; PaymentEscrow auto-refunds the 0.3 remainder
- result.receipt with partial_completion + reason="error"
  → settle_inference_receipt branches on amounts AND should_slash;
  slash is LOGGED (slash-emission hook is a later sprint)

Pin tests:
- Successful streaming (no partial_completion) still calls
  release_escrow (no behavior change for the existing path).
- Partial-completion result triggers release_escrow_split.
- Zero-completion result triggers refund_escrow.
- Missing receipt falls back to release_escrow (legacy guard).
- Slash decision on error is logged (operator-visible signal).
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import asyncio


def _make_receipt(**overrides):
    from prsm.compute.inference.models import (
        InferenceReceipt,
        ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    defaults = dict(
        job_id="j1",
        request_id="r1",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att",
        output_hash=b"\xde\xad",
        duration_seconds=1.0,
        cost_ftns=Decimal("1.0"),
        settler_node_id="s1",
    )
    defaults.update(overrides)
    return InferenceReceipt(**defaults)


def _make_partial(reason="preempted", completed=7, requested=10):
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    return PartialCompletionInfo(
        reason=reason,
        tokens_completed=completed,
        tokens_requested=requested,
        timestamp="2026-05-23T12:00:00Z",
    )


def _make_node():
    node = MagicMock()
    node._payment_escrow = MagicMock()
    node._payment_escrow.release_escrow = AsyncMock(return_value=MagicMock())
    node._payment_escrow.release_escrow_split = AsyncMock(return_value=[MagicMock()])
    node._payment_escrow.refund_escrow = AsyncMock(return_value=True)
    node.identity = MagicMock()
    node.identity.node_id = "node1"
    node.privacy_budget = None  # skip the privacy-budget branch
    return node


def _make_request():
    req = MagicMock()
    # NONE privacy tier skips the privacy-budget tracking branch
    req.privacy_tier = MagicMock()
    req.privacy_tier.value = "none"
    req.model_id = "gpt2"
    return req


def _make_result(receipt):
    res = MagicMock()
    res.receipt = receipt
    return res


def _call_settle(node, result, escrow_amount=Decimal("1.0")):
    from prsm.node.api import _settle_streaming_escrow
    escrow_entry = MagicMock()
    escrow_entry.amount = float(escrow_amount)
    return asyncio.run(_settle_streaming_escrow(
        node=node, job_id="j1",
        escrow_entry=escrow_entry, request=_make_request(),
        result=result,
    ))


# ---- Behavior matrix --------------------------------------------


def test_full_success_calls_release_escrow():
    """No partial_completion → existing release_escrow path."""
    node = _make_node()
    result = _make_result(_make_receipt())
    _call_settle(node, result)
    node._payment_escrow.release_escrow.assert_awaited_once()
    node._payment_escrow.release_escrow_split.assert_not_called()
    node._payment_escrow.refund_escrow.assert_not_called()


def test_partial_completion_calls_release_escrow_split():
    """7/10 preempted → split path with 70% of escrow."""
    node = _make_node()
    result = _make_result(_make_receipt(
        partial_completion=_make_partial(completed=7, requested=10),
    ))
    _call_settle(node, result)
    node._payment_escrow.release_escrow_split.assert_awaited_once()
    node._payment_escrow.release_escrow.assert_not_called()
    # Verify the split amount
    call = node._payment_escrow.release_escrow_split.call_args
    splits = call.args[1] if len(call.args) > 1 else call.kwargs.get("splits")
    assert len(splits) == 1
    recipient, amount = splits[0]
    assert recipient == "node1"
    assert Decimal(str(amount)) == Decimal("0.7")


def test_zero_completion_calls_refund_escrow():
    """0/10 → full refund (operator did no work)."""
    node = _make_node()
    result = _make_result(_make_receipt(
        partial_completion=_make_partial(completed=0, requested=10),
    ))
    _call_settle(node, result)
    node._payment_escrow.refund_escrow.assert_awaited_once()
    node._payment_escrow.release_escrow.assert_not_called()
    node._payment_escrow.release_escrow_split.assert_not_called()


def test_missing_receipt_falls_back_to_release_escrow():
    """No receipt (e.g. mid-stream failure handed back a result
    sans receipt) → existing release_escrow path; no regression."""
    node = _make_node()
    result = _make_result(receipt=None)
    _call_settle(node, result)
    node._payment_escrow.release_escrow.assert_awaited_once()
    node._payment_escrow.release_escrow_split.assert_not_called()
    node._payment_escrow.refund_escrow.assert_not_called()


def test_slash_decision_on_error_logged(caplog):
    """reason="error" → settle_inference_receipt runs (proportional
    credit applied) AND a slash signal is logged. Slash-emission
    hook itself is deferred to a later sprint; sprint 784 just
    surfaces the signal so operators see it in journalctl."""
    import logging
    caplog.set_level(logging.WARNING, logger="prsm.node.api")
    node = _make_node()
    result = _make_result(_make_receipt(
        partial_completion=_make_partial(
            reason="error", completed=4, requested=10,
        ),
    ))
    _call_settle(node, result)
    # Slash signal in logs
    assert any(
        "slash" in r.message.lower()
        for r in caplog.records
    )
