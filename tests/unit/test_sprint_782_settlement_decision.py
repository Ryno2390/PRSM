"""Sprint 782 — SettlementDecision composite primitive.

Sprint 780 shipped two pure-functional primitives:
  effective_cost_for_receipt(receipt) -> Decimal
  should_slash_for_receipt(receipt) -> bool

Settler-side integration code will typically need BOTH numbers
plus the implied "refund the remainder" figure (escrow_amount -
effective_cost). Sprint 782 ships the composite:

  compute_settlement_for_receipt(receipt, escrow_amount=None)
      -> SettlementDecision(
            release_to_operator: Decimal
            refund_to_payer: Decimal
            should_slash: bool
         )

Pure function — no I/O. The downstream escrow-flow code calls
this once + then invokes the appropriate PaymentEscrow APIs
(release / refund / slash) based on the decision.

escrow_amount default: receipt.cost_ftns (the "user paid full
cost" case where escrow == cost). When escrow exceeds the
priced cost (rare; promotional credits, dynamic pricing) the
caller passes it in so the refund-remainder math is correct.

Pin tests:
- SettlementDecision frozen dataclass shape.
- Full-success receipt → release=cost_ftns, refund=0, slash=False.
- 7/10 partial preempted → release=0.7*cost, refund=0.3*cost, slash=False.
- 0/10 partial preempted → release=0, refund=full, slash=False.
- "error" partial → release=proportional, refund=remainder, slash=True
  (settlement still credits work done; slash is a SEPARATE signal,
  not zero-credit).
- escrow_amount=2.0 + cost_ftns=1.0 + full success →
  release=1.0, refund=1.0 (over-funded escrow refunded down to cost).
- escrow_amount=2.0 + partial 7/10 → release=0.7, refund=1.3.
- Sum invariant: release + refund == escrow_amount (always).
"""
from __future__ import annotations

from decimal import Decimal


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


# ---- SettlementDecision dataclass ------------------------------


def test_settlement_decision_dataclass_shape():
    from prsm.economy.credit_policy import SettlementDecision
    d = SettlementDecision(
        release_to_operator=Decimal("0.7"),
        refund_to_payer=Decimal("0.3"),
        should_slash=False,
    )
    assert d.release_to_operator == Decimal("0.7")
    assert d.refund_to_payer == Decimal("0.3")
    assert d.should_slash is False


def test_settlement_decision_is_frozen():
    """Decision is a value — mutating it must raise."""
    from prsm.economy.credit_policy import SettlementDecision
    d = SettlementDecision(
        release_to_operator=Decimal("0"),
        refund_to_payer=Decimal("0"),
        should_slash=False,
    )
    import pytest
    with pytest.raises(Exception):
        d.release_to_operator = Decimal("1")  # type: ignore


# ---- compute_settlement_for_receipt -----------------------------


def test_full_success_releases_full_no_refund():
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(cost_ftns=Decimal("1.0"))
    d = compute_settlement_for_receipt(r)
    assert d.release_to_operator == Decimal("1.0")
    assert d.refund_to_payer == Decimal("0")
    assert d.should_slash is False


def test_partial_preempted_seven_of_ten():
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=7, requested=10),
    )
    d = compute_settlement_for_receipt(r)
    assert d.release_to_operator == Decimal("0.7")
    assert d.refund_to_payer == Decimal("0.3")
    assert d.should_slash is False


def test_partial_zero_completed_full_refund():
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=0, requested=10),
    )
    d = compute_settlement_for_receipt(r)
    assert d.release_to_operator == Decimal("0")
    assert d.refund_to_payer == Decimal("1.0")
    assert d.should_slash is False


def test_error_partial_still_credits_work_but_slashes():
    """Operator-attributable error (OOM, crash) still credits
    the proportional work done, but ALSO emits slash=True so
    settlement records the operator's reputation hit.
    Credit + slash are independent signals."""
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(
            reason="error", completed=4, requested=10,
        ),
    )
    d = compute_settlement_for_receipt(r)
    assert d.release_to_operator == Decimal("0.4")
    assert d.refund_to_payer == Decimal("0.6")
    assert d.should_slash is True


# ---- escrow_amount override --------------------------------------


def test_over_funded_escrow_refunds_excess_full_success():
    """Escrow > cost (over-funded by user) + full success →
    operator gets cost; payer gets refund of the excess."""
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(cost_ftns=Decimal("1.0"))
    d = compute_settlement_for_receipt(
        r, escrow_amount=Decimal("2.0"),
    )
    assert d.release_to_operator == Decimal("1.0")
    assert d.refund_to_payer == Decimal("1.0")


def test_over_funded_escrow_partial_completion():
    """Escrow=2.0, cost=1.0, completed 7/10 →
    release=0.7 (proportional of cost), refund=1.3 (rest)."""
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=7, requested=10),
    )
    d = compute_settlement_for_receipt(
        r, escrow_amount=Decimal("2.0"),
    )
    assert d.release_to_operator == Decimal("0.7")
    assert d.refund_to_payer == Decimal("1.3")


# ---- Sum invariant ----------------------------------------------


def test_sum_invariant_full_success():
    """release + refund always == escrow_amount."""
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(cost_ftns=Decimal("1.5"))
    d = compute_settlement_for_receipt(r)
    assert d.release_to_operator + d.refund_to_payer == Decimal("1.5")


def test_sum_invariant_partial():
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("3.0"),
        partial_completion=_make_partial(completed=33, requested=100),
    )
    d = compute_settlement_for_receipt(r)
    # 33/100 of 3.0 = 0.99; refund = 3.0 - 0.99 = 2.01
    assert d.release_to_operator == Decimal("0.99")
    assert d.refund_to_payer == Decimal("2.01")
    assert d.release_to_operator + d.refund_to_payer == Decimal("3.0")


def test_sum_invariant_over_funded():
    from prsm.economy.credit_policy import (
        compute_settlement_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=5, requested=10),
    )
    d = compute_settlement_for_receipt(
        r, escrow_amount=Decimal("5.0"),
    )
    # release=0.5 (5/10 of 1.0), refund=4.5 (5.0 - 0.5)
    assert d.release_to_operator == Decimal("0.5")
    assert d.refund_to_payer == Decimal("4.5")
    assert d.release_to_operator + d.refund_to_payer == Decimal("5.0")
