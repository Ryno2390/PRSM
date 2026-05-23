"""Sprint 780 — settler-side credit-policy primitives.

Vision §4.5: "preemption should not result in operator slashing;
abandonment should". Sprint 779 closed the operator end of the
preemption arc (receipt now carries partial_completion). Sprint
780 ships the FIRST two settler-side primitives that future
settlement-flow integration (sprint 781+) will call:

  effective_cost_for_receipt(receipt) -> Decimal
      Cost to credit the operator. For receipts without
      partial_completion (full success): receipt.cost_ftns
      unchanged. For partial-completion receipts: scaled by
      tokens_completed / tokens_requested (proportional credit).

  should_slash_for_receipt(receipt) -> bool
      False when receipt has partial_completion + reason in
      {"preempted", "timeout"} (NOT operator's fault).
      True for reason="error" (abandonment — operator's fault).
      None partial_completion (full success): False (nothing to
      slash; cost-side already settled).
      Pure semantic policy — does not look at duration, network
      conditions, etc. Settlement-side composes this with its
      OTHER slashing triggers (Byzantine behavior, etc.).

Pure functions — no side effects, no I/O. Sprint 781 will plumb
both into the actual settlement flow.

Pin tests:
- effective_cost: full receipt → unchanged
- effective_cost: partial with 7/10 + cost=1.0 → 0.7
- effective_cost: partial with 0/10 → 0 (no credit; consumed nothing)
- effective_cost: tokens_requested=0 protection (divide-by-zero
  defense — returns 0)
- effective_cost: tokens_completed > tokens_requested (over-claim)
  → clamped at full cost (no bonus credit)
- should_slash: no partial → False
- should_slash: reason="preempted" → False
- should_slash: reason="timeout" → False
- should_slash: reason="error" → True
- should_slash: unknown reason → True (defensive; unknown is
  treated as operator-attributable until VALID_REASONS expands)
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


# ---- effective_cost_for_receipt ---------------------------------


def test_effective_cost_full_receipt_unchanged():
    """Receipt without partial_completion → cost unchanged."""
    from prsm.economy.credit_policy import (
        effective_cost_for_receipt,
    )
    r = _make_receipt(cost_ftns=Decimal("1.5"))
    assert effective_cost_for_receipt(r) == Decimal("1.5")


def test_effective_cost_proportional_seven_of_ten():
    """7/10 tokens + cost=1.0 → 0.7."""
    from prsm.economy.credit_policy import (
        effective_cost_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=7, requested=10),
    )
    assert effective_cost_for_receipt(r) == Decimal("0.7")


def test_effective_cost_zero_completed_zero_credit():
    """No tokens produced → no credit."""
    from prsm.economy.credit_policy import (
        effective_cost_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=0, requested=10),
    )
    assert effective_cost_for_receipt(r) == Decimal("0")


def test_effective_cost_zero_requested_returns_zero():
    """Divide-by-zero defense: tokens_requested=0 → 0 credit.
    (A receipt with requested=0 is malformed; settlement should
    NEVER end up paying out on it.)"""
    from prsm.economy.credit_policy import (
        effective_cost_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=5, requested=0),
    )
    assert effective_cost_for_receipt(r) == Decimal("0")


def test_effective_cost_over_claim_clamped_at_full():
    """tokens_completed > tokens_requested → clamp at full cost
    (no bonus credit). Defends against an operator inflating
    tokens_completed past tokens_requested to dodge the
    proportional scaling."""
    from prsm.economy.credit_policy import (
        effective_cost_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("1.0"),
        partial_completion=_make_partial(completed=20, requested=10),
    )
    assert effective_cost_for_receipt(r) == Decimal("1.0")


def test_effective_cost_precision_preserved():
    """Decimal arithmetic, not float. 33/100 of 3.0 should be
    exactly 0.99 (not 0.9900000000000001)."""
    from prsm.economy.credit_policy import (
        effective_cost_for_receipt,
    )
    r = _make_receipt(
        cost_ftns=Decimal("3.0"),
        partial_completion=_make_partial(completed=33, requested=100),
    )
    assert effective_cost_for_receipt(r) == Decimal("0.99")


# ---- should_slash_for_receipt -----------------------------------


def test_should_slash_no_partial_completion_false():
    """Full success → no slash (settlement's job to credit cost)."""
    from prsm.economy.credit_policy import (
        should_slash_for_receipt,
    )
    r = _make_receipt()
    assert should_slash_for_receipt(r) is False


def test_should_slash_preempted_false():
    """Vision §4.5: cloud preemption is NOT operator's fault."""
    from prsm.economy.credit_policy import (
        should_slash_for_receipt,
    )
    r = _make_receipt(
        partial_completion=_make_partial(reason="preempted"),
    )
    assert should_slash_for_receipt(r) is False


def test_should_slash_timeout_false():
    """Operator hit max-duration limit — credit what was done
    but don't slash (it's a configurable cap, not malice)."""
    from prsm.economy.credit_policy import (
        should_slash_for_receipt,
    )
    r = _make_receipt(
        partial_completion=_make_partial(reason="timeout"),
    )
    assert should_slash_for_receipt(r) is False


def test_should_slash_error_true():
    """Runtime fault on operator side (OOM, model crash) →
    slash. This is the 'abandonment' Vision §4.5 calls out."""
    from prsm.economy.credit_policy import (
        should_slash_for_receipt,
    )
    r = _make_receipt(
        partial_completion=_make_partial(reason="error"),
    )
    assert should_slash_for_receipt(r) is True


def test_should_slash_unknown_reason_true():
    """Defensive: unknown reason strings are treated as slash-
    worthy until VALID_REASONS expands. Prevents a malicious
    operator from picking a custom reason='cosmic-rays' to
    dodge the rule. Bias: prefer false-positive slashing over
    false-negative abandonment."""
    from prsm.economy.credit_policy import (
        should_slash_for_receipt,
    )
    r = _make_receipt(
        partial_completion=_make_partial(reason="cosmic-rays"),
    )
    assert should_slash_for_receipt(r) is True
