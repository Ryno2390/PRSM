"""Sprint 780 — settler-side credit-policy primitives.

Vision §4.5: "preemption should not result in operator slashing;
abandonment should". Sprints 772-779 closed the OPERATOR end of
the preemption arc (cloud-spot detection through to signed
partial-completion receipts). This module ships the FIRST two
SETTLER-side primitives that future settlement-flow integration
(sprint 781+) will call.

Two pure functions:

    effective_cost_for_receipt(receipt) -> Decimal
        Cost to credit the operator. For full-success receipts
        (no partial_completion): receipt.cost_ftns unchanged.
        For partial-completion receipts: scaled by
        tokens_completed / tokens_requested. Decimal arithmetic
        preserved (no float-precision drift).

    should_slash_for_receipt(receipt) -> bool
        Pure-semantic slash policy keyed on partial_completion.reason:
        - None partial_completion             → False (success path)
        - reason in {preempted, timeout}      → False (not operator's fault)
        - reason="error"                      → True (abandonment)
        - any other reason                    → True (defensive default;
          prevents a malicious operator from inventing a custom
          reason to dodge the rule)

Both functions are pure: no I/O, no side effects. Settlement-side
composes these with its OTHER slash triggers (Byzantine behavior,
attestation failures, etc.) — sprint 780 only ships the partial-
completion-aware rule.

Over-claim defense: if a tampered receipt has
tokens_completed > tokens_requested, effective_cost_for_receipt
clamps at full cost (no bonus credit). The signing-payload's
canonical hash (sprint 777) makes such tampering signature-
invalid before this code ever sees the receipt — this is belt-
and-suspenders.
"""
from __future__ import annotations

from decimal import Decimal

from prsm.compute.inference.models import InferenceReceipt


# Reasons that are NOT operator-attributable (no slashing).
# Mirror Vision §4.5: cloud preemption + configurable timeout
# are not the operator's fault; runtime errors (OOM, crash) are.
_NO_SLASH_REASONS = frozenset({"preempted", "timeout"})


def effective_cost_for_receipt(receipt: InferenceReceipt) -> Decimal:
    """Cost amount to credit the operator for this receipt.

    Returns receipt.cost_ftns unchanged when there's no partial-
    completion marker. Otherwise scales proportionally by
    tokens_completed / tokens_requested, clamped to [0, full]."""
    info = getattr(receipt, "partial_completion", None)
    if info is None:
        return receipt.cost_ftns

    requested = int(getattr(info, "tokens_requested", 0))
    completed = int(getattr(info, "tokens_completed", 0))

    # Divide-by-zero defense: a malformed receipt with requested=0
    # MUST NOT pay out anything.
    if requested <= 0:
        return Decimal("0")

    # Clamp at full cost — no over-credit even if the operator
    # somehow signed a receipt claiming more completed than
    # requested (signature would normally invalidate this, but
    # defense-in-depth).
    if completed >= requested:
        return receipt.cost_ftns

    if completed <= 0:
        return Decimal("0")

    return (
        receipt.cost_ftns * Decimal(completed) / Decimal(requested)
    )


def should_slash_for_receipt(receipt: InferenceReceipt) -> bool:
    """Whether the partial-completion marker on this receipt
    should trigger a slash.

    Returns False (no slash) when:
    - receipt has no partial_completion (full-success path)
    - reason is "preempted" or "timeout" (not operator's fault)

    Returns True (slash) when:
    - reason is "error" (runtime fault — abandonment per §4.5)
    - reason is anything else (defensive default; an unknown
      reason cannot be allowed to escape the slash policy by
      virtue of being unknown)

    Settlement-side composes this with its OTHER slash triggers."""
    info = getattr(receipt, "partial_completion", None)
    if info is None:
        return False
    reason = getattr(info, "reason", "")
    return reason not in _NO_SLASH_REASONS
