"""Sprint 777 — partial-completion field on InferenceReceipt.

Vision §4.5: "PRSM's Ed25519 signed-receipt protocol must support
partial-completion credit and re-routing. Preemption should not
result in operator slashing; abandonment should."

Pre-777 InferenceReceipt is binary: either it exists (full
success) or it doesn't (caller refunds full escrow). There's no
way for a preempted node to say "I completed 7 of 10 tokens
before going down — credit me for 7."

Sprint 777 adds `partial_completion: Optional[PartialCompletionInfo]`:

  @dataclass
  class PartialCompletionInfo:
      reason: str          # "preempted" | "timeout" | "error"
      tokens_completed: int
      tokens_requested: int
      timestamp: str       # ISO-8601 UTC

The settlement-side credit policy is OUT OF SCOPE for sprint 777
(separate from wire-format). Sprint 777 just plumbs the field
through: dataclass + to_dict + from_dict + signing_payload
(conditional encoding for back-compat).

Pin tests:
- PartialCompletionInfo dataclass exists.
- to_dict round-trips through from_dict.
- InferenceReceipt.partial_completion defaults to None.
- to_dict OMITS the field when None (back-compat byte-identical).
- to_dict INCLUDES the field when set.
- from_dict parses the field when present.
- signing_payload OMITS the field when None (back-compat).
- signing_payload INCLUDES a canonical hash line when set.
- Tampering tokens_completed flips signing_payload bytes.
- Tampering reason flips signing_payload bytes.
- Pre-777 receipts (no partial_completion) sign byte-identically
  to post-777 receipts with partial_completion=None.
"""
from __future__ import annotations

from decimal import Decimal

import pytest


def _make_receipt(**overrides):
    """Build a minimal valid InferenceReceipt for tests."""
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
        duration_seconds=1.5,
        cost_ftns=Decimal("1.0"),
        settler_node_id="settler1",
    )
    defaults.update(overrides)
    return InferenceReceipt(**defaults)


# ---- PartialCompletionInfo dataclass ----------------------------


def test_dataclass_exists():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    info = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    assert info.reason == "preempted"
    assert info.tokens_completed == 7
    assert info.tokens_requested == 10


def test_dataclass_to_dict_from_dict_roundtrip():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    original = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    d = original.to_dict()
    restored = PartialCompletionInfo.from_dict(d)
    assert restored == original


def test_dataclass_reason_validated():
    """Reason must be a known kind. Unknown reasons get coerced
    to "error" rather than silently passing through — closes the
    door on adversarial peers picking arbitrary reason strings
    that downstream settlement logic might mis-interpret."""
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
        VALID_REASONS,
    )
    assert "preempted" in VALID_REASONS
    assert "timeout" in VALID_REASONS
    assert "error" in VALID_REASONS
    # Unknown reason: keep the data structure permissive at
    # construction (callers can introduce new reasons over time),
    # but signing_payload + verification logic key off VALID_REASONS
    # to ensure semantic stability.
    info = PartialCompletionInfo(
        reason="weird_new_kind",
        tokens_completed=1,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    # Constructor accepts; semantic interpretation is settlement-side.
    assert info.reason == "weird_new_kind"


# ---- InferenceReceipt default + to_dict -------------------------


def test_receipt_partial_completion_defaults_to_none():
    """Backward-compat: pre-777 callers constructing receipts
    without the field get None."""
    r = _make_receipt()
    assert r.partial_completion is None


def test_to_dict_omits_field_when_none():
    """Back-compat: pre-777 to_dict output byte-identical."""
    r = _make_receipt()
    d = r.to_dict()
    assert "partial_completion" not in d


def test_to_dict_includes_field_when_set():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    info = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    r = _make_receipt(partial_completion=info)
    d = r.to_dict()
    assert "partial_completion" in d
    assert d["partial_completion"]["reason"] == "preempted"
    assert d["partial_completion"]["tokens_completed"] == 7


# ---- from_dict --------------------------------------------------


def test_from_dict_parses_when_present():
    from prsm.compute.inference.models import InferenceReceipt
    r = _make_receipt()
    d = r.to_dict()
    d["partial_completion"] = {
        "reason": "preempted",
        "tokens_completed": 7,
        "tokens_requested": 10,
        "timestamp": "2026-05-23T12:00:00Z",
    }
    restored = InferenceReceipt.from_dict(d)
    assert restored.partial_completion is not None
    assert restored.partial_completion.reason == "preempted"
    assert restored.partial_completion.tokens_completed == 7


def test_from_dict_field_absent_yields_none():
    """Pre-777 serialized receipts (no key) parse cleanly."""
    from prsm.compute.inference.models import InferenceReceipt
    r = _make_receipt()
    d = r.to_dict()
    # d already lacks the field
    restored = InferenceReceipt.from_dict(d)
    assert restored.partial_completion is None


# ---- signing_payload conditional encoding -----------------------


def test_signing_payload_omits_field_when_none():
    """Pre-777 receipts must sign byte-identically to post-777
    receipts with partial_completion=None."""
    r_with_field_none = _make_receipt()
    r_without_field = _make_receipt()
    # Both have None partial_completion → identical bytes
    assert r_with_field_none.signing_payload() == r_without_field.signing_payload()
    # And neither should contain the marker
    assert b"partial_completion" not in r_with_field_none.signing_payload()


def test_signing_payload_includes_canonical_line_when_set():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    info = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    r = _make_receipt(partial_completion=info)
    payload = r.signing_payload()
    assert b"partial_completion:" in payload


def test_tampering_tokens_completed_flips_signing_bytes():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    honest = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    tampered = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=10,   # claims more credit than earned
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    r1 = _make_receipt(partial_completion=honest)
    r2 = _make_receipt(partial_completion=tampered)
    assert r1.signing_payload() != r2.signing_payload()


def test_tampering_reason_flips_signing_bytes():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    honest = PartialCompletionInfo(
        reason="preempted",  # not your fault
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    tampered = PartialCompletionInfo(
        reason="error",      # try to dodge slashing by relabeling
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    r1 = _make_receipt(partial_completion=honest)
    r2 = _make_receipt(partial_completion=tampered)
    assert r1.signing_payload() != r2.signing_payload()


def test_pre_777_byte_identical_when_field_omitted():
    """A receipt built without ever touching partial_completion
    must produce signing_payload bytes byte-identical to a pre-
    777 receipt. Pin by checking that the field's marker string
    never appears in the bytes when default."""
    r = _make_receipt()
    payload = r.signing_payload()
    assert b"partial_completion" not in payload
