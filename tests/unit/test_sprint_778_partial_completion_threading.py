"""Sprint 778 — thread partial_completion from ChainExecutionResult
into the signed InferenceReceipt.

Sprint 777 added the wire-format field. Sprint 778 closes the
plumbing gap: `_build_signed_receipt` reads the optional
`partial_completion` attribute off the chain executor's outcome
+ propagates it into the receipt.

This is the same pure-additive pattern sprint 413 used for
activation_noise_trace + topology_assignment:
- ChainExecutionResult gets a new Optional field (default None)
- Scheduler reads via getattr() (defensive against legacy
  ChainExecutor implementers that don't have the field)
- When None, signing-payload bytes byte-identical to pre-777
  (sprint 297 conditional encoding handles this)

Future sprint 779+ will ship the actual PRODUCER: an inference
runner hook that consults is_currently_preempted() during
streaming + populates outcome.partial_completion when the
detector flag flips mid-stream. Sprint 778 ships the carrier.

Pin tests:
- ChainExecutionResult has partial_completion field (default None).
- _build_signed_receipt threads outcome.partial_completion into
  the receipt.
- Legacy outcome WITHOUT the attribute (Protocol stand-in) gets
  None via getattr default — no AttributeError.
- Signing-payload byte-identical when outcome.partial_completion
  is None (back-compat invariant pinned across receipt path).
- When outcome.partial_completion is set, the receipt's
  partial_completion ROUND-TRIPS through to_dict/from_dict
  preserving the tamper-evident hash binding.
"""
from __future__ import annotations

import dataclasses
from decimal import Decimal
from unittest.mock import MagicMock


def _make_outcome(**overrides):
    """Build a minimal ChainExecutionResult."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    defaults = dict(
        output="hello",
        duration_seconds=1.0,
        tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )
    defaults.update(overrides)
    return ChainExecutionResult(**defaults)


def _make_request():
    """Build a minimal InferenceRequest."""
    from prsm.compute.inference.models import (
        InferenceRequest,
        ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    return InferenceRequest(
        request_id="r1",
        prompt="The capital of France is",
        model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        budget_ftns=Decimal("1.0"),
        max_tokens=10,
    )


# ---- ChainExecutionResult field ---------------------------------


def test_chain_execution_result_has_partial_completion_field():
    """Schema check — sprint 778 adds the field with default None."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    field_names = {f.name for f in dataclasses.fields(ChainExecutionResult)}
    assert "partial_completion" in field_names

    outcome = _make_outcome()
    assert outcome.partial_completion is None


# ---- _build_signed_receipt threading ----------------------------


def _make_executor_with_identity():
    """Build a minimal ParallaxScheduledExecutor harness with an
    Ed25519 identity for signing."""
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("test-node")
    exec_ = ParallaxScheduledExecutor.__new__(ParallaxScheduledExecutor)
    exec_._identity = identity
    return exec_


def test_build_signed_receipt_threads_partial_completion_when_set():
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    info = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    exec_ = _make_executor_with_identity()
    outcome = _make_outcome(partial_completion=info)
    req = _make_request()

    receipt = exec_._build_signed_receipt(
        request=req,
        cost=Decimal("0.5"),
        outcome=outcome,
        streamed=True,
    )
    assert receipt.partial_completion is not None
    assert receipt.partial_completion.reason == "preempted"
    assert receipt.partial_completion.tokens_completed == 7
    assert receipt.settler_signature != b""


def test_build_signed_receipt_partial_completion_none_when_outcome_default():
    """Pre-778 outcomes (no partial_completion attribute set) →
    receipt.partial_completion is None → signing-payload
    byte-identical to pre-778."""
    exec_ = _make_executor_with_identity()
    outcome = _make_outcome()  # default None
    req = _make_request()

    receipt = exec_._build_signed_receipt(
        request=req,
        cost=Decimal("0.5"),
        outcome=outcome,
        streamed=False,
    )
    assert receipt.partial_completion is None
    # Pre-777-byte-equivalence: marker doesn't appear in signing bytes
    assert b"partial_completion" not in receipt.signing_payload()


def test_build_signed_receipt_handles_legacy_outcome_missing_attr():
    """Defensive: a legacy ChainExecutor that returns a Protocol
    stand-in without the new field MUST NOT raise AttributeError.
    Pattern: getattr(outcome, 'partial_completion', None)."""
    from prsm.compute.tee.models import TEEType
    exec_ = _make_executor_with_identity()
    # Stand-in object that does NOT define partial_completion
    legacy = MagicMock(spec=[
        "output", "duration_seconds", "tee_attestation",
        "tee_type", "epsilon_spent",
        "activation_noise_trace", "topology_assignment",
    ])
    legacy.output = "hello"
    legacy.duration_seconds = 1.0
    legacy.tee_attestation = b"att"
    legacy.tee_type = TEEType.SOFTWARE
    legacy.epsilon_spent = 0.0
    legacy.activation_noise_trace = None
    legacy.topology_assignment = None
    req = _make_request()

    receipt = exec_._build_signed_receipt(
        request=req,
        cost=Decimal("0.5"),
        outcome=legacy,
        streamed=False,
    )
    assert receipt.partial_completion is None


# ---- Round-trip preserves binding -------------------------------


def test_signed_receipt_round_trip_preserves_partial_completion():
    """Signed receipt → to_dict → from_dict → still verifies."""
    from prsm.compute.inference.models import InferenceReceipt
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    from prsm.compute.inference.receipt import verify_receipt
    info = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=7,
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    exec_ = _make_executor_with_identity()
    outcome = _make_outcome(partial_completion=info)
    req = _make_request()
    signed = exec_._build_signed_receipt(
        request=req, cost=Decimal("0.5"),
        outcome=outcome, streamed=True,
    )

    payload_dict = signed.to_dict()
    restored = InferenceReceipt.from_dict(payload_dict)
    # Verifier reads the canonical bytes — back-compat to sprint
    # 706/707's external receipt-verify path.
    assert verify_receipt(restored, identity=exec_._identity)
    # Tampering tokens_completed should now fail verification.
    tampered_info = PartialCompletionInfo(
        reason="preempted",
        tokens_completed=10,   # claim more than honest
        tokens_requested=10,
        timestamp="2026-05-23T12:00:00Z",
    )
    tampered = dataclasses.replace(
        restored, partial_completion=tampered_info,
    )
    assert not verify_receipt(tampered, identity=exec_._identity)
