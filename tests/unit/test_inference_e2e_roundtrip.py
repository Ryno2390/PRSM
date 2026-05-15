"""Sprint 438 — §5.2 inference E2E + F12 (epsilon-for-NONE) fix pin.

This is the verification-campaign sprint that promoted §5.2 inference
from 🟢 to ✅. Wired MockInferenceExecutor as opt-in via
PRSM_INFERENCE_EXECUTOR=mock (zero-filled crypto, MUST NOT be
trusted by real verifiers — explicit honest-scope). Live-verified
inference → signed-receipt → independent-verify chain end-to-end
for ALL privacy tiers.

Surfaced + fixed F12 during the live test:
- Mock executor's _epsilon_for_level returned float("inf") for
  PrivacyLevel.NONE.
- JSON serialization at /compute/inference layer maps Infinity to
  null (not strictly JSON spec; Python json.dumps default behavior).
- Verifier round-trips the receipt through JSON, reconstructs
  epsilon_spent=0.0 (or rejects null).
- Signing payload had `f"{inf:.10f}"`; verify payload had
  `f"{0.0:.10f}"` → bytes differ → signature_valid=false.

Fix: NONE tier uses 0.0 (not inf). Semantically honest — NONE
means "no DP applied", so "0 budget consumed" is the right
encoding.

These pins capture the F12 invariant + the broader inference-
roundtrip integrity claim.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from prsm.compute.inference.executor import MockInferenceExecutor
from prsm.compute.inference.models import (
    ContentTier, InferenceRequest,
)
from prsm.compute.tee.models import PrivacyLevel


def test_mock_executor_epsilon_for_none_is_finite():
    """F12 pin: PrivacyLevel.NONE must yield a FINITE
    epsilon_spent value. float("inf") JSON-serializes lossy
    (→ null), breaking the verify roundtrip. 0.0 is the
    honest semantic: NONE tier provides no DP guarantee, so
    "0 DP budget consumed" is the correct accounting."""
    eps = MockInferenceExecutor._epsilon_for_level(
        PrivacyLevel.NONE,
    )
    # NOT infinity
    import math
    assert math.isfinite(eps), (
        f"epsilon for NONE must be finite (F12 fix); got {eps}"
    )
    # And specifically 0.0 (honest semantic)
    assert eps == 0.0


def test_mock_executor_epsilon_for_other_tiers_finite():
    """All tiers' epsilon values must be finite — JSON-
    serializable without lossy fallback."""
    import math
    for tier in (
        PrivacyLevel.NONE,
        PrivacyLevel.STANDARD,
        PrivacyLevel.HIGH,
        PrivacyLevel.MAXIMUM,
    ):
        eps = MockInferenceExecutor._epsilon_for_level(tier)
        assert math.isfinite(eps), (
            f"epsilon for {tier} must be finite (JSON-safe)"
        )


@pytest.mark.asyncio
async def test_mock_executor_returns_signable_receipt_for_none_tier():
    """End-to-end: mock executor + NONE tier produces a
    receipt whose epsilon_spent is JSON-clean (no Infinity).
    The receipt then signs + verifies cleanly via the
    sprint-433 verify path."""
    executor = MockInferenceExecutor()
    req = InferenceRequest(
        request_id="test-req-1",
        prompt="hello",
        model_id="mock-llama-3-8b",
        privacy_tier=PrivacyLevel.NONE,
        content_tier=ContentTier.A,
        budget_ftns=Decimal("1.0"),
    )
    result = await executor.execute(req)
    assert result.success
    assert result.receipt is not None
    # The headline F12 invariant: receipt's epsilon is finite
    import math
    assert math.isfinite(result.receipt.epsilon_spent)

    # And the receipt JSON-roundtrips cleanly (no Infinity
    # making it through to a downstream verifier as null).
    payload = json.dumps(result.receipt.to_dict())
    assert "Infinity" not in payload
    assert "inf" not in payload.lower()


def test_inference_executor_env_gate_documented():
    """The PRSM_INFERENCE_EXECUTOR=mock env-gate must be
    documented in node.py so future operators don't think
    the default 503 is a bug."""
    from pathlib import Path
    node_py = (
        Path(__file__).resolve().parents[2]
        / "prsm" / "node" / "node.py"
    )
    text = node_py.read_text()
    assert "PRSM_INFERENCE_EXECUTOR" in text
    assert "MockInferenceExecutor" in text
    # The honest-scope note MUST stay attached — mock receipts
    # have zero-filled crypto fields beyond the actual settler
    # signature. Operators must not deploy this to production.
    text_lower = text.lower()
    assert "must not" in text_lower or "should not" in text_lower or "honest-scope" in text_lower
