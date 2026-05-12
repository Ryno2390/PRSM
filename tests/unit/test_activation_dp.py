"""Sprint 295 — activation-layer DP noise primitive.

Vision §7 honest-limits: "Activation-inversion attacks (Zhu
et al. 2019 and follow-up literature) can partially
reconstruct input prompts from early-layer activations.
Mitigations include topology rotation per inference and
activation-layer TEE attestation, both of which are in the
Phase 2+ roadmap."

Current DP noise (executor.py:497-508) runs ONLY on the
aggregated output. Per-stage activations flow between SPRKs
without noise injection, vulnerable to inversion attacks
that recover the input prompt from intermediate activations.

This sprint ships the primitive that the streaming-inference
subsystem (Phase 3.x.7+ RpcChainExecutor) will use to inject
noise per-stage:

  ActivationDPInjector — per-stage Gaussian noise injector
    that tracks cumulative ε spend across multiple stages
    under basic composition (DP total = sum of per-stage ε).

  StageNoisePolicy — pure function mapping (tier, stage_count)
    → per-stage ε. Conservative basic-composition: per-stage
    ε = tier.ε / stage_count, so the sum across all stages
    matches the tier's claimed total ε.

  ActivationNoiseTrace — per-stage ε record for inclusion in
    InferenceReceipt; lets verifiers confirm activation noise
    was actually applied as promised.

  verify_activation_noise_trace — verifier predicate; checks
    that the sum of per-stage ε matches the tier's expected
    total within a tolerance, and that all expected stages
    are present.

Sprint 296 will wire this into RpcChainExecutor;
sprint 297 will surface activation_noise_trace as a receipt
field that verify_receipt_privacy_claim consumes.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from prsm.compute.inference.activation_dp import (
    ActivationDPInjector,
    ActivationNoiseTrace,
    StageNoisePolicy,
    verify_activation_noise_trace,
)
from prsm.compute.tee.models import PrivacyLevel


# ── StageNoisePolicy: per-stage ε allocation ─────────────


def test_stage_policy_basic_composition_standard():
    """STANDARD tier ε=8.0 split across 4 stages → ε=2.0 per
    stage (basic composition: total = sum)."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.STANDARD, stage_count=4,
    )
    assert policy.per_stage_epsilon == pytest.approx(2.0)
    assert policy.total_expected_epsilon == pytest.approx(8.0)


def test_stage_policy_high_tier():
    """HIGH tier ε=4.0 split across 4 stages → ε=1.0 each."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.HIGH, stage_count=4,
    )
    assert policy.per_stage_epsilon == pytest.approx(1.0)
    assert policy.total_expected_epsilon == pytest.approx(4.0)


def test_stage_policy_maximum_tier():
    """MAXIMUM tier ε=1.0 split across 4 stages → ε=0.25
    each. Tighter privacy = more noise per stage."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.MAXIMUM, stage_count=4,
    )
    assert policy.per_stage_epsilon == pytest.approx(0.25)
    assert policy.total_expected_epsilon == pytest.approx(1.0)


def test_stage_policy_none_tier_disabled():
    """PrivacyLevel.NONE → per-stage noise disabled (ε=inf
    meaning "no clipping, no noise")."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.NONE, stage_count=4,
    )
    assert math.isinf(policy.per_stage_epsilon)
    assert policy.enabled is False


def test_stage_policy_rejects_zero_stages():
    with pytest.raises(ValueError):
        StageNoisePolicy.for_tier(
            PrivacyLevel.STANDARD, stage_count=0,
        )


def test_stage_policy_rejects_negative_stages():
    with pytest.raises(ValueError):
        StageNoisePolicy.for_tier(
            PrivacyLevel.STANDARD, stage_count=-1,
        )


def test_stage_policy_single_stage_takes_full_budget():
    """Single stage gets the full tier ε."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.STANDARD, stage_count=1,
    )
    assert policy.per_stage_epsilon == pytest.approx(8.0)


# ── ActivationDPInjector: per-stage injection ────────────


def test_injector_applies_noise_when_enabled():
    """Noise injection MUST change the tensor when enabled
    + epsilon is finite. Compare L2 norm of difference > 0."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.STANDARD, stage_count=4,
    )
    injector = ActivationDPInjector(policy=policy)
    activation = np.ones(100, dtype=np.float64)
    noised = injector.inject_stage(activation, stage_index=0)
    diff_norm = np.linalg.norm(noised - activation)
    assert diff_norm > 0.0


def test_injector_no_noise_when_disabled():
    """NONE tier passes through unchanged (modulo clipping
    for L2 norm normalization)."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.NONE, stage_count=4,
    )
    injector = ActivationDPInjector(policy=policy)
    activation = np.ones(100, dtype=np.float64) * 0.01
    noised = injector.inject_stage(activation, stage_index=0)
    np.testing.assert_array_almost_equal(noised, activation)


def test_injector_records_per_stage_epsilon():
    """Each inject_stage call records its ε in the trace.
    Sum across all stages should equal total budget."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.STANDARD, stage_count=4,
    )
    injector = ActivationDPInjector(policy=policy)
    for i in range(4):
        injector.inject_stage(
            np.ones(10, dtype=np.float64),
            stage_index=i,
        )
    trace = injector.trace()
    assert len(trace.per_stage_epsilon) == 4
    # Each stage gets ε=2.0 (8.0/4)
    for eps in trace.per_stage_epsilon:
        assert eps == pytest.approx(2.0)
    assert trace.total_epsilon_spent == pytest.approx(8.0)


def test_injector_tracks_dropped_no_noise_stage():
    """When NONE tier is active, the trace records ε=0 per
    stage but still tracks stage_index → caller can verify
    every stage was visited even if no noise applied."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.NONE, stage_count=4,
    )
    injector = ActivationDPInjector(policy=policy)
    for i in range(4):
        injector.inject_stage(
            np.ones(10, dtype=np.float64),
            stage_index=i,
        )
    trace = injector.trace()
    assert len(trace.per_stage_epsilon) == 4
    for eps in trace.per_stage_epsilon:
        assert eps == 0.0
    assert trace.total_epsilon_spent == 0.0


def test_injector_rejects_duplicate_stage_index():
    """A stage_index must not be injected twice — that would
    leak privacy (use the same noise budget twice for the
    same stage)."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.STANDARD, stage_count=4,
    )
    injector = ActivationDPInjector(policy=policy)
    injector.inject_stage(
        np.ones(10, dtype=np.float64), stage_index=0,
    )
    with pytest.raises(ValueError):
        injector.inject_stage(
            np.ones(10, dtype=np.float64), stage_index=0,
        )


def test_injector_rejects_out_of_range_stage_index():
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.STANDARD, stage_count=4,
    )
    injector = ActivationDPInjector(policy=policy)
    with pytest.raises(ValueError):
        injector.inject_stage(
            np.ones(10, dtype=np.float64), stage_index=4,
        )
    with pytest.raises(ValueError):
        injector.inject_stage(
            np.ones(10, dtype=np.float64), stage_index=-1,
        )


def test_injector_clips_l2_norm_per_stage():
    """Activation gets clipped to the policy's clip_norm
    before processing. Use NONE tier so the test isolates
    the clip step (no noise added) — output should have
    L2 ≈ clip_norm. Sensitivity bound is necessary for the
    DP guarantee."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.NONE, stage_count=2,
    )
    injector = ActivationDPInjector(policy=policy)
    # 100-element vector all 10.0 → L2 norm = 100; clip_norm
    # defaults to 1.0 → clipped L2 should be exactly 1.0
    activation = np.ones(100, dtype=np.float64) * 10.0
    clipped = injector.inject_stage(activation, stage_index=0)
    assert np.linalg.norm(clipped) == pytest.approx(
        1.0, abs=1e-9,
    )


def test_injector_no_clip_when_within_norm():
    """Activation below clip_norm passes through with NO
    rescaling under NONE tier."""
    policy = StageNoisePolicy.for_tier(
        PrivacyLevel.NONE, stage_count=2,
    )
    injector = ActivationDPInjector(policy=policy)
    # L2 norm = 0.1 < clip_norm=1.0 → no rescaling
    activation = np.ones(100, dtype=np.float64) * 0.01
    out = injector.inject_stage(activation, stage_index=0)
    np.testing.assert_array_almost_equal(out, activation)


# ── ActivationNoiseTrace round-trip ──────────────────────


def test_trace_to_dict():
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[2.0, 2.0, 2.0, 2.0],
        total_epsilon_spent=8.0,
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    d = trace.to_dict()
    assert d["per_stage_epsilon"] == [2.0, 2.0, 2.0, 2.0]
    assert d["total_epsilon_spent"] == 8.0
    assert d["stage_count"] == 4
    assert d["tier"] == "standard"


def test_trace_from_dict_round_trip():
    original = ActivationNoiseTrace(
        per_stage_epsilon=[1.0, 1.0],
        total_epsilon_spent=2.0,
        clip_norm=1.0,
        stage_count=2,
        tier="high",
    )
    d = original.to_dict()
    restored = ActivationNoiseTrace.from_dict(d)
    assert restored == original


# ── verify_activation_noise_trace ────────────────────────


def test_verify_trace_matches_tier_under_basic_composition():
    """Trace with 4 stages × ε=2.0 each → total ε=8.0
    matches STANDARD tier's claimed total."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[2.0, 2.0, 2.0, 2.0],
        total_epsilon_spent=8.0,
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.STANDARD,
    )
    assert ok is True
    assert reason == ""


def test_verify_trace_detects_over_spend():
    """Trace claims to be STANDARD but total ε=12 exceeds
    the tier's promised 8."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[3.0, 3.0, 3.0, 3.0],
        total_epsilon_spent=12.0,
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.STANDARD,
    )
    assert ok is False
    assert "exceed" in reason.lower() or "over" in reason.lower()


def test_verify_trace_detects_missing_stages():
    """stage_count=4 but per_stage_epsilon has only 3 entries
    → one stage was skipped (potential noise-skip attack)."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[2.0, 2.0, 2.0],
        total_epsilon_spent=6.0,
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.STANDARD,
    )
    assert ok is False
    assert (
        "stage" in reason.lower()
        and ("missing" in reason.lower()
             or "count" in reason.lower())
    )


def test_verify_trace_tier_none_allows_zero_epsilon():
    """NONE tier expects zero ε per stage. Trace honors
    that → ok."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[0.0, 0.0, 0.0, 0.0],
        total_epsilon_spent=0.0,
        clip_norm=1.0,
        stage_count=4,
        tier="none",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.NONE,
    )
    assert ok is True


def test_verify_trace_tier_mismatch():
    """Trace says STANDARD but verifier was told to expect
    HIGH → tier-claim mismatch."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[2.0, 2.0, 2.0, 2.0],
        total_epsilon_spent=8.0,
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.HIGH,
    )
    assert ok is False
    assert "tier" in reason.lower()


def test_verify_trace_negative_epsilon_rejected():
    """A negative ε would credit-back the budget. Defense
    against malicious trace forgery."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[2.0, 2.0, -1.0, 5.0],
        total_epsilon_spent=8.0,
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.STANDARD,
    )
    assert ok is False
    assert (
        "negative" in reason.lower()
        or "non-positive" in reason.lower()
        or "invalid" in reason.lower()
    )


def test_verify_trace_total_mismatch_with_sum():
    """total_epsilon_spent claimed but doesn't match sum
    of per-stage values → corruption."""
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[2.0, 2.0, 2.0, 2.0],
        total_epsilon_spent=999.0,  # lie
        clip_norm=1.0,
        stage_count=4,
        tier="standard",
    )
    ok, reason = verify_activation_noise_trace(
        trace, expected_tier=PrivacyLevel.STANDARD,
    )
    assert ok is False
    assert (
        "total" in reason.lower()
        or "mismatch" in reason.lower()
    )
