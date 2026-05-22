"""Sprint 702 — PRSM_PARALLAX_TIER_GATE=advisory bypass.

Live-attest of `privacy_tier=standard` against the NYC software-only
fleet was correctly rejected by Adapter B (TierGateAdapter): "no GPU
in pool has hardware-TEE attestation; required for privacy_level=
standard". This is the right production behavior.

But it blocks exercising the activation-DP injection code path
(sprints 295/413/414) in any environment that lacks real TEE
hardware. Sprint 702 mirrors sprint 686's stake-eligibility advisory
pattern: `PRSM_PARALLAX_TIER_GATE=advisory` lets non-TEE operators
serve tier ≥ standard requests so the DP injection path can be
live-attested.

Production posture is UNCHANGED — the default is enforced, typos
fall through to enforced (no accidental disabling), and a loud
WARNING fires on construction so operators don't silently run with
fake-TEE in prod.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_advisory_mode_passes_software_tier_attestations(monkeypatch):
    """PRSM_PARALLAX_TIER_GATE=advisory → tier_attestation=tier-none
    (software-only) GPUs pass through Adapter B for tier=standard."""
    monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "advisory")
    from prsm.node.inference_wiring import _wrap_tier_gate_for_advisory
    from prsm.compute.parallax_scheduling.trust_adapter import (
        TierGateAdapter,
    )
    from prsm.compute.parallax_scheduling.prsm_types import (
        ParallaxGPU, TIER_ATTESTATION_NONE,
    )
    from prsm.compute.tee.models import PrivacyLevel

    real = TierGateAdapter()
    wrapped = _wrap_tier_gate_for_advisory(real)
    assert wrapped is not real

    gpus = [ParallaxGPU(
        node_id="a" * 32, region="default", layer_capacity=12,
        stake_amount=1, tier_attestation=TIER_ATTESTATION_NONE,
        tflops_fp16=4.0, memory_gb=8.0, memory_bandwidth_gbps=50.0,
    )]
    # Real adapter would raise TierGateRejected for tier=STANDARD
    # with software-only attestation. Wrapper passes through.
    result = wrapped.filter(gpus, PrivacyLevel.STANDARD)
    assert len(result) == 1


def test_enforced_mode_unchanged(monkeypatch):
    """Default (env unset OR enforced) → returns real TierGateAdapter
    unwrapped. Production semantics preserved."""
    monkeypatch.delenv("PRSM_PARALLAX_TIER_GATE", raising=False)
    from prsm.node.inference_wiring import _wrap_tier_gate_for_advisory
    from prsm.compute.parallax_scheduling.trust_adapter import (
        TierGateAdapter,
    )
    real = TierGateAdapter()
    wrapped = _wrap_tier_gate_for_advisory(real)
    assert wrapped is real  # no wrapping


def test_advisory_logs_warning(monkeypatch, caplog):
    """Advisory mode must log loud WARNING so operators don't
    silently run with fake-TEE in prod."""
    monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "advisory")
    from prsm.node.inference_wiring import _wrap_tier_gate_for_advisory
    from prsm.compute.parallax_scheduling.trust_adapter import (
        TierGateAdapter,
    )
    import logging
    real = TierGateAdapter()
    with caplog.at_level(logging.WARNING):
        _wrap_tier_gate_for_advisory(real)
    assert any(
        "advisory" in r.message.lower() and "tier" in r.message.lower()
        for r in caplog.records
    )


def test_invalid_value_falls_through_to_enforced(monkeypatch):
    """PRSM_PARALLAX_TIER_GATE=garbage → enforced (safe default).
    Typo cannot accidentally disable production tier filtering."""
    monkeypatch.setenv("PRSM_PARALLAX_TIER_GATE", "garbage")
    from prsm.node.inference_wiring import _wrap_tier_gate_for_advisory
    from prsm.compute.parallax_scheduling.trust_adapter import (
        TierGateAdapter,
    )
    real = TierGateAdapter()
    wrapped = _wrap_tier_gate_for_advisory(real)
    assert wrapped is real


def test_tier_none_still_passes_in_enforced(monkeypatch):
    """Tier NONE is always allowed regardless of advisory env —
    pin the existing non-DP path stays unaffected."""
    monkeypatch.delenv("PRSM_PARALLAX_TIER_GATE", raising=False)
    from prsm.compute.parallax_scheduling.trust_adapter import (
        TierGateAdapter,
    )
    from prsm.compute.parallax_scheduling.prsm_types import (
        ParallaxGPU, TIER_ATTESTATION_NONE,
    )
    from prsm.compute.tee.models import PrivacyLevel
    real = TierGateAdapter()
    gpus = [ParallaxGPU(
        node_id="a" * 32, region="default", layer_capacity=12,
        stake_amount=1, tier_attestation=TIER_ATTESTATION_NONE,
        tflops_fp16=4.0, memory_gb=8.0, memory_bandwidth_gbps=50.0,
    )]
    # tier=NONE always passes
    result = real.filter(gpus, PrivacyLevel.NONE)
    assert len(result) == 1


def test_advisory_in_parallax_readiness_registry():
    """sprint 696's parallax-readiness CLI should list the new env
    var so operators see it in preflight output."""
    from prsm.cli import _PARALLAX_ENV_REGISTRY
    names = {row[0] for row in _PARALLAX_ENV_REGISTRY}
    assert "PRSM_PARALLAX_TIER_GATE" in names, (
        "sprint 702 must register PRSM_PARALLAX_TIER_GATE in the "
        "parallax-readiness CLI registry"
    )
