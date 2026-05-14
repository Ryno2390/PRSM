"""Sprint 413 — ParallaxScheduledExecutor wires sprint-297
receipt privacy fields end-to-end.

Sprint 297 capstone added two Optional fields to
``InferenceReceipt``:

    activation_noise_trace   (sprint 295 DP per-stage trace)
    topology_assignment      (sprint 296 chain-rotation hash)

Plus the conditional signing-payload encoding so receipts
with these fields verify against settlers. But the live
inference path (ParallaxScheduledExecutor.execute) wasn't
populating them — they stayed None on every live receipt.

This sprint extends ``ChainExecutionResult`` with two
matching Optional fields, then has ``_build_signed_receipt``
thread them into the InferenceReceipt when the chain
executor populates them. Backwards-compat preserved:
ChainExecutors that don't populate the new fields produce
receipts identical to pre-sprint-413.
"""
from __future__ import annotations

from typing import Optional, Any

import pytest


# ── ChainExecutionResult schema extension ────────────────


def test_chain_execution_result_accepts_optional_privacy_fields():
    """Sprint 297 fields surface on the chain-executor's
    result shape so producers (e.g., a DP-aware
    RpcChainExecutor decorator) can pass them through."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.inference.activation_dp import (
        ActivationNoiseTrace,
    )

    trace = ActivationNoiseTrace(
        per_stage_epsilon=[0.1, 0.2, 0.1],
        total_epsilon_spent=0.4,
        clip_norm=1.0,
        stage_count=3,
        tier="standard",
    )
    topo = _dummy_topology()
    result = ChainExecutionResult(
        output="hello world",
        duration_seconds=0.5,
        tee_attestation=b"\x00" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.4,
        activation_noise_trace=trace,
        topology_assignment=topo,
    )
    assert result.activation_noise_trace is trace
    assert result.topology_assignment is topo


def test_chain_execution_result_privacy_fields_default_to_none():
    """Pre-sprint-413 callers omit the new fields entirely
    and get None — pure-additive contract."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )

    result = ChainExecutionResult(
        output="hello",
        duration_seconds=0.5,
        tee_attestation=b"\x00" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.0,
    )
    assert result.activation_noise_trace is None
    assert result.topology_assignment is None


# ── _build_signed_receipt wiring ─────────────────────────


@pytest.mark.asyncio
async def test_receipt_carries_activation_noise_trace_when_present():
    """When the chain executor returns a result with an
    activation_noise_trace, the signed receipt MUST carry
    it so verifiers can check DP application."""
    from prsm.compute.inference.activation_dp import (
        ActivationNoiseTrace,
    )
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )

    trace = ActivationNoiseTrace(
        per_stage_epsilon=[0.1, 0.2],
        total_epsilon_spent=0.3,
        clip_norm=1.0,
        stage_count=2,
        tier="zero-trust",
    )
    outcome = ChainExecutionResult(
        output="hello",
        duration_seconds=1.0,
        tee_attestation=b"\x01" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.3,
        activation_noise_trace=trace,
    )

    executor = _build_executor()
    receipt = executor._build_signed_receipt(
        request=_dummy_request(),
        cost=_dummy_cost(),
        outcome=outcome,
        streamed=False,
    )
    assert receipt.activation_noise_trace is trace


@pytest.mark.asyncio
async def test_receipt_carries_topology_assignment_when_present():
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )

    topology = _dummy_topology()
    outcome = ChainExecutionResult(
        output="hi",
        duration_seconds=0.5,
        tee_attestation=b"\x02" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.0,
        topology_assignment=topology,
    )

    executor = _build_executor()
    receipt = executor._build_signed_receipt(
        request=_dummy_request(),
        cost=_dummy_cost(),
        outcome=outcome,
        streamed=False,
    )
    assert receipt.topology_assignment == topology


@pytest.mark.asyncio
async def test_receipt_carries_both_fields_when_present():
    from prsm.compute.inference.activation_dp import (
        ActivationNoiseTrace,
    )
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )

    trace = ActivationNoiseTrace(
        per_stage_epsilon=[0.1],
        total_epsilon_spent=0.1,
        clip_norm=1.0,
        stage_count=1,
        tier="zero-trust",
    )
    topology = _dummy_topology()
    outcome = ChainExecutionResult(
        output="ok",
        duration_seconds=0.2,
        tee_attestation=b"\x03" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.1,
        activation_noise_trace=trace,
        topology_assignment=topology,
    )

    executor = _build_executor()
    receipt = executor._build_signed_receipt(
        request=_dummy_request(),
        cost=_dummy_cost(),
        outcome=outcome,
        streamed=False,
    )
    assert receipt.activation_noise_trace is trace
    assert receipt.topology_assignment == topology


# ── Backwards-compat ─────────────────────────────────────


@pytest.mark.asyncio
async def test_receipt_fields_remain_none_when_outcome_lacks_them():
    """Pre-sprint-413 ChainExecutors that don't populate the
    new fields produce receipts identical to before — both
    fields stay None. Critical backwards-compat pin."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )

    outcome = ChainExecutionResult(
        output="legacy",
        duration_seconds=0.1,
        tee_attestation=b"\x04" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.0,
    )

    executor = _build_executor()
    receipt = executor._build_signed_receipt(
        request=_dummy_request(),
        cost=_dummy_cost(),
        outcome=outcome,
        streamed=False,
    )
    assert receipt.activation_noise_trace is None
    assert receipt.topology_assignment is None


@pytest.mark.asyncio
async def test_signing_payload_unchanged_when_fields_none():
    """If the new fields are None, the signing payload must
    be byte-identical to a pre-sprint-413 receipt — settler
    signatures over pre-sprint-413 receipts MUST keep
    verifying. Sprint 297 pinned this via conditional
    encoding; this test pins it via the parallax path."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )

    outcome = ChainExecutionResult(
        output="no-privacy",
        duration_seconds=0.1,
        tee_attestation=b"\x05" * 64,
        tee_type=_dummy_tee(),
        epsilon_spent=0.0,
    )

    executor = _build_executor()
    receipt = executor._build_signed_receipt(
        request=_dummy_request(),
        cost=_dummy_cost(),
        outcome=outcome,
        streamed=False,
    )
    payload = receipt.signing_payload()
    # Both new field keys MUST be absent from the canonical
    # payload bytes when their values are None.
    assert b"activation_noise_trace" not in payload
    assert b"topology_assignment" not in payload


# ── helpers ──────────────────────────────────────────────


def _dummy_tee():
    from prsm.compute.tee.models import TEEType
    return TEEType.NONE


def _dummy_topology():
    from prsm.compute.inference.topology_rotation import (
        TopologyAssignment,
    )
    return TopologyAssignment(
        positions={(0, 0): "node-a", (1, 0): "node-b"},
        stage_count=2,
        slots_per_stage=1,
    )


def _dummy_request():
    from decimal import Decimal
    from prsm.compute.inference.models import (
        InferenceRequest, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    return InferenceRequest(
        prompt="hello",
        model_id="test-model",
        budget_ftns=Decimal("10.0"),
        request_id="req-123",
        max_tokens=10,
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.STANDARD,
    )


def _dummy_cost():
    from decimal import Decimal
    return Decimal("1.0")


def _build_executor():
    """Bypass ParallaxScheduledExecutor's heavy constructor
    by instantiating via ``__new__`` and setting only the
    fields ``_build_signed_receipt`` actually reads. The
    method only needs ``self._identity`` for sign_receipt;
    everything else comes from arguments."""
    from unittest.mock import MagicMock
    from prsm.compute.inference.parallax_executor import (
        ParallaxScheduledExecutor,
    )
    executor = ParallaxScheduledExecutor.__new__(
        ParallaxScheduledExecutor,
    )
    node_identity = MagicMock()
    node_identity.node_id = "test-node"
    node_identity.sign = MagicMock(return_value=b"\x00" * 64)
    executor._identity = node_identity
    return executor
