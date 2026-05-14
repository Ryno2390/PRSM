"""Sprint 419 — ActivationDPAwareChainExecutor decorator.

Sibling of sprint-414's TopologyAwareChainExecutor.
Threads sprint-295's ActivationDPInjector into the
dispatch path via sprint-418's post_stage_hook integration
point.

Per request:
  1. Build a StageNoisePolicy for the request's
     privacy_tier × chain length
  2. Build a fresh ActivationDPInjector (per-run state
     so double-spend protection works concurrently)
  3. Call inner.execute_chain with the injector wired as
     the post_stage_hook
  4. Read injector.trace() after the call
  5. Return result with activation_noise_trace populated

Tier NONE special-case: skip the hook entirely; receipt
carries no activation_noise_trace (semantically: "this
request didn't ask for DP, so no claim is made").
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest


def _request(*, privacy_tier=None):
    from prsm.compute.inference.models import (
        InferenceRequest, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    tier = privacy_tier or PrivacyLevel.STANDARD
    return InferenceRequest(
        prompt="dp-e2e",
        model_id="m",
        budget_ftns=Decimal("1.0"),
        privacy_tier=tier,
        content_tier=ContentTier.A,
    )


def _chain(stages):
    chain = MagicMock()
    chain.stages = list(stages)
    chain.layer_ranges = [(i, i + 1) for i in range(len(stages))]
    return chain


def _outcome():
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    return ChainExecutionResult(
        output="dp-out",
        duration_seconds=0.1,
        tee_attestation=b"\x00" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.0,
    )


def _inner_that_runs_hook(stages, *, outcome=None):
    """Mock inner that ACTUALLY invokes the post_stage_hook
    once per stage (simulating what RpcChainExecutor's
    dispatch loop does). Returns the supplied outcome."""
    if outcome is None:
        outcome = _outcome()

    def execute_chain(*, request, chain, post_stage_hook=None, **kwargs):
        activation = np.ones((4,), dtype=np.float32)
        for stage_index in range(len(chain.stages)):
            # Simulate per-stage activation update
            activation = activation * 2
            if post_stage_hook is not None:
                activation = post_stage_hook(activation, stage_index)
        return outcome

    inner = MagicMock()
    inner.execute_chain = MagicMock(side_effect=execute_chain)
    return inner


# ── Constructor ──────────────────────────────────────────


def test_constructor_rejects_inner_without_execute_chain():
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    with pytest.raises(ValueError, match="execute_chain"):
        ActivationDPAwareChainExecutor(inner=object())


def test_constructor_rejects_none():
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    with pytest.raises(ValueError):
        ActivationDPAwareChainExecutor(inner=None)


# ── Tier NONE: no DP applied, no trace ───────────────────


def test_tier_none_skips_dp_no_trace():
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    inner = _inner_that_runs_hook(["a", "b"])
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    result = decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.NONE),
        chain=_chain(["a", "b"]),
    )
    assert result.activation_noise_trace is None
    # Inner was called without a post_stage_hook
    call = inner.execute_chain.call_args
    assert call.kwargs.get("post_stage_hook") is None


# ── Non-NONE tier: hook wired + trace populated ──────────


def test_standard_tier_wires_hook_and_populates_trace():
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    inner = _inner_that_runs_hook(["a", "b", "c"])
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    result = decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.STANDARD),
        chain=_chain(["a", "b", "c"]),
    )
    assert result.activation_noise_trace is not None
    trace = result.activation_noise_trace
    # 3 stages → 3 per-stage epsilon entries
    assert trace.stage_count == 3
    assert len(trace.per_stage_epsilon) == 3
    # tier label propagated
    assert trace.tier == PrivacyLevel.STANDARD.value


def test_higher_tier_yields_higher_total_epsilon():
    """Per sprint 295: higher privacy tier (more strict
    privacy) → SMALLER ε. Standard < ZeroTrust ε."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    decorator = ActivationDPAwareChainExecutor(
        inner=_inner_that_runs_hook(["a", "b"]),
    )
    standard = decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.STANDARD),
        chain=_chain(["a", "b"]),
    )
    maximum = decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.MAXIMUM),
        chain=_chain(["a", "b"]),
    )
    # Both traces present
    assert standard.activation_noise_trace is not None
    assert maximum.activation_noise_trace is not None
    # MAXIMUM is stricter — smaller ε budget
    assert (
        maximum.activation_noise_trace.total_epsilon_spent
        < standard.activation_noise_trace.total_epsilon_spent
    )


# ── Hook fires correctly per stage ───────────────────────


def test_hook_called_once_per_stage_in_order():
    """Pinned: the decorator wires the injector such that
    each stage is injected exactly once, in order."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    seen_stages = []

    def execute_chain(*, request, chain, post_stage_hook=None, **kwargs):
        activation = np.ones((4,), dtype=np.float32)
        for stage_index in range(len(chain.stages)):
            if post_stage_hook is not None:
                # Record what stage_index the hook sees
                pre_hook = activation.copy()
                activation = post_stage_hook(activation, stage_index)
                seen_stages.append(stage_index)
                # Hook returns a clipped + possibly-noised
                # activation
                assert activation.shape == pre_hook.shape
        return _outcome()

    inner = MagicMock()
    inner.execute_chain = MagicMock(side_effect=execute_chain)

    decorator = ActivationDPAwareChainExecutor(inner=inner)
    decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.STANDARD),
        chain=_chain(["a", "b", "c"]),
    )
    assert seen_stages == [0, 1, 2]


# ── Pass-through of inner result fields ──────────────────


def test_inner_result_fields_preserved():
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType, PrivacyLevel

    distinct_outcome = ChainExecutionResult(
        output="preserve-me",
        duration_seconds=99.0,
        tee_attestation=b"\xde" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.42,
    )
    inner = _inner_that_runs_hook(["a"], outcome=distinct_outcome)
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    result = decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.STANDARD),
        chain=_chain(["a"]),
    )
    assert result.output == "preserve-me"
    assert result.duration_seconds == 99.0
    assert result.epsilon_spent == 0.42


def test_inner_topology_assignment_preserved():
    """Composition with sprint-414 decorator: if inner
    already populated topology_assignment, the DP decorator
    MUST forward it through unchanged."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.inference.topology_rotation import (
        TopologyAssignment,
    )
    from prsm.compute.tee.models import TEEType, PrivacyLevel

    topo = TopologyAssignment(
        positions={(0, 0): "x"},
        stage_count=1,
        slots_per_stage=1,
    )
    inner_outcome = ChainExecutionResult(
        output="composed",
        duration_seconds=1.0,
        tee_attestation=b"\x00" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.0,
        topology_assignment=topo,
    )
    inner = _inner_that_runs_hook(["x"], outcome=inner_outcome)
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    result = decorator.execute_chain(
        request=_request(privacy_tier=PrivacyLevel.STANDARD),
        chain=_chain(["x"]),
    )
    # DP trace added
    assert result.activation_noise_trace is not None
    # Topology preserved
    assert result.topology_assignment is topo


# ── Error propagation ────────────────────────────────────


def test_inner_exceptions_propagate():
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    inner = MagicMock()
    inner.execute_chain = MagicMock(
        side_effect=RuntimeError("inner boom"),
    )
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    with pytest.raises(RuntimeError, match="inner boom"):
        decorator.execute_chain(
            request=_request(privacy_tier=PrivacyLevel.STANDARD),
            chain=_chain(["a"]),
        )


# ── Conflict detection: caller already passed a hook ─────


def test_rejects_conflicting_post_stage_hook_in_kwargs():
    """If the caller passes their own post_stage_hook,
    the decorator can't safely compose (DP injection
    must be the only mutator of activations). Surface a
    clear error rather than silently overriding."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    inner = _inner_that_runs_hook(["a"])
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    with pytest.raises(
        ValueError, match="post_stage_hook",
    ):
        decorator.execute_chain(
            request=_request(privacy_tier=PrivacyLevel.STANDARD),
            chain=_chain(["a"]),
            post_stage_hook=lambda a, i: a,
        )


# ── Double-spend protection via the injector ─────────────


def test_double_inject_attempt_propagates_value_error():
    """The injector's double-spend defense — if the inner
    executor calls the hook twice with the same stage_index
    (e.g., a malicious or buggy inner), the ValueError
    propagates."""
    from prsm.compute.inference.activation_dp_aware_executor import (
        ActivationDPAwareChainExecutor,
    )
    from prsm.compute.tee.models import PrivacyLevel

    def double_inject(*, request, chain, post_stage_hook=None, **kwargs):
        activation = np.ones((4,), dtype=np.float32)
        if post_stage_hook is not None:
            post_stage_hook(activation, 0)
            post_stage_hook(activation, 0)  # double-inject same stage
        return _outcome()

    inner = MagicMock()
    inner.execute_chain = MagicMock(side_effect=double_inject)
    decorator = ActivationDPAwareChainExecutor(inner=inner)
    with pytest.raises(ValueError, match="double-spend|already"):
        decorator.execute_chain(
            request=_request(privacy_tier=PrivacyLevel.STANDARD),
            chain=_chain(["a", "b"]),  # policy.stage_count=2
        )
