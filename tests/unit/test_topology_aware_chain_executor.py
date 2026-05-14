"""Sprint 414 — TopologyAwareChainExecutor decorator.

Wraps an inner ChainExecutor and populates
``outcome.topology_assignment`` from the chain's actual
dispatch structure. The assignment is a STRUCTURAL fact
about which nodes handled which (stage, slot) — verifiable
by any party that has the chain definition.

Sound semantics: the decorator records the topology the
inner executor ACTUALLY used. There's no risk of a
misleading claim — a verifier can rebuild the same
TopologyAssignment from the chain.stages and check the
stable_hash() matches.

DP injection (sprint 295 activation_noise_trace) is NOT
covered by this decorator — it needs per-stage integration
inside the inner executor's dispatch loop. Deferred to a
follow-on sprint.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── Helpers ──────────────────────────────────────────────


def _chain(stages):
    """Minimal GPUChain-like with stages + layer_ranges."""
    chain = MagicMock()
    chain.stages = list(stages)
    chain.layer_ranges = [(i, i + 1) for i in range(len(stages))]
    return chain


def _outcome_with(*, output="ok"):
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    return ChainExecutionResult(
        output=output,
        duration_seconds=0.1,
        tee_attestation=b"\x00" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.0,
    )


def _inner_executor(outcome=None):
    """A mock inner ChainExecutor whose execute_chain returns
    a fixed ChainExecutionResult (default: no topology)."""
    if outcome is None:
        outcome = _outcome_with()
    inner = MagicMock()
    inner.execute_chain = MagicMock(return_value=outcome)
    return inner


def _request():
    from decimal import Decimal
    from prsm.compute.inference.models import (
        InferenceRequest, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel
    return InferenceRequest(
        prompt="hi",
        model_id="m",
        budget_ftns=Decimal("1.0"),
        privacy_tier=PrivacyLevel.STANDARD,
        content_tier=ContentTier.A,
    )


# ── Constructor ──────────────────────────────────────────


def test_constructor_rejects_inner_without_execute_chain():
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    with pytest.raises(ValueError, match="execute_chain"):
        TopologyAwareChainExecutor(inner=object())


def test_constructor_rejects_none():
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    with pytest.raises(ValueError):
        TopologyAwareChainExecutor(inner=None)


# ── Topology population ──────────────────────────────────


def test_assignment_records_actual_stages():
    """The assignment's positions dict reflects what the
    chain's stages list says — one slot per stage."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    inner = _inner_executor()
    decorator = TopologyAwareChainExecutor(inner=inner)
    chain = _chain(["node-a", "node-b", "node-c"])
    result = decorator.execute_chain(request=_request(), chain=chain)
    assert result.topology_assignment is not None
    topo = result.topology_assignment
    assert topo.stage_count == 3
    assert topo.slots_per_stage == 1
    assert topo.positions == {
        (0, 0): "node-a",
        (1, 0): "node-b",
        (2, 0): "node-c",
    }


def test_assignment_stable_hash_deterministic_for_same_stages():
    """Two requests with identical stage lists produce
    identical stable_hash() values — gives verifiers a
    deterministic check."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    chain = _chain(["x", "y"])
    d1 = TopologyAwareChainExecutor(inner=_inner_executor())
    d2 = TopologyAwareChainExecutor(inner=_inner_executor())
    r1 = d1.execute_chain(request=_request(), chain=chain)
    r2 = d2.execute_chain(request=_request(), chain=chain)
    assert (
        r1.topology_assignment.stable_hash()
        == r2.topology_assignment.stable_hash()
    )


def test_assignment_hash_changes_when_stages_differ():
    """Different chains MUST produce different hashes —
    the assignment isn't accidentally constant."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    d = TopologyAwareChainExecutor(inner=_inner_executor())
    r1 = d.execute_chain(
        request=_request(), chain=_chain(["a", "b"]),
    )
    r2 = d.execute_chain(
        request=_request(), chain=_chain(["a", "c"]),
    )
    assert (
        r1.topology_assignment.stable_hash()
        != r2.topology_assignment.stable_hash()
    )


# ── Pass-through of inner result fields ──────────────────


def test_inner_result_fields_preserved():
    """The decorator MUST NOT mutate inner.output /
    duration_seconds / tee_attestation / tee_type /
    epsilon_spent — only adds topology_assignment."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    inner_outcome = ChainExecutionResult(
        output="distinctive-output",
        duration_seconds=42.5,
        tee_attestation=b"\xab" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.7,
    )
    decorator = TopologyAwareChainExecutor(
        inner=_inner_executor(outcome=inner_outcome),
    )
    result = decorator.execute_chain(
        request=_request(), chain=_chain(["n1"]),
    )
    assert result.output == "distinctive-output"
    assert result.duration_seconds == 42.5
    assert result.tee_attestation == b"\xab" * 64
    assert result.epsilon_spent == 0.7


def test_inner_activation_noise_trace_preserved():
    """If the inner executor populated
    activation_noise_trace (e.g., a future DP-aware
    decorator below this one), the topology decorator
    MUST forward it through unchanged."""
    from prsm.compute.inference.activation_dp import (
        ActivationNoiseTrace,
    )
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    trace = ActivationNoiseTrace(
        per_stage_epsilon=[0.05, 0.05],
        total_epsilon_spent=0.1,
        clip_norm=1.0,
        stage_count=2,
        tier="zero-trust",
    )
    inner_outcome = ChainExecutionResult(
        output="dp",
        duration_seconds=1.0,
        tee_attestation=b"\xcc" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.1,
        activation_noise_trace=trace,
    )
    decorator = TopologyAwareChainExecutor(
        inner=_inner_executor(outcome=inner_outcome),
    )
    result = decorator.execute_chain(
        request=_request(), chain=_chain(["n1", "n2"]),
    )
    assert result.activation_noise_trace is trace


# ── Inner execute_chain still called with original args ──


def test_inner_executor_called_with_original_args():
    """Decorator is transparent at the inner-call boundary —
    no modification of request or chain."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    inner = _inner_executor()
    decorator = TopologyAwareChainExecutor(inner=inner)
    req = _request()
    chain = _chain(["x", "y", "z"])
    decorator.execute_chain(request=req, chain=chain)
    inner.execute_chain.assert_called_once_with(
        request=req, chain=chain,
    )


# ── Error propagation ────────────────────────────────────


def test_inner_exceptions_propagate():
    """The decorator doesn't swallow inner errors — a
    ChainExecutionError raised by the inner must reach
    the caller unchanged."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    inner = MagicMock()
    inner.execute_chain = MagicMock(
        side_effect=RuntimeError("inner failure"),
    )
    decorator = TopologyAwareChainExecutor(inner=inner)
    with pytest.raises(RuntimeError, match="inner failure"):
        decorator.execute_chain(
            request=_request(), chain=_chain(["x"]),
        )


# ── Empty / edge cases ───────────────────────────────────


def test_single_stage_chain():
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    decorator = TopologyAwareChainExecutor(
        inner=_inner_executor(),
    )
    result = decorator.execute_chain(
        request=_request(), chain=_chain(["solo"]),
    )
    assert result.topology_assignment.stage_count == 1
    assert result.topology_assignment.positions == {
        (0, 0): "solo",
    }


def test_passes_through_arbitrary_kwargs_to_inner():
    """Sprint 418 — TopologyAwareChainExecutor must
    pass-through arbitrary kwargs (like post_stage_hook)
    to the inner executor so a future DP-aware sibling
    decorator can stack above us and thread its hook
    down to the RpcChainExecutor."""
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    inner = _inner_executor()
    decorator = TopologyAwareChainExecutor(inner=inner)
    sentinel_hook = lambda a, i: a  # noqa: E731
    decorator.execute_chain(
        request=_request(),
        chain=_chain(["n1"]),
        post_stage_hook=sentinel_hook,
    )
    # Inner saw the kwarg verbatim
    call = inner.execute_chain.call_args
    assert call.kwargs.get("post_stage_hook") is sentinel_hook


def test_does_not_overwrite_inner_topology_assignment():
    """If the inner executor (e.g., a future DP-aware
    decorator that also computes a topology) already
    populated topology_assignment, this decorator MUST NOT
    clobber it. Pre-existing assignment wins."""
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.inference.topology_rotation import (
        TopologyAssignment,
    )
    from prsm.compute.tee.models import TEEType
    from prsm.compute.inference.topology_aware_executor import (
        TopologyAwareChainExecutor,
    )
    pre_existing = TopologyAssignment(
        positions={(0, 0): "from-inner"},
        stage_count=1,
        slots_per_stage=1,
    )
    inner_outcome = ChainExecutionResult(
        output="pre-set",
        duration_seconds=0.1,
        tee_attestation=b"\x00" * 64,
        tee_type=TEEType.NONE,
        epsilon_spent=0.0,
        topology_assignment=pre_existing,
    )
    decorator = TopologyAwareChainExecutor(
        inner=_inner_executor(outcome=inner_outcome),
    )
    # Even with a stages list that would normally produce a
    # different assignment, the decorator respects what's
    # already there.
    result = decorator.execute_chain(
        request=_request(),
        chain=_chain(["different-node-1", "different-node-2"]),
    )
    assert result.topology_assignment is pre_existing
