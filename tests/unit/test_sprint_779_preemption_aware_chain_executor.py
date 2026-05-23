"""Sprint 779 — PreemptionAwareChainExecutor decorator.

Sprint 778 wired the partial_completion CARRIER. Sprint 779
ships the first concrete PRODUCER: a chain-executor decorator
that observes `is_currently_preempted()` around the inner
.execute_chain() call. If the flag transitions clear→set during
execution, the decorator marks the outcome with a
PartialCompletionInfo so the resulting signed receipt records:
  - reason="preempted"
  - tokens_completed=N    (whitespace-split count from output)
  - tokens_requested=R    (from request.max_tokens, default N)
  - timestamp=ISO-UTC-now

Mirrors sprint 414's TopologyAwareChainExecutor decorator
pattern. Composition contract:
- If inner already set partial_completion → respect (never
  clobber upstream decorator's annotation)
- Detects preemption ONLY when it fires DURING this call
  (clear-at-start + set-at-end). Steady-state preempted from
  before the call is sprint 774's dispatch-gate territory.
- Streaming path is a pass-through (per-token frames have no
  receipt to mark)

Pin tests:
- Class exists with `execute_chain` method
- Inner already populated → passthrough (no clobber)
- No preemption → inner result unchanged
- Preemption during execution → marker added (correct fields)
- Pre-call preempted state → no marker (steady-state covered by
  sprint 774, not 779)
- Inner errors propagate (decorator never swallows)
- tokens_completed reflects output word-count
- tokens_requested reflects request.max_tokens
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_outcome(output="hello world", **overrides):
    from prsm.compute.inference.parallax_executor import (
        ChainExecutionResult,
    )
    from prsm.compute.tee.models import TEEType
    defaults = dict(
        output=output,
        duration_seconds=1.0,
        tee_attestation=b"att",
        tee_type=TEEType.SOFTWARE,
        epsilon_spent=0.0,
    )
    defaults.update(overrides)
    return ChainExecutionResult(**defaults)


def _make_inner(returns_outcome):
    inner = MagicMock()
    inner.execute_chain.return_value = returns_outcome
    return inner


def _req(max_tokens=10):
    r = MagicMock()
    r.max_tokens = max_tokens
    return r


# ---- Class shape ------------------------------------------------


def test_class_exists():
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    assert hasattr(PreemptionAwareChainExecutor, "execute_chain")


def test_init_rejects_inner_without_execute_chain():
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    import pytest
    bad = object()  # no execute_chain
    with pytest.raises(ValueError):
        PreemptionAwareChainExecutor(inner=bad)


# ---- Composition: respect upstream decorators ------------------


def test_inner_partial_completion_preserved():
    """If a higher decorator already set partial_completion,
    sprint 779 must NOT clobber it."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    from prsm.compute.inference.partial_completion import (
        PartialCompletionInfo,
    )
    info = PartialCompletionInfo(
        reason="timeout",
        tokens_completed=3,
        tokens_requested=10,
        timestamp="2026-05-23T11:00:00Z",
    )
    inner_result = _make_outcome(partial_completion=info)
    dec = PreemptionAwareChainExecutor(
        inner=_make_inner(inner_result),
    )
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        return_value=True,
    ):
        result = dec.execute_chain(
            request=_req(10), chain=MagicMock(),
        )
    assert result.partial_completion is info  # not replaced


# ---- Happy path: no preemption -------------------------------


def test_no_preemption_no_marker():
    """Inner ran cleanly + flag never flipped → partial_completion
    stays None."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    inner_result = _make_outcome(output="hello world")
    dec = PreemptionAwareChainExecutor(
        inner=_make_inner(inner_result),
    )
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        return_value=False,
    ):
        result = dec.execute_chain(
            request=_req(10), chain=MagicMock(),
        )
    assert result.partial_completion is None


# ---- Preemption DURING execution ------------------------------


def test_preemption_during_execution_marks_outcome():
    """Flag clear at call-start, set by call-end → marker added."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    inner_result = _make_outcome(output="The capital of France is the")
    dec = PreemptionAwareChainExecutor(
        inner=_make_inner(inner_result),
    )
    # Two-call pattern: first call (before inner) returns False,
    # second call (after inner) returns True
    flag_values = iter([False, True])
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        side_effect=lambda: next(flag_values),
    ):
        result = dec.execute_chain(
            request=_req(10), chain=MagicMock(),
        )
    assert result.partial_completion is not None
    assert result.partial_completion.reason == "preempted"
    # 6 words in "The capital of France is the"
    assert result.partial_completion.tokens_completed == 6
    assert result.partial_completion.tokens_requested == 10
    # ISO-8601 UTC timestamp
    assert result.partial_completion.timestamp.endswith("Z")


def test_preempted_already_at_start_no_new_marker():
    """Steady-state preempted (flag set BEFORE call too) — sprint
    774 dispatch gate should have already 503'd; if we're here,
    don't double-mark. Decorator only marks the clear→set
    transition that happens during execution."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    inner_result = _make_outcome()
    dec = PreemptionAwareChainExecutor(
        inner=_make_inner(inner_result),
    )
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        return_value=True,  # set both before AND after
    ):
        result = dec.execute_chain(
            request=_req(10), chain=MagicMock(),
        )
    assert result.partial_completion is None


# ---- Error propagation ---------------------------------------


def test_inner_errors_propagate():
    """Decorator never swallows exceptions from inner."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    import pytest
    inner = MagicMock()
    inner.execute_chain.side_effect = RuntimeError("boom")
    dec = PreemptionAwareChainExecutor(inner=inner)
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        return_value=False,
    ), pytest.raises(RuntimeError, match="boom"):
        dec.execute_chain(request=_req(10), chain=MagicMock())


# ---- Streaming pass-through ----------------------------------


def test_streaming_passthrough():
    """execute_chain_streaming forwards to inner unchanged."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    inner = MagicMock()
    sentinel = iter(["tok1", "tok2"])
    inner.execute_chain_streaming.return_value = sentinel
    dec = PreemptionAwareChainExecutor(inner=inner)
    out = dec.execute_chain_streaming(
        request=_req(10), chain=MagicMock(),
    )
    assert out is sentinel
    inner.execute_chain_streaming.assert_called_once()


# ---- tokens_requested fallback ------------------------------


def test_tokens_requested_falls_back_to_completed_when_max_tokens_missing():
    """Some requests don't carry max_tokens (older clients).
    Fallback: tokens_requested = tokens_completed (no partial-
    completion signal but the marker still records the event)."""
    from prsm.compute.inference.preemption_aware_executor import (
        PreemptionAwareChainExecutor,
    )
    inner_result = _make_outcome(output="hello world")
    dec = PreemptionAwareChainExecutor(
        inner=_make_inner(inner_result),
    )
    req = MagicMock(spec=[])  # no max_tokens attribute
    flag_values = iter([False, True])
    with patch(
        "prsm.node.preemption.is_currently_preempted",
        side_effect=lambda: next(flag_values),
    ):
        result = dec.execute_chain(request=req, chain=MagicMock())
    assert result.partial_completion is not None
    assert result.partial_completion.tokens_completed == 2
    assert result.partial_completion.tokens_requested == 2
