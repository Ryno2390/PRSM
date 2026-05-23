"""Sprint 779 — PreemptionAwareChainExecutor decorator.

Wraps an inner ChainExecutor + populates sprint-777's
`partial_completion` field on the returned ChainExecutionResult
when the sprint-772 preemption detector flag transitions from
clear to set DURING the inner .execute_chain() call.

This is the first concrete PRODUCER of partial-completion
markers. Sprints 772-778 built the detection + gates + wire
format + scheduler plumbing; sprint 779 actually constructs the
marker from runtime state.

## Composition

Mirrors sprint 414's TopologyAwareChainExecutor pattern.
Operator wiring at node-construction time:

    base = RpcChainExecutor(...)
    base = TopologyAwareChainExecutor(inner=base)
    base = PreemptionAwareChainExecutor(inner=base)
    # ...further decorators above this one preserve partial_
    # completion via the dataclasses.replace pass-through

Decorator contract:
- Calls inner.execute_chain(request=, chain=, **kwargs) unchanged
- If inner already populated partial_completion (e.g. upstream
  decorator) → respects that pre-existing value, never clobbers
- Detects preemption ONLY when the flag transitions from clear
  to set DURING the call. Steady-state preempted from before the
  call is sprint 774's dispatch-gate territory; reaching here
  with the flag already set means either the gate wasn't wired
  or the request bypassed dispatch — don't double-mark.
- Inner errors propagate unchanged. The decorator never swallows.

## Approximate token-counting

tokens_completed is computed from `len(result.output.split())`
(whitespace-token count). This is a deterministic proxy that:
- Is verifiable from the receipt's output_hash (settlement-side
  can rebuild the same count from the output text the receipt
  commits to)
- Doesn't require loading a model-specific tokenizer in the
  decorator path
- Slightly under-counts vs real tokenizers (which split BPE
  subwords) — which is conservative from a credit-claim
  perspective (operator can never claim MORE than honest)

A future sprint can replace this with a tokenizer-aware count
if settlement-side requires sub-token granularity.

## Streaming path

execute_chain_streaming is a pure pass-through — per-token frames
have no single ChainExecutionResult to mark. The streaming caller
is responsible for noticing preemption mid-stream (sprint 780+).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
)
from prsm.compute.inference.partial_completion import (
    PartialCompletionInfo,
)


def _iso_utc_now() -> str:
    """ISO-8601 UTC timestamp with 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _word_count(s: str) -> int:
    return len((s or "").split())


class PreemptionAwareChainExecutor:
    """Wraps a ChainExecutor + marks partial-completion on the
    outcome when preemption fires mid-execution."""

    def __init__(self, *, inner: Any) -> None:
        if inner is None or not hasattr(inner, "execute_chain"):
            raise ValueError(
                "PreemptionAwareChainExecutor requires an "
                "inner with .execute_chain(request=, chain=) "
                "method"
            )
        self._inner = inner

    def execute_chain(
        self, *, request: Any, chain: Any, **kwargs: Any,
    ) -> ChainExecutionResult:
        # Lazy import keeps decorator usable in tests that don't
        # have the daemon's preemption module wired (the import
        # itself is cheap but the helper is module-state-aware).
        from prsm.node.preemption import is_currently_preempted

        was_preempted_before = is_currently_preempted()

        result = self._inner.execute_chain(
            request=request, chain=chain, **kwargs,
        )

        # Respect upstream decorator's annotation
        if getattr(result, "partial_completion", None) is not None:
            return result

        is_preempted_now = is_currently_preempted()
        if was_preempted_before or not is_preempted_now:
            # Either steady-state preempted (sprint 774's job) or
            # never preempted (full success). Either way, no
            # marker.
            return result

        # Flag transitioned clear→set during inner execution.
        tokens_completed = _word_count(result.output)
        tokens_requested = getattr(
            request, "max_tokens", tokens_completed,
        )
        info = PartialCompletionInfo(
            reason="preempted",
            tokens_completed=tokens_completed,
            tokens_requested=tokens_requested,
            timestamp=_iso_utc_now(),
        )
        return replace(result, partial_completion=info)

    def execute_chain_streaming(
        self, *, request: Any, chain: Any, **kwargs: Any,
    ) -> Any:
        """Pass-through. Per-token frames have no receipt to mark.
        The streaming caller is responsible for noticing
        preemption mid-stream + producing a marker at receipt-
        build time."""
        if not hasattr(self._inner, "execute_chain_streaming"):
            raise AttributeError(
                "PreemptionAwareChainExecutor."
                "execute_chain_streaming requires the inner "
                "executor to support streaming"
            )
        return self._inner.execute_chain_streaming(
            request=request, chain=chain, **kwargs,
        )
