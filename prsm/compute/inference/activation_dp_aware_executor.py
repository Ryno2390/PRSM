"""Sprint 419 — ActivationDPAwareChainExecutor decorator.

Sibling of sprint 414's ``TopologyAwareChainExecutor``.
Threads sprint-295's ``ActivationDPInjector`` into the
dispatch path via sprint-418's ``post_stage_hook``
integration point.

## Soundness

Unlike the topology decorator (which records a STRUCTURAL
fact post-dispatch), DP must MUTATE the data path:
calibrated Gaussian noise is added to each stage's
activation before it's passed to the next stage. The
``activation_noise_trace`` on the resulting receipt
faithfully records:

  - per-stage ε actually allocated (and applied)
  - total ε spent across the chain
  - clip norm used for sensitivity bounding
  - the chain's stage count
  - the privacy tier label

The injector's ε accounting matches what was actually
applied — no "shadow trace" failure mode where the
receipt claims DP that wasn't applied.

## Composition

  base = RpcChainExecutor(...)
  with_topology = TopologyAwareChainExecutor(inner=base)
  with_dp = ActivationDPAwareChainExecutor(inner=with_topology)

Both decorators commute: the topology decorator does
``**kwargs`` pass-through (sprint 418), so the DP
decorator's ``post_stage_hook`` reaches the base
RpcChainExecutor's dispatch loop verbatim.

## Tier NONE special case

When ``request.privacy_tier`` is ``NONE``, the policy
constructor returns ``enabled=False``. The decorator
skips installing the hook entirely AND skips populating
the trace. Semantic: this request didn't ask for DP, so
no claim is made on the receipt. Avoids the misleading-
trace failure where a receipt carries a trace for a
request that legitimately didn't want DP.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from prsm.compute.inference.activation_dp import (
    ActivationDPInjector,
    StageNoisePolicy,
)
from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
)


class ActivationDPAwareChainExecutor:
    """Wraps a ChainExecutor + applies sprint-295 DP
    injection on every activation transfer between stages.

    Per request:
      1. Build StageNoisePolicy for (privacy_tier, len(stages))
      2. Build a fresh ActivationDPInjector
      3. Call inner.execute_chain with the injector wired
         as post_stage_hook
      4. Read injector.trace() after the call
      5. Return result with activation_noise_trace populated

    Constructor parameters:
      inner       any object exposing execute_chain(request=,
                  chain=, post_stage_hook=, ...). Must accept
                  the sprint-418 post_stage_hook kwarg.
      clip_norm   L2 clip applied to activations before
                  noise (sensitivity bound). Default 1.0.
      delta       δ parameter for Gaussian DP. Default 1e-5.

    Conflict-detection:
      If a caller passes ``post_stage_hook=`` in the
      execute_chain call, ValueError is raised. DP
      injection must be the only mutator of activations
      to keep the receipt's claim sound — silently
      composing with another hook would risk subtle
      ordering / mutation bugs.
    """

    def __init__(
        self,
        *,
        inner: Any,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
    ) -> None:
        if inner is None or not hasattr(inner, "execute_chain"):
            raise ValueError(
                "ActivationDPAwareChainExecutor requires an "
                "inner with .execute_chain(request=, chain=, "
                "post_stage_hook=, ...) method"
            )
        self._inner = inner
        self._clip_norm = float(clip_norm)
        self._delta = float(delta)

    def execute_chain(
        self,
        *,
        request: Any,
        chain: Any,
        **kwargs: Any,
    ) -> ChainExecutionResult:
        if "post_stage_hook" in kwargs:
            raise ValueError(
                "ActivationDPAwareChainExecutor cannot compose "
                "with a caller-supplied post_stage_hook — DP "
                "injection must be the only activation mutator "
                "to keep the receipt's privacy claim sound"
            )

        stage_count = len(chain.stages)
        policy = StageNoisePolicy.for_tier(
            request.privacy_tier,
            stage_count,
            clip_norm=self._clip_norm,
            delta=self._delta,
        )

        # Tier NONE: skip the hook entirely; no trace.
        # Semantic: request didn't ask for DP, no claim made.
        if not policy.enabled:
            return self._inner.execute_chain(
                request=request, chain=chain, **kwargs,
            )

        injector = ActivationDPInjector(policy)
        result = self._inner.execute_chain(
            request=request,
            chain=chain,
            post_stage_hook=injector.inject_stage,
            **kwargs,
        )
        trace = injector.trace()
        return replace(result, activation_noise_trace=trace)

    def execute_chain_streaming(
        self, *, request: Any, chain: Any, **kwargs: Any,
    ) -> Any:
        """Sprint 689 — passthrough to inner for SSE streaming.

        For tier NONE the DP injector wouldn't fire even on the
        unary path (policy.enabled=False), so streaming passes
        through cleanly. For tier STANDARD/HIGH/MAXIMUM we
        currently can't combine activation-DP injection with
        per-token streaming — DP injection runs once-per-stage
        but streaming yields frames per token; integrating them
        requires a streaming-aware injector that maintains state
        across token frames. Raise a structured error pointing
        at the gap rather than silently dropping DP.
        """
        if not hasattr(self._inner, "execute_chain_streaming"):
            raise AttributeError(
                "ActivationDPAwareChainExecutor.execute_chain_streaming "
                "requires the inner executor to support streaming"
            )
        stage_count = len(chain.stages)
        policy = StageNoisePolicy.for_tier(
            request.privacy_tier,
            stage_count,
            clip_norm=self._clip_norm,
            delta=self._delta,
        )
        if policy.enabled:
            raise RuntimeError(
                f"ActivationDPAwareChainExecutor: privacy_tier="
                f"{request.privacy_tier!r} requires DP injection, "
                f"but DP-aware streaming isn't yet wired. Use "
                f"unary /compute/inference for non-NONE tiers, "
                f"or NONE tier for streaming."
            )
        return self._inner.execute_chain_streaming(
            request=request, chain=chain, **kwargs,
        )
