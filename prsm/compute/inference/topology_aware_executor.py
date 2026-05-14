"""Sprint 414 — TopologyAwareChainExecutor decorator.

Wraps an inner ``ChainExecutor`` and populates the sprint-
413 ``topology_assignment`` field on the returned
``ChainExecutionResult``. The assignment is a STRUCTURAL
record of which nodes handled which (stage, slot) — built
directly from the chain's ``stages`` list at the moment
of dispatch.

The assignment is verifiable: any party with the chain
definition can rebuild the same ``TopologyAssignment``
and check the ``stable_hash()`` matches the signed
receipt's claim.

DP injection (sprint 295 ``activation_noise_trace``) is
NOT covered here — it requires per-stage integration
inside the inner executor's dispatch loop, which can't be
done from a pure wrapper. That's a separate sprint
(``ActivationDPAwareChainExecutor`` is the natural
sibling).

## Composition

This decorator is intentionally narrow + composable.
Operator wiring at node-construction time:

    base = RpcChainExecutor(...)
    decorated = TopologyAwareChainExecutor(inner=base)
    # Later, when DP wiring lands:
    # decorated = ActivationDPAwareChainExecutor(
    #     inner=decorated, dp_policy=...,
    # )

Both decorators commute: a future DP-aware decorator
above this one preserves the topology_assignment via the
sprint-413 ``replace()`` pass-through pattern; below this
one, the topology decorator preserves any DP trace via
the same mechanism (pinned by test_inner_activation_
noise_trace_preserved).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from prsm.compute.inference.parallax_executor import (
    ChainExecutionResult,
)
from prsm.compute.inference.topology_rotation import (
    TopologyAssignment,
)


class TopologyAwareChainExecutor:
    """Wraps a ChainExecutor + records the topology in the
    outcome.

    Composition contract:
      - Calls ``inner.execute_chain(request=, chain=)``
        unchanged
      - Reads ``chain.stages`` after the call returns and
        builds a ``TopologyAssignment`` recording each
        stage's node assignment (1 slot per stage)
      - Returns the inner's result with
        ``topology_assignment`` populated via
        ``dataclasses.replace()``
      - If the inner already populated
        ``topology_assignment`` (e.g., upstream decorator),
        respects that pre-existing value — never clobbers

    Errors from the inner executor propagate unchanged —
    the decorator never swallows.
    """

    def __init__(self, *, inner: Any) -> None:
        if inner is None or not hasattr(inner, "execute_chain"):
            raise ValueError(
                "TopologyAwareChainExecutor requires an "
                "inner with .execute_chain(request=, chain=) "
                "method"
            )
        self._inner = inner

    def execute_chain(
        self, *, request: Any, chain: Any,
    ) -> ChainExecutionResult:
        # Pass through to inner — let any ChainExecutionError
        # propagate unchanged
        result = self._inner.execute_chain(
            request=request, chain=chain,
        )

        # If the inner already populated topology_assignment,
        # respect it
        if getattr(result, "topology_assignment", None) is not None:
            return result

        # Build the assignment from chain.stages
        positions = {
            (stage_index, 0): node_id
            for stage_index, node_id in enumerate(chain.stages)
        }
        topology = TopologyAssignment(
            positions=positions,
            stage_count=len(chain.stages),
            slots_per_stage=1,
        )

        return replace(result, topology_assignment=topology)
